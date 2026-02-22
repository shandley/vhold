#!/usr/bin/env python3
"""Build Swiss-Prot reference database for FANTASIA-style GO term transfer.

Downloads Swiss-Prot (reviewed UniProt), filters to proteins with experimental
GO annotations, extracts ProstT5 embeddings, and outputs .npz + metadata JSON.

Supports two scopes:
- --scope viral: ~17K reviewed viral proteins (fast, for validation)
- --scope all:   ~570K reviewed Swiss-Prot proteins (cross-domain discovery)

Usage:
    # Viral scope (fast, ~30 min on A100)
    python scripts/build_swissprot_reference.py \
        --scope viral --output ~/.vhold/databases/swissprot/ \
        --device cuda --batch-size 32

    # Full Swiss-Prot (cross-domain, ~16 hrs on A100)
    python scripts/build_swissprot_reference.py \
        --scope all --output ~/.vhold/databases/swissprot/ \
        --device cuda --batch-size 32 --checkpoint-interval 1000

    # Resume interrupted run
    python scripts/build_swissprot_reference.py \
        --scope all --output ~/.vhold/databases/swissprot/ \
        --device cuda --batch-size 32 --resume
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vhold.databases.swissprot import GOAnnotation, SwissProtEntry
from vhold.utils.constants import (
    EMBEDDING_DIM,
    EXPERIMENTAL_EVIDENCE_CODES,
    PROSTT5_MODEL_NAME,
    SWISSPROT_ALL_EMBEDDINGS,
    SWISSPROT_ALL_METADATA,
    SWISSPROT_VIRAL_EMBEDDINGS,
    SWISSPROT_VIRAL_METADATA,
    get_model_dir,
)

# UniProt XML namespace
UNIPROT_NS = "{http://uniprot.org/uniprot}"

# Download URLs
SWISSPROT_XML_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz"


def download_swissprot(output_path: Path) -> Path:
    """Download Swiss-Prot XML (gzipped) from UniProt FTP.

    Args:
        output_path: Directory to save the file

    Returns:
        Path to downloaded gzipped XML file
    """
    dest = output_path / "uniprot_sprot.xml.gz"
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Swiss-Prot XML already downloaded: {dest} ({size_mb:.0f} MB)")
        return dest

    print(f"Downloading Swiss-Prot from {SWISSPROT_XML_URL}")
    print("This may take several minutes (~4 GB)...")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".gz.partial")

    request = urllib.request.Request(
        SWISSPROT_XML_URL,
        headers={"User-Agent": "vHold/1.0 (https://github.com/shandley/vhold)"},
    )

    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(partial, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100 * downloaded / total
                    print(
                        f"\r  {downloaded / (1024 * 1024):.0f} / "
                        f"{total / (1024 * 1024):.0f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

    partial.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\nDownloaded {size_mb:.0f} MB to {dest}")
    return dest


def parse_swissprot_xml(
    xml_path: Path,
    scope: str = "viral",
    require_go: bool = True,
) -> list[SwissProtEntry]:
    """Parse Swiss-Prot XML and extract entries with experimental GO annotations.

    Streams the XML to avoid loading the entire ~90 GB uncompressed file into
    memory. Uses iterparse with element clearing for constant memory usage.

    Args:
        xml_path: Path to gzipped Swiss-Prot XML
        scope: "viral" (only viral entries) or "all" (all Swiss-Prot)
        require_go: If True, skip entries without experimental GO terms

    Returns:
        List of SwissProtEntry with experimental GO annotations
    """
    entries: list[SwissProtEntry] = []
    n_total = 0
    n_skipped_scope = 0
    n_skipped_no_go = 0
    n_skipped_no_seq = 0

    print(f"Parsing Swiss-Prot XML (scope={scope}, require_go={require_go})...", flush=True)
    start_time = time.time()

    # Open gzipped XML
    if str(xml_path).endswith(".gz"):
        fh = gzip.open(xml_path, "rb")
    else:
        fh = open(xml_path, "rb")

    try:
        # Use start/end events to track the root and free parsed entries.
        # The key memory optimization: after processing each <entry>, we
        # clear it AND remove it from the root element. Without removal,
        # root accumulates 570K+ child references and the tree grows to
        # tens of GB.
        context = ET.iterparse(fh, events=("start", "end"))
        root = None

        for event, elem in context:
            if event == "start":
                if root is None:
                    root = elem  # Capture the <uniprot> root element
                continue

            # Only process "end" events for <entry> elements
            if elem.tag != f"{UNIPROT_NS}entry":
                continue

            n_total += 1

            # Extract accession
            acc_elem = elem.find(f"{UNIPROT_NS}accession")
            if acc_elem is None or acc_elem.text is None:
                elem.clear()
                if root is not None:
                    root.remove(elem)
                continue
            accession = acc_elem.text

            # Extract organism and lineage EARLY for fast scope filtering
            lineage_elem = elem.find(f"{UNIPROT_NS}organism/{UNIPROT_NS}lineage")
            lineage_parts: list[str] = []
            if lineage_elem is not None:
                for taxon in lineage_elem.findall(f"{UNIPROT_NS}taxon"):
                    if taxon.text:
                        lineage_parts.append(taxon.text)
            lineage = "; ".join(lineage_parts)

            # Filter by scope EARLY (skip ~97% of entries for viral scope)
            if scope == "viral" and not lineage.startswith("Viruses"):
                n_skipped_scope += 1
                elem.clear()
                if root is not None:
                    root.remove(elem)
                continue

            # Extract name
            name_elem = elem.find(f"{UNIPROT_NS}name")
            name = name_elem.text if name_elem is not None and name_elem.text else ""

            # Extract organism
            organism_elem = elem.find(
                f"{UNIPROT_NS}organism/{UNIPROT_NS}name[@type='scientific']"
            )
            organism = (
                organism_elem.text
                if organism_elem is not None and organism_elem.text
                else ""
            )

            # Extract sequence
            seq_elem = elem.find(f"{UNIPROT_NS}sequence")
            sequence = ""
            length = 0
            if seq_elem is not None and seq_elem.text:
                sequence = seq_elem.text.replace("\n", "").replace(" ", "")
                length = len(sequence)
            if not sequence:
                n_skipped_no_seq += 1
                elem.clear()
                if root is not None:
                    root.remove(elem)
                continue

            # Extract GO annotations with experimental evidence
            go_annotations: list[GOAnnotation] = []
            for dbref in elem.findall(f"{UNIPROT_NS}dbReference[@type='GO']"):
                go_id = dbref.get("id", "")
                if not go_id:
                    continue

                go_term = ""
                aspect = ""
                evidence_code = ""

                for prop in dbref.findall(f"{UNIPROT_NS}property"):
                    prop_type = prop.get("type", "")
                    prop_value = prop.get("value", "")
                    if prop_type == "term":
                        # Format: "C:cytoplasm" or "F:ATP binding" or "P:viral process"
                        if ":" in prop_value:
                            aspect_code, term = prop_value.split(":", 1)
                            aspect = {"C": "CC", "F": "MF", "P": "BP"}.get(
                                aspect_code, aspect_code
                            )
                            go_term = term
                        else:
                            go_term = prop_value
                    elif prop_type == "evidence":
                        # Format: "ECO:0000314" or sometimes the code directly
                        evidence_code = _eco_to_evidence_code(prop_value)

                # Only keep experimental evidence
                if evidence_code in EXPERIMENTAL_EVIDENCE_CODES:
                    go_annotations.append(
                        GOAnnotation(
                            go_id=go_id,
                            go_term=go_term,
                            aspect=aspect,
                            evidence_code=evidence_code,
                        )
                    )

            # Filter: require at least one experimental GO annotation
            if require_go and not go_annotations:
                n_skipped_no_go += 1
                elem.clear()
                if root is not None:
                    root.remove(elem)
                continue

            entries.append(
                SwissProtEntry(
                    accession=accession,
                    name=name,
                    organism=organism,
                    lineage=lineage,
                    go_annotations=go_annotations,
                    sequence=sequence,
                    length=length,
                )
            )

            # Progress
            if n_total % 50000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Processed {n_total:,} entries, "
                    f"kept {len(entries):,} ({elapsed:.0f}s)",
                    flush=True,
                )

            # Clear element and remove from root to free memory
            elem.clear()
            if root is not None:
                root.remove(elem)
    finally:
        fh.close()

    elapsed = time.time() - start_time
    print(f"\nParsing complete in {elapsed:.0f}s:")
    print(f"  Total entries processed:  {n_total:,}")
    print(f"  Skipped (scope):          {n_skipped_scope:,}")
    print(f"  Skipped (no exp GO):      {n_skipped_no_go:,}")
    print(f"  Skipped (no sequence):    {n_skipped_no_seq:,}")
    print(f"  Kept:                     {len(entries):,}")

    return entries


# ECO code to evidence code mapping (Gene Ontology standard)
ECO_TO_EVIDENCE = {
    "ECO:0000269": "EXP",  # Experimental evidence used in manual assertion
    "ECO:0000314": "IDA",  # Direct assay evidence
    "ECO:0000353": "IPI",  # Physical interaction evidence
    "ECO:0000315": "IMP",  # Mutant phenotype evidence
    "ECO:0000316": "IGI",  # Genetic interaction evidence
    "ECO:0000270": "IEP",  # Expression pattern evidence
    "ECO:0000304": "TAS",  # Traceable author statement
    "ECO:0000305": "IC",   # Inferred by curator
    # Computational codes (excluded by filter, but map them for completeness)
    "ECO:0000501": "IEA",  # Inferred from electronic annotation
    "ECO:0000250": "ISS",  # Inferred from sequence similarity
    "ECO:0000266": "ISO",  # Inferred from sequence orthology
    "ECO:0000247": "ISA",  # Inferred from sequence alignment
    "ECO:0000255": "ISM",  # Inferred from sequence model
    "ECO:0000317": "IGC",  # Inferred from genomic context
    "ECO:0000318": "IBA",  # Inferred from biological aspect of ancestor
    "ECO:0000319": "IBD",  # Inferred from biological aspect of descendant
    "ECO:0000320": "IKR",  # Inferred from key residues
    "ECO:0000321": "IRD",  # Inferred from rapid divergence
    "ECO:0000245": "RCA",  # Inferred from reviewed computational analysis
    "ECO:0000303": "NAS",  # Non-traceable author statement
    "ECO:0000307": "ND",   # No biological data available
}


def _eco_to_evidence_code(eco_str: str) -> str:
    """Convert ECO code to GO evidence code.

    UniProt XML uses ECO codes (ECO:0000314), but GO uses evidence codes (IDA).
    Some entries may have the evidence code directly.

    Args:
        eco_str: ECO code string (e.g., "ECO:0000314") or evidence code

    Returns:
        GO evidence code (e.g., "IDA") or empty string if unknown
    """
    eco_str = eco_str.strip()

    # Already an evidence code?
    if not eco_str.startswith("ECO:"):
        return eco_str

    # Extract the ECO code (may have additional qualifier after space)
    eco_code = eco_str.split()[0] if " " in eco_str else eco_str
    return ECO_TO_EVIDENCE.get(eco_code, "")


def extract_embeddings(
    entries: list[SwissProtEntry],
    device: torch.device,
    model_dir: Path | None = None,
    batch_size: int = 32,
    max_length: int = 1024,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 1000,
    resume: bool = False,
) -> np.ndarray:
    """Extract ProstT5 encoder embeddings for all entries.

    Uses mean pooling over encoder hidden states to produce a single
    1024-dim L2-normalized vector per protein.

    Args:
        entries: Swiss-Prot entries with sequences
        device: Torch device (cuda, mps, cpu)
        model_dir: ProstT5 model directory
        batch_size: Batch size for embedding extraction
        max_length: Maximum sequence length in tokens
        checkpoint_dir: Directory for saving checkpoints
        checkpoint_interval: Save checkpoint every N proteins
        resume: Whether to resume from existing checkpoint

    Returns:
        (N, 1024) float16 array of L2-normalized embeddings
    """
    from vhold.features.prostt5 import prepare_prostt5_sequence

    n_total = len(entries)
    dim = EMBEDDING_DIM

    # Check for existing checkpoint
    start_idx = 0
    checkpoint_path = checkpoint_dir / "swissprot_checkpoint.npz" if checkpoint_dir else None

    if resume and checkpoint_path and checkpoint_path.exists():
        ckpt = np.load(checkpoint_path)
        existing_emb = ckpt["embeddings"]
        start_idx = int(ckpt["next_idx"].item())
        embeddings = np.zeros((n_total, dim), dtype=np.float16)
        embeddings[:start_idx] = existing_emb[:start_idx]
        print(f"Resuming from checkpoint: {start_idx}/{n_total} done")
    else:
        embeddings = np.zeros((n_total, dim), dtype=np.float16)

    if start_idx >= n_total:
        print("All embeddings already extracted")
        return embeddings

    # Load model
    print(f"Loading ProstT5 encoder on {device}...")
    model_name = PROSTT5_MODEL_NAME
    if model_dir and (model_dir / "config.json").exists():
        model_name = str(model_dir)

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    # Process in batches
    start_time = time.time()
    processed = 0

    for batch_start in range(start_idx, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_entries = entries[batch_start:batch_end]

        # Prepare sequences
        sequences = [
            prepare_prostt5_sequence(e.sequence[:max_length]) for e in batch_entries
        ]

        tokens = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling over sequence length (ignoring padding)
            hidden = outputs.last_hidden_state  # (B, L, 1024)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts  # (B, 1024)

            # L2 normalize
            norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
            normalized = pooled / norms

        emb_np = normalized.cpu().numpy().astype(np.float16)
        embeddings[batch_start:batch_end] = emb_np
        processed += len(batch_entries)

        # Progress reporting
        total_done = start_idx + processed
        if total_done % max(100, batch_size) == 0 or batch_end == n_total:
            elapsed = time.time() - start_time
            rate = processed / max(elapsed, 1)
            remaining_count = n_total - total_done
            eta = remaining_count / max(rate, 1e-6)
            print(
                f"  {total_done:,}/{n_total:,} "
                f"({100 * total_done / n_total:.1f}%) "
                f"{rate:.1f} prot/s, ETA {eta / 3600:.1f}h"
            )

        # Checkpoint
        if checkpoint_path and processed % checkpoint_interval == 0:
            np.savez_compressed(
                checkpoint_path,
                embeddings=embeddings,
                next_idx=np.array(total_done),
            )
            print(f"  Checkpoint saved at {total_done}")

    elapsed = time.time() - start_time
    print(f"\nExtracted {n_total:,} embeddings in {elapsed / 3600:.1f}h")

    return embeddings


def save_outputs(
    entries: list[SwissProtEntry],
    embeddings: np.ndarray,
    output_dir: Path,
    scope: str,
) -> None:
    """Save embeddings and metadata to output files.

    Args:
        entries: Swiss-Prot entries (sequences removed from metadata to save space)
        embeddings: (N, 1024) float16 embeddings
        output_dir: Output directory
        scope: "viral" or "all"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if scope == "viral":
        emb_file = SWISSPROT_VIRAL_EMBEDDINGS
        meta_file = SWISSPROT_VIRAL_METADATA
    else:
        emb_file = SWISSPROT_ALL_EMBEDDINGS
        meta_file = SWISSPROT_ALL_METADATA

    # Save embeddings
    protein_ids = np.array([e.accession for e in entries], dtype=object)
    emb_path = output_dir / emb_file
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        protein_ids=protein_ids,
    )
    emb_size = emb_path.stat().st_size / (1024 * 1024)
    print(f"Saved embeddings: {emb_path} ({emb_size:.1f} MB)")

    # Save metadata (without sequences, to reduce file size)
    metadata = {}
    for entry in entries:
        metadata[entry.accession] = entry.to_dict()
    meta_path = output_dir / meta_file
    with open(meta_path, "w") as f:
        json.dump(metadata, f, separators=(",", ":"))
    meta_size = meta_path.stat().st_size / (1024 * 1024)
    print(f"Saved metadata:   {meta_path} ({meta_size:.1f} MB)")


def print_summary(entries: list[SwissProtEntry], scope: str) -> None:
    """Print summary statistics of extracted entries."""
    n = len(entries)
    n_go = sum(len(e.go_annotations) for e in entries)
    n_bp = sum(1 for e in entries for g in e.go_annotations if g.aspect == "BP")
    n_mf = sum(1 for e in entries for g in e.go_annotations if g.aspect == "MF")
    n_cc = sum(1 for e in entries for g in e.go_annotations if g.aspect == "CC")

    # Unique GO terms
    unique_go = len({g.go_id for e in entries for g in e.go_annotations})

    # Organisms
    organisms = {e.organism for e in entries}

    # Cross-domain (non-viral)
    n_viral = sum(1 for e in entries if e.is_viral)
    n_nonviral = n - n_viral

    # Length stats
    lengths = [e.length for e in entries]
    mean_len = sum(lengths) / max(len(lengths), 1)

    # Evidence code distribution
    evidence_counts: dict[str, int] = {}
    for e in entries:
        for g in e.go_annotations:
            evidence_counts[g.evidence_code] = evidence_counts.get(g.evidence_code, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Swiss-Prot Reference Database Summary (scope={scope})")
    print(f"{'=' * 60}")
    print(f"Proteins:           {n:,}")
    print(f"  Viral:            {n_viral:,}")
    print(f"  Non-viral:        {n_nonviral:,}")
    print(f"Unique organisms:   {len(organisms):,}")
    print(f"GO annotations:     {n_go:,}")
    print(f"  BP:               {n_bp:,}")
    print(f"  MF:               {n_mf:,}")
    print(f"  CC:               {n_cc:,}")
    print(f"Unique GO terms:    {unique_go:,}")
    print(f"Mean seq length:    {mean_len:.0f} aa")
    print(f"\nEvidence code distribution:")
    for code in sorted(evidence_counts, key=lambda c: evidence_counts[c], reverse=True):
        print(f"  {code}: {evidence_counts[code]:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Swiss-Prot reference database for GO term transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Viral proteins only (fast validation, ~30 min on A100)
  python scripts/build_swissprot_reference.py \\
    --scope viral --output ~/.vhold/databases/swissprot/ \\
    --device cuda --batch-size 32

  # Full Swiss-Prot (cross-domain discovery, ~16 hrs on A100)
  python scripts/build_swissprot_reference.py \\
    --scope all --output ~/.vhold/databases/swissprot/ \\
    --device cuda --batch-size 32 --checkpoint-interval 1000

  # Resume interrupted run
  python scripts/build_swissprot_reference.py \\
    --scope all --output ~/.vhold/databases/swissprot/ \\
    --device cuda --batch-size 32 --resume

  # Parse only (no embedding extraction, for testing)
  python scripts/build_swissprot_reference.py \\
    --scope viral --output /tmp/swissprot/ --parse-only
""",
    )
    parser.add_argument(
        "--scope",
        choices=["viral", "all"],
        default="viral",
        help="Scope: 'viral' (~17K proteins) or 'all' (~570K proteins). Default: viral",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for embeddings and metadata files",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to existing uniprot_sprot.xml.gz (skip download)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for ProstT5 (auto, cuda, mps, cpu). Default: auto",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="ProstT5 model directory (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction. Default: 32",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N proteins. Default: 1000",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Parse XML and save metadata only (no embedding extraction)",
    )
    parser.add_argument(
        "--no-require-go",
        action="store_true",
        help="Include proteins without experimental GO annotations",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir) if args.model_dir else get_model_dir()

    # Step 1: Download Swiss-Prot XML
    if args.xml:
        xml_path = Path(args.xml)
        if not xml_path.exists():
            print(f"ERROR: Specified XML file not found: {xml_path}")
            sys.exit(1)
    else:
        xml_path = download_swissprot(output_dir)

    # Step 2: Parse XML and extract entries
    entries = parse_swissprot_xml(
        xml_path,
        scope=args.scope,
        require_go=not args.no_require_go,
    )

    if not entries:
        print("ERROR: No entries extracted. Check scope and data.")
        sys.exit(1)

    # Print summary
    print_summary(entries, args.scope)

    if args.parse_only:
        # Save metadata only
        metadata = {}
        for entry in entries:
            metadata[entry.accession] = entry.to_dict()

        if args.scope == "viral":
            meta_file = SWISSPROT_VIRAL_METADATA
        else:
            meta_file = SWISSPROT_ALL_METADATA

        meta_path = output_dir / meta_file
        with open(meta_path, "w") as f:
            json.dump(metadata, f, separators=(",", ":"))
        meta_size = meta_path.stat().st_size / (1024 * 1024)
        print(f"\nSaved metadata only: {meta_path} ({meta_size:.1f} MB)")
        return

    # Step 3: Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Step 4: Extract embeddings
    print(f"\n{'=' * 60}")
    print(f"Extracting ProstT5 embeddings for {len(entries):,} proteins")
    print(f"{'=' * 60}")

    embeddings = extract_embeddings(
        entries,
        device=device,
        model_dir=model_dir,
        batch_size=args.batch_size,
        checkpoint_dir=output_dir if args.resume else None,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )

    # Step 5: Save outputs
    save_outputs(entries, embeddings, output_dir, args.scope)

    # Clean up checkpoint
    checkpoint_path = output_dir / "swissprot_checkpoint.npz"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up")

    print(f"\nDone! Swiss-Prot reference database built at {output_dir}")


if __name__ == "__main__":
    main()
