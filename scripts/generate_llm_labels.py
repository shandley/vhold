#!/usr/bin/env python3
"""Generate expanded training labels using LLM classification.

Identifies proteins classified as "unknown" by the keyword system that have
informative descriptions, de-duplicates by description, and batch-classifies
them using Claude Haiku. Saves a mapping file usable by train_classifier.py.

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/generate_llm_labels.py \
        --output results/llm_labels.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vhold.databases.bfvd import load_bfvd_enriched_metadata
from vhold.databases.viro3d import (
    get_annotation_expansion,
    get_viro3d_annotation,
    load_viro3d_metadata,
)
from vhold.results.categories import (
    ALL_CATEGORIES,
    AnnotationEvidence,
    get_classification_source,
)
from vhold.utils.constants import get_db_dir

VALID_CATEGORIES = set(ALL_CATEGORIES)

# Descriptions that are definitely uninformative — skip LLM
SKIP_PATTERNS = {
    "uncharacterized", "hypothetical", "deleted", "unnamed",
    "unknown function", "putative orfan", "predicted protein",
    "putative protein",
}


def collect_informative_unknowns(protein_ids, source_dbs, enriched, viro3d_meta, expansion):
    """Find unknown proteins with informative descriptions.

    Returns:
        desc_to_proteins: dict mapping (description, gene, organism) -> [protein_ids]
        desc_counts: Counter of description strings
    """
    desc_to_proteins = defaultdict(list)
    desc_counts = Counter()
    desc_context = {}  # description -> (gene, organism) for LLM context

    for i, pid in enumerate(protein_ids):
        pid_str = str(pid)
        src = str(source_dbs[i])

        desc = ""
        gene = ""
        organism = ""
        pfam = go_bp = go_mf = None

        if src == "bfvd" and pid_str in enriched:
            row = enriched[pid_str]
            desc = row.get("protein_name", "")
            gene = row.get("gene_name", "")
            pfam = row.get("pfam_domains") or None
            go_bp = row.get("go_bp") or None
            go_mf = row.get("go_mf") or None
            lineage = row.get("lineage", "") or ""
            if lineage and "," in lineage:
                organism = lineage.split(",")[-1].strip()
            else:
                organism = lineage
        elif src == "viro3d" and viro3d_meta is not None:
            ann = get_viro3d_annotation(
                pid_str, viro3d_meta,
                annotations_df=None, annotation_expansion=expansion,
            )
            if ann:
                desc = ann.get("description", "")
                gene = ann.get("gene", "")
                organism = ann.get("organism", "")

        evidence = AnnotationEvidence(pfam=pfam, go_bp=go_bp, go_mf=go_mf)
        cat, _ = get_classification_source(desc, gene, evidence)

        if cat != "unknown":
            continue

        desc_lower = (desc or "").lower().strip()
        if not desc_lower or desc_lower == "deleted":
            continue
        if any(skip in desc_lower for skip in SKIP_PATTERNS):
            continue
        if desc_lower.startswith("orf") and len(desc_lower) < 6:
            continue

        # Use description as the dedup key (gene/organism provide context)
        desc_to_proteins[desc].append(pid_str)
        desc_counts[desc] += 1
        if desc not in desc_context:
            desc_context[desc] = (gene, organism)

    return desc_to_proteins, desc_counts, desc_context


def build_batch_prompt(descriptions_with_context):
    """Build a batch classification prompt.

    Args:
        descriptions_with_context: list of (idx, description, gene, organism)

    Returns:
        system prompt, user prompt
    """
    system = (
        "You are a virology expert. Classify each viral protein into exactly one category.\n"
        "Valid categories: structural, replication, protease, nuclease, packaging, "
        "regulatory, movement, lysis, host_interaction, entry, unknown\n\n"
        "Rules:\n"
        "- structural: capsid, envelope, matrix, nucleoprotein, glycoprotein, virion proteins\n"
        "- replication: polymerase, helicase, primase, RdRp, capping enzymes, DNA-binding, "
        "thymidine kinase, dUTPase, nucleotide metabolism\n"
        "- protease: proteases, peptidases\n"
        "- nuclease: nucleases, integrases, ligases, glycosylases, resolvases\n"
        "- packaging: terminase, portal, scaffold\n"
        "- regulatory: transcription factors, kinases, transactivators, "
        "immediate-early proteins, transforming proteins\n"
        "- movement: plant virus movement proteins\n"
        "- lysis: lysins, holins, endolysins, lysozymes, transglycosylases\n"
        "- host_interaction: immune evasion, ankyrin repeats, F-box, ubiquitin ligases, "
        "interferon antagonists, anti-restriction\n"
        "- entry: receptor binding, membrane fusion, neuraminidase\n"
        "- unknown: cannot determine from description\n\n"
        'Return ONLY a JSON array: [{"id": 1, "category": "structural"}, ...]'
    )

    lines = ["Classify these viral proteins:"]
    for idx, desc, gene, organism in descriptions_with_context:
        parts = [f'{idx}. "{desc}"']
        if gene:
            parts.append(f"(gene: {gene})")
        if organism:
            parts.append(f"[{organism[:50]}]")
        lines.append(" ".join(parts))

    user = "\n".join(lines)
    return system, user


def classify_batch(client, descriptions_with_context, model="claude-haiku-4-5-20251001"):
    """Classify a batch of descriptions via LLM.

    Args:
        client: Anthropic client
        descriptions_with_context: list of (idx, description, gene, organism)
        model: Model to use

    Returns:
        dict mapping idx -> category
    """
    system, user = build_batch_prompt(descriptions_with_context)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = response.content[0].text.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        results = json.loads(text)
        mapping = {}
        for item in results:
            idx = item.get("id")
            cat = item.get("category", "unknown")
            if cat not in VALID_CATEGORIES:
                cat = "unknown"
            mapping[idx] = cat
        return mapping
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: Failed to parse batch response: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate expanded training labels via LLM"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON file for label mapping",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="Database directory (default: ~/.vhold/databases)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=40,
        help="Descriptions per LLM call (default: 40)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-haiku-4-5-20251001",
        help="LLM model (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Maximum number of batches to process (for testing)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Resume from checkpoint file",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_dir = Path(args.db_dir) if args.db_dir else get_db_dir()

    # Load embedding DB
    from vhold.databases.embeddings import get_embedding_db_path
    emb_path = get_embedding_db_path(db_dir)
    print(f"Loading embedding database from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    protein_ids = data["protein_ids"]
    source_dbs = data["source_dbs"]
    print(f"Loaded {len(protein_ids)} proteins")

    # Load metadata
    print("Loading BFVD enriched metadata...")
    enriched = load_bfvd_enriched_metadata(db_dir)
    print(f"Loaded {len(enriched)} BFVD entries")

    try:
        print("Loading Viro3D metadata...")
        viro3d_meta = load_viro3d_metadata(db_dir)
        expansion = get_annotation_expansion(db_dir)
        expansion.load_all()
        print("Viro3D metadata loaded")
    except Exception:
        viro3d_meta = None
        expansion = None
        print("Viro3D metadata not available")

    # Collect informative unknowns (single pass — also collects context)
    print("\nCollecting informative unknown proteins...")
    desc_to_proteins, desc_counts, desc_context = collect_informative_unknowns(
        protein_ids, source_dbs, enriched, viro3d_meta, expansion,
    )

    total_proteins = sum(len(pids) for pids in desc_to_proteins.values())
    print(f"Found {total_proteins:,} proteins with {len(desc_to_proteins):,} unique descriptions")

    # Load checkpoint if resuming
    desc_labels = {}
    if args.checkpoint and Path(args.checkpoint).exists():
        with open(args.checkpoint) as f:
            desc_labels = json.load(f)
        print(f"Resumed from checkpoint: {len(desc_labels)} descriptions already classified")

    # Prepare batches of unique descriptions not yet classified
    remaining = [d for d in desc_to_proteins if d not in desc_labels]
    print(f"Descriptions to classify: {len(remaining):,}")

    # Sort by count (most common first) so early batches have highest impact
    remaining.sort(key=lambda d: desc_counts[d], reverse=True)

    batches = []
    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i : i + args.batch_size]
        batches.append(batch)

    if args.max_batches:
        batches = batches[: args.max_batches]

    total_batches = len(batches)
    print(f"Total batches: {total_batches} ({args.batch_size} per batch)")

    # Estimate cost
    est_input_tokens = total_batches * (300 + args.batch_size * 15)
    est_output_tokens = total_batches * args.batch_size * 15
    est_cost = est_input_tokens / 1e6 * 0.80 + est_output_tokens / 1e6 * 4.00
    print(f"Estimated cost: ${est_cost:.2f}")
    print()

    # Process batches
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    start_time = time.time()
    api_calls = 0
    errors = 0

    for batch_idx, batch in enumerate(batches):
        batch_items = []
        for j, desc in enumerate(batch, 1):
            gene, organism = desc_context.get(desc, ("", ""))
            batch_items.append((j, desc, gene, organism))

        try:
            results = classify_batch(client, batch_items, model=args.model)
            api_calls += 1

            for j, desc in enumerate(batch, 1):
                if j in results:
                    desc_labels[desc] = results[j]
                else:
                    desc_labels[desc] = "unknown"

        except Exception as e:
            errors += 1
            print(f"  Batch {batch_idx + 1}: ERROR: {e}")
            # Mark all as unknown on error
            for desc in batch:
                desc_labels[desc] = "unknown"
            # Rate limit: wait before retry
            time.sleep(2)

        # Progress reporting
        if (batch_idx + 1) % 25 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed * 60
            non_unknown = sum(1 for c in desc_labels.values() if c != "unknown")
            print(
                f"  Batch {batch_idx + 1}/{total_batches} "
                f"({elapsed:.0f}s, {rate:.0f} batches/min) "
                f"- {non_unknown}/{len(desc_labels)} non-unknown"
            )

        # Checkpoint every 100 batches
        if (batch_idx + 1) % 100 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(desc_labels, f)

    elapsed = time.time() - start_time

    # Map descriptions back to protein IDs
    protein_labels = {}
    for desc, category in desc_labels.items():
        if category == "unknown":
            continue
        for pid in desc_to_proteins.get(desc, []):
            protein_labels[pid] = category

    # Save final output
    output_data = {
        "description_labels": desc_labels,
        "protein_labels": protein_labels,
        "stats": {
            "total_descriptions": len(desc_labels),
            "non_unknown_descriptions": sum(1 for c in desc_labels.values() if c != "unknown"),
            "total_proteins_labeled": len(protein_labels),
            "api_calls": api_calls,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1),
            "model": args.model,
            "category_distribution": dict(Counter(protein_labels.values())),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    stats = output_data["stats"]
    print(f"Descriptions classified: {stats['total_descriptions']:,}")
    print(f"Non-unknown descriptions: {stats['non_unknown_descriptions']:,}")
    print(f"Proteins newly labeled: {stats['total_proteins_labeled']:,}")
    print(f"API calls: {stats['api_calls']}")
    print(f"Errors: {stats['errors']}")
    print(f"Time: {stats['elapsed_seconds']:.1f}s")
    print(f"Estimated cost: ${est_cost:.2f}")
    print()
    print("Category distribution:")
    for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {count:>6}")
    print()
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
