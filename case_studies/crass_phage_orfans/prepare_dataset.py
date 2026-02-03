#!/usr/bin/env python3
"""
Prepare crAssphage protein dataset for vHold case study.

Selects a subset of proteins including:
- ORFans (hypothetical proteins)
- Proteins with NCBI annotations
- Proteins characterized in Structural Atlas (Nature 2023)
"""

import re
from pathlib import Path


def parse_fasta(fasta_path: Path) -> list[dict]:
    """Parse FASTA file into list of protein records."""
    proteins = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    proteins.append({
                        "header": current_header,
                        "sequence": "".join(current_seq)
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last protein
        if current_header:
            proteins.append({
                "header": current_header,
                "sequence": "".join(current_seq)
            })

    return proteins


def extract_info(header: str) -> dict:
    """Extract protein info from NCBI header."""
    info = {}

    # Extract locus tag (gp number)
    locus_match = re.search(r'\[locus_tag=KP06_(gp\d+)\]', header)
    if locus_match:
        info["locus_tag"] = locus_match.group(1)

    # Extract protein ID
    prot_match = re.search(r'\[protein_id=(YP_\d+\.\d+)\]', header)
    if prot_match:
        info["protein_id"] = prot_match.group(1)

    # Extract protein name
    name_match = re.search(r'\[protein=([^\]]+)\]', header)
    if name_match:
        info["protein"] = name_match.group(1)

    return info


# Ground truth from literature:
# - 2023 Nature paper: "Structural atlas of a human gut crassvirus"
# - Mass spectrometry confirmed virion proteins
# - Cryo-EM structural characterization

LITERATURE_ANNOTATIONS = {
    # Capsid and structural proteins (from Nature 2023)
    "gp73": {"function": "minor capsid protein", "category": "structural", "source": "cryo-EM"},
    "gp74": {"function": "major capsid protein", "category": "structural", "source": "cryo-EM"},
    "gp75": {"function": "head decoration protein", "category": "structural", "source": "cryo-EM"},
    "gp76": {"function": "head fiber protein", "category": "structural", "source": "cryo-EM"},
    "gp77": {"function": "vertex protein", "category": "structural", "source": "cryo-EM"},

    # Tail proteins
    "gp58": {"function": "tail tube protein", "category": "structural", "source": "cryo-EM"},
    "gp59": {"function": "tail protein", "category": "structural", "source": "MS"},
    "gp60": {"function": "tail protein", "category": "structural", "source": "MS"},
    "gp61": {"function": "portal protein", "category": "packaging", "source": "cryo-EM"},

    # RNA polymerase complex (virion-packaged)
    "gp47": {"function": "RNA polymerase subunit", "category": "transcription", "source": "MS/structure"},
    "gp49": {"function": "RNA polymerase subunit", "category": "transcription", "source": "MS"},
    "gp50": {"function": "virion RNA polymerase", "category": "transcription", "source": "MS/structure"},

    # Cargo proteins
    "gp45": {"function": "cargo protein 1", "category": "unknown", "source": "MS"},
    "gp46": {"function": "cargo protein 2", "category": "unknown", "source": "MS"},

    # Replication (NCBI annotations)
    "gp12": {"function": "DNA helicase", "category": "replication", "source": "NCBI"},
    "gp14": {"function": "DNA polymerase", "category": "replication", "source": "NCBI"},
    "gp22": {"function": "DNA ligase", "category": "nuclease", "source": "NCBI"},
    "gp72": {"function": "DNA helicase", "category": "replication", "source": "NCBI"},

    # Nucleotide metabolism
    "gp13": {"function": "uracil-DNA glycosylase", "category": "nuclease", "source": "NCBI"},
    "gp23": {"function": "dNMP kinase", "category": "replication", "source": "NCBI"},
    "gp25": {"function": "thymidylate synthase", "category": "replication", "source": "NCBI"},
    "gp29": {"function": "dUTPase", "category": "replication", "source": "NCBI"},
    "gp30": {"function": "endonuclease", "category": "nuclease", "source": "NCBI"},

    # Lysis
    "gp73_alt": {"function": "N-acetylmuramidase", "category": "lysis", "source": "NCBI"},

    # BACON domain protein (unique to crAssphage)
    "gp64": {"function": "BACON domain protein", "category": "unknown", "source": "literature"},
}


def main():
    """Process crAssphage proteins and create test dataset."""
    input_path = Path(__file__).parent / "crass_proteins_raw.fasta"
    output_path = Path(__file__).parent / "crass_orfans.fasta"

    proteins = parse_fasta(input_path)
    print(f"Total proteins: {len(proteins)}")

    # Process each protein
    selected = []
    for p in proteins:
        info = extract_info(p["header"])
        info["sequence"] = p["sequence"]
        info["length"] = len(p["sequence"])

        # Skip very short proteins
        if info["length"] < 50:
            continue

        # Add literature annotation if available
        locus = info.get("locus_tag", "")
        if locus in LITERATURE_ANNOTATIONS:
            info["ground_truth"] = LITERATURE_ANNOTATIONS[locus]
        else:
            info["ground_truth"] = {
                "function": info.get("protein", "hypothetical protein"),
                "category": "unknown" if "hypothetical" in info.get("protein", "") else "unknown",
                "source": "NCBI"
            }

        selected.append(info)

    print(f"After length filter (>=50 aa): {len(selected)}")

    # Select diverse subset for testing
    # Include all proteins with literature annotations plus some ORFans
    test_proteins = []

    # Add all proteins with non-hypothetical annotations
    for p in selected:
        if p["locus_tag"] in LITERATURE_ANNOTATIONS:
            test_proteins.append(p)
        elif "hypothetical" not in p.get("protein", "hypothetical"):
            test_proteins.append(p)

    print(f"Proteins with annotations: {len(test_proteins)}")

    # Add some ORFans (hypothetical proteins of varying sizes)
    orfans = [p for p in selected if "hypothetical" in p.get("protein", "")]
    # Sort by length and take diverse sample
    orfans.sort(key=lambda x: x["length"])

    # Take ORFans at different size ranges
    for i in range(0, len(orfans), len(orfans) // 15):  # ~15 ORFans
        if orfans[i] not in test_proteins:
            test_proteins.append(orfans[i])

    print(f"Total test proteins: {len(test_proteins)}")

    # Write FASTA
    with open(output_path, "w") as f:
        for p in test_proteins:
            locus = p.get("locus_tag", "unknown")
            length = p["length"]
            protein_name = p.get("protein", "hypothetical protein")
            f.write(f">{locus}|{length}aa|{protein_name}\n")
            # Wrap sequence at 70 chars
            seq = p["sequence"]
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + "\n")

    print(f"Wrote {len(test_proteins)} proteins to {output_path}")

    # Print summary
    print("\nProtein summary:")
    annotated = sum(1 for p in test_proteins if p["locus_tag"] in LITERATURE_ANNOTATIONS)
    orfan_count = sum(1 for p in test_proteins if "hypothetical" in p.get("protein", ""))
    print(f"  With literature annotations: {annotated}")
    print(f"  ORFans (hypothetical): {orfan_count}")

    # Print selected proteins
    print("\nSelected proteins:")
    for p in sorted(test_proteins, key=lambda x: int(x["locus_tag"][2:]) if x["locus_tag"].startswith("gp") else 0):
        locus = p.get("locus_tag", "?")
        length = p["length"]
        name = p.get("protein", "?")[:40]
        gt = p.get("ground_truth", {}).get("function", "?")[:30]
        print(f"  {locus:6} {length:4}aa  {name:40}  [{gt}]")


if __name__ == "__main__":
    main()
