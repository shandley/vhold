#!/usr/bin/env python3
"""
Fetch diverse eukaryotic virus proteins for vHold case study.

Targets virus families that demonstrate vHold's eukaryotic virus capabilities:
- Paramyxoviridae (Nipah, Hendra, measles)
- Rhabdoviridae (rabies, VSV)
- Filoviridae (Ebola, Marburg)
- Bunyavirales (hantaviruses)
- Togaviridae (alphaviruses)
- Flaviviridae (dengue, Zika)
"""

import subprocess
import sys
from pathlib import Path

# Representative eukaryotic virus genomes (RefSeq accessions)
VIRUS_GENOMES = {
    # Paramyxoviridae - negative-sense ssRNA
    "Nipah_virus": "NC_002728",  # Henipavirus
    "Hendra_virus": "NC_001906",  # Henipavirus
    "Measles_virus": "NC_001498",  # Morbillivirus
    "Mumps_virus": "NC_002200",  # Rubulavirus

    # Rhabdoviridae - negative-sense ssRNA
    "Rabies_virus": "NC_001542",
    "VSV": "NC_001560",  # Vesicular stomatitis virus

    # Filoviridae - negative-sense ssRNA
    "Ebola_virus": "NC_002549",  # Zaire ebolavirus
    "Marburg_virus": "NC_001608",

    # Bunyavirales - segmented negative-sense ssRNA
    "Hantaan_virus_S": "NC_005218",  # Hantavirus S segment
    "Hantaan_virus_M": "NC_005219",
    "Hantaan_virus_L": "NC_005222",
    "RVFV_S": "NC_014395",  # Rift Valley fever

    # Togaviridae - positive-sense ssRNA
    "Chikungunya": "NC_004162",
    "Sindbis_virus": "NC_001547",

    # Flaviviridae - positive-sense ssRNA
    "Dengue_1": "NC_001477",
    "Zika_virus": "NC_035889",
    "Yellow_fever": "NC_002031",

    # Coronaviridae (beyond SARS-CoV-2)
    "MERS_CoV": "NC_019843",
    "HCoV_229E": "NC_002645",  # Common cold coronavirus
}


def fetch_proteins(accession: str, virus_name: str) -> str:
    """Fetch proteins from NCBI for a given accession."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta_cds_aa&retmode=text"

    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except Exception as e:
        print(f"Error fetching {virus_name} ({accession}): {e}", file=sys.stderr)
        return ""


def main():
    """Fetch all virus proteins and create test dataset."""
    output_dir = Path(__file__).parent
    all_proteins_path = output_dir / "all_eukaryotic_proteins.fasta"
    test_proteins_path = output_dir / "test_proteins.fasta"

    print("Fetching eukaryotic virus proteins from NCBI...")
    print()

    all_proteins = []
    protein_counts = {}

    for virus_name, accession in VIRUS_GENOMES.items():
        print(f"Fetching {virus_name} ({accession})...")
        fasta = fetch_proteins(accession, virus_name)

        if fasta:
            # Count proteins
            count = fasta.count(">")
            protein_counts[virus_name] = count
            all_proteins.append(f"# {virus_name}\n{fasta}")
            print(f"  Found {count} proteins")
        else:
            print(f"  Failed to fetch")

    # Write all proteins
    with open(all_proteins_path, "w") as f:
        f.write("\n".join(all_proteins))

    print()
    print(f"Total viruses: {len(protein_counts)}")
    print(f"Total proteins: {sum(protein_counts.values())}")
    print(f"Saved to: {all_proteins_path}")

    # Create curated test set
    print()
    print("Creating curated test set...")
    print("(Select specific proteins of interest)")


if __name__ == "__main__":
    main()
