#!/usr/bin/env python3
"""
Prepare curated test set of eukaryotic virus proteins.

Selects diverse proteins across families with mix of:
- Well-characterized (structural) for validation
- Accessory proteins (often poorly annotated)
- Hypothetical proteins (true ORFans)
"""

import re
from pathlib import Path

# Proteins to include in test set (protein_id -> ground truth)
TEST_PROTEINS = {
    # Nipah virus (Henipavirus) - BSL-4 pathogen
    "NP_112021.1": {
        "virus": "Nipah virus",
        "gene": "N",
        "function": "nucleocapsid protein",
        "category": "structural",
        "notes": "RNA binding, encapsidation"
    },
    "NP_112023.1": {
        "virus": "Nipah virus",
        "gene": "V",
        "function": "V protein",
        "category": "regulatory",
        "notes": "Interferon antagonist, immune evasion"
    },
    "NP_112024.1": {
        "virus": "Nipah virus",
        "gene": "C",
        "function": "C protein",
        "category": "regulatory",
        "notes": "Interferon antagonist, accessory protein"
    },
    "NP_112025.1": {
        "virus": "Nipah virus",
        "gene": "M",
        "function": "matrix protein",
        "category": "structural",
        "notes": "Virion assembly, budding"
    },
    "NP_112026.1": {
        "virus": "Nipah virus",
        "gene": "F",
        "function": "fusion protein",
        "category": "structural",
        "notes": "Membrane fusion, class I fusion protein"
    },
    "NP_112027.1": {
        "virus": "Nipah virus",
        "gene": "G",
        "function": "attachment glycoprotein",
        "category": "structural",
        "notes": "Receptor binding (ephrin-B2/B3)"
    },

    # Hendra virus (closely related to Nipah)
    "NP_047111.2": {
        "virus": "Hendra virus",
        "gene": "F",
        "function": "fusion protein",
        "category": "structural",
        "notes": "Membrane fusion"
    },
    "NP_047109.1": {
        "virus": "Hendra virus",
        "gene": "C",
        "function": "C protein",
        "category": "regulatory",
        "notes": "Accessory protein"
    },

    # Measles virus
    "NP_056918.1": {
        "virus": "Measles virus",
        "gene": "N",
        "function": "nucleocapsid protein",
        "category": "structural",
        "notes": "RNA encapsidation"
    },
    "YP_003873249.2": {
        "virus": "Measles virus",
        "gene": "V",
        "function": "V protein",
        "category": "regulatory",
        "notes": "Interferon antagonist"
    },
    "NP_056922.1": {
        "virus": "Measles virus",
        "gene": "F",
        "function": "fusion protein",
        "category": "structural",
        "notes": "Class I fusion protein"
    },

    # Ebola virus (Filoviridae)
    "NP_066243.1": {
        "virus": "Ebola virus",
        "gene": "NP",
        "function": "nucleoprotein",
        "category": "structural",
        "notes": "Major structural protein"
    },
    "NP_066244.1": {
        "virus": "Ebola virus",
        "gene": "VP35",
        "function": "polymerase cofactor",
        "category": "replication",
        "notes": "Interferon antagonist, polymerase cofactor"
    },
    "NP_066245.1": {
        "virus": "Ebola virus",
        "gene": "VP40",
        "function": "matrix protein",
        "category": "structural",
        "notes": "Budding, filament formation"
    },
    "NP_066246.1": {
        "virus": "Ebola virus",
        "gene": "GP",
        "function": "spike glycoprotein",
        "category": "structural",
        "notes": "Receptor binding, membrane fusion"
    },
    "NP_066249.1": {
        "virus": "Ebola virus",
        "gene": "VP30",
        "function": "transcription activator",
        "category": "regulatory",
        "notes": "Transcription initiation"
    },
    "NP_066250.1": {
        "virus": "Ebola virus",
        "gene": "VP24",
        "function": "secondary matrix protein",
        "category": "structural",
        "notes": "Nucleocapsid maturation, immune evasion"
    },

    # Rabies virus (Rhabdoviridae)
    "NP_056793.1": {
        "virus": "Rabies virus",
        "gene": "N",
        "function": "nucleoprotein",
        "category": "structural",
        "notes": "RNA encapsidation"
    },
    "NP_056795.1": {
        "virus": "Rabies virus",
        "gene": "M",
        "function": "matrix protein",
        "category": "structural",
        "notes": "Virion assembly"
    },
    "NP_056796.1": {
        "virus": "Rabies virus",
        "gene": "G",
        "function": "glycoprotein",
        "category": "structural",
        "notes": "Receptor binding, membrane fusion"
    },

    # VSV (model rhabdovirus)
    "NP_041713.1": {
        "virus": "VSV",
        "gene": "NS",
        "function": "phosphoprotein",
        "category": "replication",
        "notes": "Polymerase cofactor"
    },
    "NP_041714.1": {
        "virus": "VSV",
        "gene": "M",
        "function": "matrix protein",
        "category": "structural",
        "notes": "Budding, host shutoff"
    },

    # Hantavirus (labeled as "hypothetical" - true ORFans!)
    "NP_941977.1": {
        "virus": "Hantaan virus",
        "gene": "N",
        "function": "nucleocapsid protein",
        "category": "structural",
        "notes": "NCBI: hypothetical protein - good ORFan test"
    },
    "NP_941978.1": {
        "virus": "Hantaan virus",
        "gene": "GPC",
        "function": "glycoprotein precursor",
        "category": "structural",
        "notes": "NCBI: hypothetical protein - good ORFan test"
    },

    # Chikungunya (Alphavirus)
    "NP_690589.2": {
        "virus": "Chikungunya",
        "gene": "structural",
        "function": "structural polyprotein",
        "category": "structural",
        "notes": "C-E3-E2-6K-E1 polyprotein"
    },

    # MERS-CoV (Coronavirus, eukaryotic)
    "YP_009047202.1": {
        "virus": "MERS-CoV",
        "gene": "S",
        "function": "spike protein",
        "category": "structural",
        "notes": "Receptor binding (DPP4)"
    },
    "YP_009047207.1": {
        "virus": "MERS-CoV",
        "gene": "M",
        "function": "membrane protein",
        "category": "structural",
        "notes": "Virion assembly"
    },
    "YP_009047208.1": {
        "virus": "MERS-CoV",
        "gene": "N",
        "function": "nucleocapsid protein",
        "category": "structural",
        "notes": "RNA packaging"
    },

    # Dengue virus (Flavivirus)
    "NP_059433.1": {
        "virus": "Dengue virus 1",
        "gene": "polyprotein",
        "function": "polyprotein",
        "category": "unknown",
        "notes": "Cleaved into C, prM, E, NS1-NS5"
    },
}


def parse_fasta(fasta_path: Path) -> dict:
    """Parse FASTA file into dict of protein_id -> sequence."""
    proteins = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    proteins[current_id] = {
                        "header": current_header,
                        "sequence": "".join(current_seq)
                    }
                current_header = line[1:]
                # Extract protein_id
                match = re.search(r'\[protein_id=([^\]]+)\]', current_header)
                current_id = match.group(1) if match else None
                current_seq = []
            elif line and not line.startswith("#"):
                current_seq.append(line)

        # Don't forget last protein
        if current_id:
            proteins[current_id] = {
                "header": current_header,
                "sequence": "".join(current_seq)
            }

    return proteins


def main():
    """Create curated test set."""
    input_path = Path(__file__).parent / "all_eukaryotic_proteins.fasta"
    output_path = Path(__file__).parent / "test_proteins.fasta"
    ground_truth_path = Path(__file__).parent / "ground_truth.json"

    # Parse all proteins
    all_proteins = parse_fasta(input_path)
    print(f"Loaded {len(all_proteins)} proteins")

    # Select test proteins
    test_set = []
    found = []
    missing = []

    for protein_id, gt in TEST_PROTEINS.items():
        if protein_id in all_proteins:
            test_set.append({
                "protein_id": protein_id,
                "header": all_proteins[protein_id]["header"],
                "sequence": all_proteins[protein_id]["sequence"],
                "ground_truth": gt
            })
            found.append(protein_id)
        else:
            missing.append(protein_id)

    print(f"Found: {len(found)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print(f"Missing proteins: {missing}")

    # Write test FASTA
    with open(output_path, "w") as f:
        for p in test_set:
            protein_id = p["protein_id"]
            virus = p["ground_truth"]["virus"]
            gene = p["ground_truth"]["gene"]
            length = len(p["sequence"])
            f.write(f">{protein_id}|{virus}|{gene}|{length}aa\n")
            seq = p["sequence"]
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + "\n")

    print(f"Wrote {len(test_set)} proteins to {output_path}")

    # Write ground truth JSON
    import json
    ground_truth = {
        "name": "Eukaryotic Virus Proteins",
        "description": "Diverse proteins from mammalian viruses (Paramyxoviridae, Filoviridae, Rhabdoviridae, Coronaviridae)",
        "total_proteins": len(test_set),
        "proteins": {p["protein_id"]: p["ground_truth"] for p in test_set},
        "virus_families": {
            "Paramyxoviridae": ["Nipah virus", "Hendra virus", "Measles virus"],
            "Filoviridae": ["Ebola virus"],
            "Rhabdoviridae": ["Rabies virus", "VSV"],
            "Hantaviridae": ["Hantaan virus"],
            "Togaviridae": ["Chikungunya"],
            "Coronaviridae": ["MERS-CoV"],
            "Flaviviridae": ["Dengue virus 1"]
        },
        "expected_categories": {
            "structural": "Capsid, envelope, matrix proteins",
            "regulatory": "Accessory proteins, immune evasion",
            "replication": "Polymerase cofactors"
        }
    }

    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote ground truth to {ground_truth_path}")

    # Summary by virus
    print("\nProteins by virus:")
    virus_counts = {}
    for p in test_set:
        v = p["ground_truth"]["virus"]
        virus_counts[v] = virus_counts.get(v, 0) + 1
    for v, c in sorted(virus_counts.items()):
        print(f"  {v}: {c}")


if __name__ == "__main__":
    main()
