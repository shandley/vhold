#!/usr/bin/env python3
"""Download go-basic.obo and build GO term name → GO ID mapping.

Parses OBO format to extract id/name pairs:
    id: GO:0039694
    name: viral genome replication

Includes EXACT synonyms for broader coverage. Outputs:
    src/vhold/data/go_term_map.json
    {"viral genome replication": "GO:0039694", ...}

Usage:
    python scripts/build_go_term_map.py
    python scripts/build_go_term_map.py --obo /path/to/go-basic.obo  # use local file
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

GO_OBO_URL = "http://current.geneontology.org/ontology/go-basic.obo"
OUTPUT_PATH = Path(__file__).parent.parent / "src" / "vhold" / "data" / "go_term_map.json"

SYNONYM_RE = re.compile(r'^synonym:\s+"(.+?)"\s+EXACT\s+')


def download_obo(url: str) -> str:
    """Download go-basic.obo and return contents."""
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")
    print(f"Downloaded {len(data):,} bytes")
    return data


def parse_obo(text: str) -> dict[str, str]:
    """Parse OBO format and extract name → GO ID mapping.

    Includes primary names and EXACT synonyms.

    Returns:
        Dict mapping lowercase GO term name → GO ID (e.g., GO:0039694)
    """
    name_to_id: dict[str, str] = {}
    current_id = None
    current_name = None
    current_synonyms: list[str] = []
    in_term = False

    def save_term():
        if current_id and current_name:
            name_to_id[current_name.lower()] = current_id
            for syn in current_synonyms:
                key = syn.lower()
                if key not in name_to_id:
                    name_to_id[key] = current_id

    for line in text.splitlines():
        line = line.strip()

        if line == "[Term]":
            save_term()
            in_term = True
            current_id = None
            current_name = None
            current_synonyms = []
            continue
        elif line.startswith("[") and line.endswith("]"):
            save_term()
            in_term = False
            current_id = None
            current_name = None
            current_synonyms = []
            continue

        if not in_term:
            continue

        if line.startswith("id: "):
            current_id = line[4:]
        elif line.startswith("name: "):
            current_name = line[6:]
        elif line.startswith("synonym:"):
            m = SYNONYM_RE.match(line)
            if m:
                current_synonyms.append(m.group(1))
        elif line == "":
            save_term()
            in_term = False
            current_id = None
            current_name = None
            current_synonyms = []

    # Handle last term
    save_term()

    return name_to_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GO term name→ID map")
    parser.add_argument("--obo", type=Path, help="Local go-basic.obo file")
    args = parser.parse_args()

    if args.obo:
        text = args.obo.read_text()
        print(f"Read {len(text):,} bytes from {args.obo}")
    else:
        text = download_obo(GO_OBO_URL)

    name_to_id = parse_obo(text)

    print(f"Parsed {len(name_to_id):,} GO term name → ID mappings")

    # Show some examples
    examples = [
        "viral genome replication",
        "RNA binding",
        "viral process",
        "DNA-templated transcription",
        "structural molecule activity",
    ]
    for term in examples:
        go_id = name_to_id.get(term.lower(), "NOT FOUND")
        print(f"  {term} → {go_id}")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(name_to_id, f, sort_keys=True, indent=None, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
