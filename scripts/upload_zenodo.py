#!/usr/bin/env python3
"""Upload vHold supplementary models and databases to Zenodo.

Creates a new Zenodo deposit, uploads files, and publishes.

Usage:
    ZENODO_TOKEN=your_token python scripts/upload_zenodo.py

    # Sandbox (for testing):
    ZENODO_TOKEN=your_token python scripts/upload_zenodo.py --sandbox

    # Dry run (check files exist without uploading):
    python scripts/upload_zenodo.py --dry-run

Files uploaded:
    - vhold_embeddings.npz (822 MB) - ProstT5 reference embeddings
    - bfvd_metadata_enriched.tsv (79 MB) - Enriched BFVD annotations
    - vhold_classifier.pt (2.6 MB) - MLP functional classifier

After publishing, update src/vhold/utils/constants.py with the record ID.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# Files to upload with their local paths and descriptions
FILES = [
    {
        "local_path": Path.home() / ".vhold/databases/embeddings/vhold_embeddings.npz",
        "filename": "vhold_embeddings.npz",
        "description": (
            "Pre-computed ProstT5 encoder embeddings for 436,237 viral protein "
            "structures (351K BFVD + 85K Viro3D). Float16, L2-normalized, 1024 dimensions. "
            "Used for embedding-based triage (--triage)."
        ),
    },
    {
        "local_path": Path.home() / ".vhold/databases/bfvd/bfvd_metadata_enriched.tsv",
        "filename": "bfvd_metadata_enriched.tsv",
        "description": (
            "Enriched BFVD metadata with UniProt annotations for 345,730 proteins. "
            "Contains protein names, gene names, GO terms, Pfam domains, keywords, "
            "function descriptions, and taxonomic lineage from UniProt REST API."
        ),
    },
    {
        "local_path": Path.home() / ".vhold/models/classifier/vhold_classifier.pt",
        "filename": "vhold_classifier.pt",
        "description": (
            "Trained MLP functional classifier (1024->512->256->11) for viral protein "
            "category prediction. Macro F1 0.692, 660K parameters, 2.6 MB."
        ),
    },
]

DEPOSIT_METADATA = {
    "metadata": {
        "title": "vHold: Supplementary Models and Databases",
        "upload_type": "dataset",
        "description": (
            "<p>Supplementary data files for "
            "<a href='https://github.com/shandley/vhold'>vHold</a>, "
            "a viral protein annotation tool using structural homology "
            "(ProstT5 + Foldseek).</p>"
            "<p>This record contains:</p>"
            "<ul>"
            "<li><b>vhold_embeddings.npz</b> (822 MB): Pre-computed ProstT5 encoder "
            "embeddings for 436,237 viral protein structures from BFVD and Viro3D. "
            "Used for fast embedding-based triage.</li>"
            "<li><b>bfvd_metadata_enriched.tsv</b> (79 MB): Enriched BFVD metadata "
            "with UniProt annotations (protein names, GO terms, Pfam domains, "
            "keywords, taxonomic lineage) for 345,730 proteins.</li>"
            "<li><b>vhold_classifier.pt</b> (2.6 MB): Trained MLP functional "
            "classifier for viral protein category prediction (macro F1 0.692).</li>"
            "</ul>"
            "<p>Install via: <code>vhold install --embeddings --classifier</code></p>"
        ),
        "creators": [
            {"name": "Handley, Scott A.", "affiliation": "Washington University in St. Louis"}
        ],
        "keywords": [
            "viral proteins",
            "structural bioinformatics",
            "ProstT5",
            "protein embeddings",
            "functional annotation",
            "Foldseek",
        ],
        "license": "MIT",
        "related_identifiers": [
            {
                "identifier": "https://github.com/shandley/vhold",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
    }
}


def get_api_base(sandbox: bool) -> str:
    """Get the Zenodo API base URL."""
    if sandbox:
        return "https://sandbox.zenodo.org/api"
    return "https://zenodo.org/api"


def create_deposit(token: str, base_url: str) -> dict:
    """Create a new empty deposit."""
    response = requests.post(
        f"{base_url}/deposit/depositions",
        params={"access_token": token},
        json={},
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def update_metadata(token: str, base_url: str, deposit_id: int, metadata: dict) -> dict:
    """Update deposit metadata."""
    response = requests.put(
        f"{base_url}/deposit/depositions/{deposit_id}",
        params={"access_token": token},
        json=metadata,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def upload_file(token: str, bucket_url: str, filepath: Path, filename: str) -> dict:
    """Upload a file to a deposit bucket."""
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Uploading {filename} ({size_mb:.1f} MB)...")

    with open(filepath, "rb") as f:
        response = requests.put(
            f"{bucket_url}/{filename}",
            data=f,
            params={"access_token": token},
        )
    response.raise_for_status()
    result = response.json()
    checksum = result.get("checksum", "unknown")
    print(f"    Done. Checksum: {checksum}")
    return result


def publish_deposit(token: str, base_url: str, deposit_id: int) -> dict:
    """Publish a deposit (makes it permanent and assigns DOI)."""
    response = requests.post(
        f"{base_url}/deposit/depositions/{deposit_id}/actions/publish",
        params={"access_token": token},
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Upload vHold files to Zenodo")
    parser.add_argument(
        "--sandbox", action="store_true",
        help="Use Zenodo sandbox (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check files exist without uploading",
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="Create deposit and upload but don't publish (allows review)",
    )
    args = parser.parse_args()

    # Check files exist
    print("Checking local files...")
    all_ok = True
    for file_info in FILES:
        path = file_info["local_path"]
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {file_info['filename']}: {size_mb:.1f} MB")
        else:
            print(f"  {file_info['filename']}: MISSING ({path})")
            all_ok = False

    if not all_ok:
        print("\nSome files are missing. Cannot proceed.")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. All files found.")
        sys.exit(0)

    # Get token
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("\nError: ZENODO_TOKEN environment variable not set.")
        print("Get a token from: https://zenodo.org/account/settings/applications/")
        sys.exit(1)

    base_url = get_api_base(args.sandbox)
    env_label = "SANDBOX" if args.sandbox else "PRODUCTION"
    print(f"\nUsing Zenodo {env_label}: {base_url}")

    # Create deposit
    print("\nCreating deposit...")
    deposit = create_deposit(token, base_url)
    deposit_id = deposit["id"]
    bucket_url = deposit["links"]["bucket"]
    print(f"  Deposit ID: {deposit_id}")
    print(f"  Bucket URL: {bucket_url}")

    # Update metadata
    print("\nSetting metadata...")
    update_metadata(token, base_url, deposit_id, DEPOSIT_METADATA)
    print("  Done.")

    # Upload files
    print("\nUploading files...")
    for file_info in FILES:
        upload_file(token, bucket_url, file_info["local_path"], file_info["filename"])

    if args.no_publish:
        print(f"\nDeposit created but NOT published (--no-publish).")
        print(f"  Review at: https://{'sandbox.' if args.sandbox else ''}zenodo.org/deposit/{deposit_id}")
        print(f"\n  To publish manually, visit the URL above and click 'Publish'.")
    else:
        # Publish
        print("\nPublishing...")
        published = publish_deposit(token, base_url, deposit_id)
        record_id = published["id"]
        doi = published["metadata"]["doi"]
        record_url = published["links"]["record_html"]

        print(f"\n{'=' * 60}")
        print(f"Published successfully!")
        print(f"{'=' * 60}")
        print(f"  Record ID: {record_id}")
        print(f"  DOI: {doi}")
        print(f"  URL: {record_url}")
        print(f"\n  File URLs:")
        files_base = f"https://{'sandbox.' if args.sandbox else ''}zenodo.org/records/{record_id}/files"
        for file_info in FILES:
            print(f"    {files_base}/{file_info['filename']}")

        print(f"\n  Update src/vhold/utils/constants.py:")
        print(f'    VHOLD_ZENODO_BASE_URL = "{files_base}"')

    print("\nDone.")


if __name__ == "__main__":
    main()
