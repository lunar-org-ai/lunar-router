#!/usr/bin/env python3
"""
Script to package weights for distribution.

Creates a zip file ready to upload to GitHub Releases, S3, or HuggingFace.

Usage:
    python scripts/package_weights.py ./weights/mmlu-v1 -o ./dist/weights-mmlu-v1.zip
"""

import os
import sys
import json
import hashlib
import zipfile
import argparse
from pathlib import Path


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def package_weights(
    weights_dir: Path,
    output_path: Path,
    name: str = None,
    version: str = "1.0.0",
) -> dict:
    """
    Package weights directory into a zip file.

    Returns metadata dict with size and sha256.
    """
    weights_dir = Path(weights_dir)
    output_path = Path(output_path)

    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    # Validate structure
    clusters_dir = weights_dir / "clusters"
    profiles_dir = weights_dir / "profiles"

    if not clusters_dir.exists():
        raise ValueError(f"Missing clusters directory: {clusters_dir}")
    if not profiles_dir.exists():
        raise ValueError(f"Missing profiles directory: {profiles_dir}")

    # Count profiles
    profiles = list(profiles_dir.glob("*.json"))
    print(f"Found {len(profiles)} model profiles")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create zip file
    print(f"Creating: {output_path}")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add all files
        for file_path in weights_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(weights_dir.parent)
                zf.write(file_path, arcname)
                print(f"  Added: {arcname}")

    # Calculate metadata
    size_bytes = output_path.stat().st_size
    sha256 = calculate_sha256(output_path)

    print(f"\n✓ Package created!")
    print(f"  Size: {size_bytes / (1024*1024):.2f} MB")
    print(f"  SHA256: {sha256}")

    return {
        "name": name or weights_dir.name,
        "version": version,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "profiles": [p.stem for p in profiles],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Package weights for distribution",
    )
    parser.add_argument(
        "weights_dir",
        type=Path,
        help="Path to weights directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output zip file path",
    )
    parser.add_argument(
        "-n", "--name",
        default=None,
        help="Package name",
    )
    parser.add_argument(
        "-v", "--version",
        default="1.0.0",
        help="Package version",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output metadata as JSON",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = Path(f"./dist/{args.weights_dir.name}.zip")

    try:
        metadata = package_weights(
            args.weights_dir,
            args.output,
            name=args.name,
            version=args.version,
        )

        if args.json:
            print("\nMetadata JSON:")
            print(json.dumps(metadata, indent=2))

        print(f"\n📦 Ready to upload: {args.output}")
        print("\nUpload to:")
        print("  - GitHub Releases: https://github.com/YOUR_ORG/lunar-router-data/releases")
        print("  - S3: aws s3 cp dist/weights.zip s3://your-bucket/")
        print("  - HuggingFace: huggingface-cli upload YOUR_ORG/lunar-router-weights ./dist/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
