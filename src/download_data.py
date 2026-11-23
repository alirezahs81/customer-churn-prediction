"""Download and verify the IBM Telco Customer Churn sample dataset."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEFAULT_PATH = ROOT / "data" / "telco_customer_churn.csv"


def download_dataset(
    destination: Path = DEFAULT_PATH,
    url: str = DATA_URL,
    expected_sha256: str = DATA_SHA256,
) -> Path:
    """Download the source CSV, verify its SHA-256, then write it locally."""
    with urlopen(url, timeout=30) as response:
        content = response.read()

    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Dataset checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}. "
            "The file was not saved."
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PATH,
        help="Destination CSV path (default: data/telco_customer_churn.csv).",
    )
    args = parser.parse_args()
    saved_path = download_dataset(destination=args.output)
    print(f"Verified dataset saved to: {saved_path}")


if __name__ == "__main__":
    main()
