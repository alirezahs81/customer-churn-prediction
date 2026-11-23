"""Tests for the dataset download and integrity check."""
import hashlib
from io import BytesIO

import pytest

from src import download_data


def test_download_dataset_verifies_hash_and_saves_file(tmp_path, monkeypatch):
    payload = b"customerID,Churn\nA1,Yes\nB2,No\n"
    expected_hash = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        download_data,
        "urlopen",
        lambda url, timeout: BytesIO(payload),
    )

    destination = tmp_path / "data" / "sample.csv"
    saved_path = download_data.download_dataset(
        destination=destination,
        url="https://example.test/sample.csv",
        expected_sha256=expected_hash,
    )

    assert saved_path == destination
    assert destination.read_bytes() == payload


def test_download_dataset_rejects_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"not the expected dataset"
    monkeypatch.setattr(
        download_data,
        "urlopen",
        lambda url, timeout: BytesIO(payload),
    )

    destination = tmp_path / "sample.csv"
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_data.download_dataset(
            destination=destination,
            url="https://example.test/sample.csv",
            expected_sha256="0" * 64,
        )

    assert not destination.exists()
