"""Data loading, merging and validation on a tiny synthetic Elliptic-shaped dataset."""

import numpy as np
import pandas as pd
import pytest

from driftguard import data
from driftguard.data import DataValidationError


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for tx, step in [(10, 1), (11, 1), (12, 1), (20, 2), (21, 2)]:
        rows.append([tx, step, *rng.normal(size=93 + 72).round(4)])
    nested = tmp_path / "elliptic_bitcoin_dataset"
    nested.mkdir()
    pd.DataFrame(rows).to_csv(nested / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId": [10, 11, 12, 20, 21], "class": ["1", "2", "unknown", "2", "1"]}).to_csv(
        nested / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame({"txId1": [10, 10, 11, 20], "txId2": [11, 12, 12, 21]}).to_csv(
        nested / "elliptic_txs_edgelist.csv", index=False
    )
    return tmp_path


def _load(raw_dir):
    find = lambda name: data.find_raw_file(name, [raw_dir])  # noqa: E731
    features = data.read_features(find("elliptic_txs_features.csv"))
    classes = data.read_classes(find("elliptic_txs_classes.csv"))
    edges = data.read_edges(find("elliptic_txs_edgelist.csv"))
    return data.merge(features, classes, edges), edges


def test_find_raw_file_searches_nested_folders(raw_dir):
    assert data.find_raw_file("elliptic_txs_classes.csv", [raw_dir]).name == "elliptic_txs_classes.csv"
    with pytest.raises(FileNotFoundError):
        data.find_raw_file("missing.csv", [raw_dir])


def test_merge_maps_labels_and_names_columns(raw_dir):
    df, _ = _load(raw_dir)
    assert list(df.columns[:3]) == ["txId", "step", "y"]
    assert "local_93" in df.columns and "agg_72" in df.columns
    y = df.set_index("txId")["y"]
    assert y[10] == 1 and y[11] == 0 and np.isnan(y[12])


def test_degree_features(raw_dir):
    df, _ = _load(raw_dir)
    d = df.set_index("txId")
    assert d.loc[10, "out_degree"] == 2 and d.loc[10, "in_degree"] == 0
    assert d.loc[12, "in_degree"] == 2 and d.loc[12, "out_degree"] == 0


def test_validate_passes_on_expected_counts(raw_dir):
    df, edges = _load(raw_dir)
    stats = data.validate(df, edges, expected_rows=5, expected_steps=2, expected_illicit=2, expected_licit=2)
    assert stats["unknown"] == 1


def test_validate_fails_loudly_on_wrong_counts(raw_dir):
    df, edges = _load(raw_dir)
    with pytest.raises(DataValidationError, match="illicit"):
        data.validate(df, edges, expected_rows=5, expected_steps=2, expected_illicit=3, expected_licit=2)


def test_validate_rejects_edges_crossing_steps(raw_dir):
    df, edges = _load(raw_dir)
    bad = pd.concat([edges, pd.DataFrame({"txId1": [10], "txId2": [20]})], ignore_index=True)
    with pytest.raises(DataValidationError, match="cross time steps"):
        data.validate(df, bad, expected_rows=5, expected_steps=2, expected_illicit=2, expected_licit=2)
