#!/usr/bin/env bash
# Downloads the Elliptic Bitcoin dataset (CC BY-NC-ND 4.0 -- do not redistribute).
# Requires a Kaggle API token at ~/.kaggle/kaggle.json and accepted dataset terms.
set -euo pipefail
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
