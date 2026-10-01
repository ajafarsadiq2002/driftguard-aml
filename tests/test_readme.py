"""README numbers must be exactly what results.json produces (no hand-edited results)."""

import json

import pytest

from driftguard import config
from driftguard.readme import README, render

RESULTS = config.ARTIFACTS_DIR / "results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="results.json not generated yet")
def test_readme_generated_blocks_match_results_json():
    text = README.read_text(encoding="utf-8")
    assert render(text, json.loads(RESULTS.read_text())) == text, (
        "README is out of date with artifacts/results.json. Run: python -m driftguard.readme"
    )


def test_render_requires_block_markers():
    with pytest.raises(ValueError):
        render("# no markers here", {})
