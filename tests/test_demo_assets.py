from pathlib import Path

from scripts.generate_demo_assets import generate_demo_assets


def test_generate_demo_assets(tmp_path) -> None:
    assets = generate_demo_assets(tmp_path)

    expected = {
        "prep",
        "prep_trace",
        "transcript",
        "review",
        "interview_trace",
    }
    assert expected == set(assets)

    prep = Path(assets["prep"]).read_text(encoding="utf-8")
    transcript = Path(assets["transcript"]).read_text(encoding="utf-8")
    review = Path(assets["review"]).read_text(encoding="utf-8")

    assert "# Demo Prep Run" in prep
    assert "# Demo Mock Interview" in transcript
    assert "Transcript:" in transcript
    assert "## Next Drill" in review
