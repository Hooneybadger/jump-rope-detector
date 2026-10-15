from pathlib import Path

import pytest

from tools.eval import render_validation


@pytest.mark.parametrize("mode", ["basic", "alternating", "double"])
def test_validation_mode_filter_preserves_output_root(tmp_path, monkeypatch, mode):
    root = tmp_path / "inputs"
    manifest = root / "external_data" / "manifest"
    manifest.mkdir(parents=True)
    (manifest / "sources.csv").write_text("id,title,channel,url\n", encoding="utf-8")
    monkeypatch.setattr(render_validation, "ROOT", root)
    monkeypatch.setattr(render_validation, "bench_records", lambda _: [
        dict(id=f"{kind}_01", mode=kind) for kind in ("basic", "alternating", "double")
    ])
    monkeypatch.setattr(render_validation, "json_records", lambda *_: [
        dict(id=f"-{kind}_seg0", mode=kind, video_id=f"-{kind}")
        for kind in ("basic", "alternating", "double")
    ])
    output = tmp_path / "new_validation"
    jobs = render_validation.jobs(set(), mode=mode, out_dir=output)
    assert [job["record"]["id"] for job in jobs] == [f"{mode}_01", f"-{mode}_seg0"]
    assert all(Path(job["out"]).is_relative_to(output) for job in jobs)
    only = render_validation.jobs({f"-{mode}_seg0"}, mode=mode, out_dir=output)
    assert len(only) == 1
    assert Path(only[0]["out"]).parent == output / "external"
