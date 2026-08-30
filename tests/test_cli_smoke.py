"""End-to-end: the quickstart contract. CI runs this on every PR (<2 min)."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import parallax_bench.cli as cli_module
from parallax_bench.cli import app

REPO = Path(__file__).resolve().parent.parent
runner = CliRunner()


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path


def test_validate_smoke(workdir):
    result = runner.invoke(app, ["validate", "--subset", "smoke",
                                 "--data-dir", str(REPO / "benchmark")])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_quickstart_run_score_report(workdir):
    data = ["--data-dir", str(REPO / "benchmark")]
    result = runner.invoke(
        app, ["run", "--system", "baseline-local", "--subset", "smoke", *data]
    )
    assert result.exit_code == 0, result.output
    assert "done=180" in result.output

    result = runner.invoke(app, ["score", *data])
    assert result.exit_code == 0, result.output

    out_dirs = list((workdir / "runs").iterdir())
    assert len(out_dirs) == 1
    metrics = pd.read_csv(out_dirs[0] / "metrics.csv")
    diag = metrics[
        (metrics.metric == "ndcg@10") & (metrics.query_lang == metrics.index_lang)
    ]
    assert len(diag) == 3
    # BM25 on the diagonal of the smoke set must be near-perfect; if this
    # drops, ingest or scoring broke — not the baseline
    assert (diag["mean"] > 0.9).all(), diag
    assert (metrics[metrics.metric == "ndcg@10"].missing_rate == 0).all()
    assert (out_dirs[0] / "parallax_summary.csv").is_file()
    assert (out_dirs[0] / "parallax_summary.json").is_file()
    assert (out_dirs[0] / "matrices" / "ndcg_at_10.csv").is_file()
    assert (out_dirs[0] / "matrices" / "ndcg_at_10_clp.csv").is_file()
    assert (out_dirs[0] / "matrices" / "ndcg_at_10_en_delta.csv").is_file()
    assert (out_dirs[0] / "matrices" / "origin_translated" / "ndcg_at_10.csv").is_file()

    pytest.importorskip("matplotlib", reason="heatmap dependency is not installed")
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0, result.output
    assert "ndcg@10" in result.output
    assert "Cross-Lingual Penalty" in result.output
    assert (out_dirs[0] / "matrices" / "absolute_ndcg_at_10_matrix.png").is_file()
    assert (out_dirs[0] / "matrices" / "parallax_ndcg_at_10_matrix.png").is_file()

    result = runner.invoke(app, ["report", "--origin", "native"])
    assert result.exit_code == 0, result.output
    assert "not available" in result.output


def test_run_is_resumable(workdir):
    data = ["--data-dir", str(REPO / "benchmark")]
    result = runner.invoke(
        app, ["run", "--system", "baseline-local", "--subset", "smoke", *data]
    )
    run_id = next(
        line.split()[2] for line in result.output.splitlines() if line.startswith("created run")
    )
    # resuming a finished run is a no-op, not an error
    result = runner.invoke(
        app,
        ["run", "--system", "baseline-local", "--subset", "smoke", *data,
         "--resume", run_id, "--no-ingest"],
    )
    assert result.exit_code == 0, result.output
    assert "done=180" in result.output


def test_generation_phase_on_smoke(workdir):
    data = ["--data-dir", str(REPO / "benchmark")]
    result = runner.invoke(
        app, ["run", "--system", "baseline-local", "--subset", "smoke",
              "--phase", "generation", *data]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["score", *data])
    assert result.exit_code == 0, result.output
    out_dir = next((workdir / "runs").iterdir())
    gen = pd.read_csv(out_dir / "generation.csv")
    assert set(gen.regime) >= {"mono", "multi"}
    # extractive baseline answers come verbatim from the target-language corpus,
    # so on the mono diagonal the detected language must match the query language
    mono = gen[gen.regime == "mono"]
    assert (mono.lang_correct.dropna().astype(bool)).mean() > 0.8
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0, result.output
    assert "generation (mechanical metrics)" in result.output
    assert "no retrieval metrics" in result.output


def test_experiment_ingests_once_for_both_phases(workdir, monkeypatch):
    ingest_calls = 0
    real_ingest = cli_module.do_ingest

    def counting_ingest(*args, **kwargs):
        nonlocal ingest_calls
        ingest_calls += 1
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(cli_module, "do_ingest", counting_ingest)
    monkeypatch.setattr(cli_module, "report", lambda **kwargs: None)
    result = runner.invoke(
        app,
        [
            "experiment",
            "--system",
            "baseline-local",
            "--subset",
            "smoke",
            "--data-dir",
            str(REPO / "benchmark"),
            "--skip-fetch",
            "--skip-verify",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment complete" in result.output
    assert "retrieval:" in result.output
    assert "generation:" in result.output
    assert ingest_calls == 1


def test_experiment_no_ingest_reuses_existing_indexes(workdir, monkeypatch):
    data = ["--data-dir", str(REPO / "benchmark")]
    result = runner.invoke(
        app, ["ingest", "--system", "baseline-local", "--subset", "smoke", *data]
    )
    assert result.exit_code == 0, result.output

    def unexpected_ingest(*args, **kwargs):
        raise AssertionError("experiment must not ingest with --no-ingest")

    monkeypatch.setattr(cli_module, "do_ingest", unexpected_ingest)
    monkeypatch.setattr(cli_module, "score", lambda **kwargs: None)
    monkeypatch.setattr(cli_module, "report", lambda **kwargs: None)
    result = runner.invoke(
        app,
        [
            "experiment",
            "--system",
            "baseline-local",
            "--subset",
            "smoke",
            *data,
            "--skip-fetch",
            "--skip-verify",
            "--no-ingest",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment complete" in result.output


def test_one_command_experiment_workflow(workdir):
    pytest.importorskip("matplotlib", reason="heatmap dependency is not installed")
    result = runner.invoke(
        app,
        [
            "experiment",
            "--system",
            "baseline-local",
            "--subset",
            "smoke",
            "--data-dir",
            str(REPO / "benchmark"),
            "--skip-fetch",
            "--skip-verify",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment complete" in result.output
    assert "retrieval:" in result.output
    assert "generation:" in result.output
    out_dirs = list((workdir / "runs").iterdir())
    assert len(out_dirs) == 2
    retrieval_dir = next(path for path in out_dirs if "-retrieval-" in path.name)
    generation_dir = next(path for path in out_dirs if "-generation-" in path.name)
    assert (retrieval_dir / "parallax_summary.json").is_file()
    assert (generation_dir / "generation.csv").is_file()
    assert not (generation_dir / "parallax_summary.json").exists()
