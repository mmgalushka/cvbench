from click.testing import CliRunner

from cvbench.cli.train import train


def test_train_rejects_unsafe_output_name(tmp_path):
    result = CliRunner().invoke(train, [str(tmp_path), "--output", str(tmp_path / "x --lr 1e-3")])
    assert result.exit_code == 2
    assert "--output" in result.output


def test_train_rejects_existing_output_without_overwrite(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    result = CliRunner().invoke(train, [str(tmp_path), "--output", str(run)])
    assert result.exit_code == 2
    assert "already exists" in result.output
