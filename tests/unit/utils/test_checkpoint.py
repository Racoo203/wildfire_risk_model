import time

from wildfire_susceptibility.utils.checkpoint import (
    cache_is_valid,
    compute_cache_signature,
    write_cache_signature,
)


def _touch(path, content="x"):
    path.write_text(content)


def test_cache_is_valid_when_nothing_changed(tmp_path, minimal_modeling_config):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sig_path = tmp_path / "output.sig.json"
    _touch(input_path)
    _touch(output_path)

    write_cache_signature(sig_path, compute_cache_signature(minimal_modeling_config, [input_path]))

    assert cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])


def test_cache_is_invalid_when_output_missing(tmp_path, minimal_modeling_config):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"  # never written
    sig_path = tmp_path / "output.sig.json"
    _touch(input_path)

    write_cache_signature(sig_path, compute_cache_signature(minimal_modeling_config, [input_path]))

    assert not cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])


def test_cache_is_invalid_when_sig_file_missing(tmp_path, minimal_modeling_config):
    """A cache written before this staleness check existed (or by any
    stage that hasn't been wired to write_cache_signature) has no sidecar
    -- must be treated as stale rather than silently trusted, matching the
    exact on-disk state that caused the original class-4 bug."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sig_path = tmp_path / "output.sig.json"  # never written
    _touch(input_path)
    _touch(output_path)

    assert not cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])


def test_cache_is_invalid_when_input_regenerated_after_output(tmp_path, minimal_modeling_config):
    """Reproduces the actual incident: an upstream input file gets
    regenerated (new mtime, new content -- e.g. stage_integration re-ran
    with a new label scheme) after the downstream cached output was
    written. The old existence-only check would have kept serving the
    stale output; this must now detect the mismatch and refuse to."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sig_path = tmp_path / "output.sig.json"

    _touch(input_path, "old raw data, 4 classes")
    _touch(output_path, "clean data derived from old raw data")
    write_cache_signature(sig_path, compute_cache_signature(minimal_modeling_config, [input_path]))

    assert cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])

    # Upstream input regenerated -- output was never recomputed from it.
    time.sleep(0.01)
    _touch(input_path, "new raw data, 5 classes")

    assert not cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])


def test_cache_is_invalid_when_config_changes(tmp_path, minimal_modeling_config):
    """Reproduces the other half of the incident: config fields that
    govern the stage's output (e.g. labels.n_classes, labels.clean_labels)
    changed without the upstream input file being touched at all."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sig_path = tmp_path / "output.sig.json"
    _touch(input_path)
    _touch(output_path)

    write_cache_signature(sig_path, compute_cache_signature(minimal_modeling_config, [input_path]))
    assert cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])

    minimal_modeling_config["labels"]["n_classes"] = 5
    assert not cache_is_valid([output_path], sig_path, minimal_modeling_config, [input_path])
