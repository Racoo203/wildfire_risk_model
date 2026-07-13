"""Feature builder tests. Scope: currently covers BoundaryBuilder's
return-value contract only (fixed alongside the Phase 1 stage-contract
work). Broader feature-builder coverage remains Phase 7 debt."""

from wildfire_susceptibility.features.boundary import BoundaryBuilder


def test_boundary_process_returns_output_paths_on_fresh_build(
    minimal_config, synthetic_boundary, monkeypatch
):
    """Regression test: process() used to return None on a fresh build
    and only return the output_paths dict on a cache hit — breaking any
    caller relying on the return value (e.g. stage_preprocess)."""
    config = minimal_config
    config["data_sources"] = {"cua": {"data_dir": str(synthetic_boundary.parent)}}

    # BoundaryBuilder reads a specific filename from data_dir; point it
    # at our synthetic fixture by renaming, since the fixture already
    # writes a valid Essex-attributed shapefile.
    expected_src = synthetic_boundary.parent / "CTYUA_DEC_2024_UK_BFC.shp"
    if not expected_src.exists():
        synthetic_boundary.rename(expected_src)
        # shapefiles are multi-file; rename sidecars too
        for ext in (".shx", ".dbf", ".prj", ".cpg"):
            sidecar = synthetic_boundary.with_suffix(ext)
            if sidecar.exists():
                sidecar.rename(expected_src.with_suffix(ext))

    builder = BoundaryBuilder(config)
    result = builder.process()

    assert result is not None
    assert "boundary" in result
    assert result["boundary"].exists()


def test_boundary_process_returns_output_paths_on_cache_hit(minimal_config, synthetic_boundary):
    """Cache-hit path already worked before the fix — guard against
    regressing it while fixing the fresh-build path."""
    config = minimal_config
    output_dir = config["base"]["output_dir"]
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cached_path = Path(output_dir) / "boundary.shp"
    synthetic_boundary.rename(cached_path)
    for ext in (".shx", ".dbf", ".prj", ".cpg"):
        sidecar = synthetic_boundary.with_suffix(ext)
        if sidecar.exists():
            sidecar.rename(cached_path.with_suffix(ext))

    config["data_sources"] = {"cua": {"data_dir": output_dir}}
    builder = BoundaryBuilder(config)
    result = builder.process()

    assert result == {"boundary": cached_path}