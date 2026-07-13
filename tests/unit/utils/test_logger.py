import logging
from wildfire_susceptibility.utils.logger import setup_logger

def test_child_logger_propagates_to_file_handler(tmp_path):
    # Isolate from any handlers a prior test attached to the same
    # module-level logger name — setup_logger() intentionally
    # short-circuits if handlers already exist (avoids duplicate
    # handlers across repeated calls within one real pipeline run),
    # which breaks test isolation across a shared pytest process.
    logging.getLogger("wildfire_susceptibility").handlers.clear()

    log_file = tmp_path / "pipeline.log"
    setup_logger(log_file=log_file, level="INFO")

    child_logger = logging.getLogger("wildfire_susceptibility.features.topography")
    child_logger.info("synthetic test message 12345")

    for handler in logging.getLogger("wildfire_susceptibility").handlers:
        handler.flush()

    assert log_file.exists()
    contents = log_file.read_text()
    assert "synthetic test message 12345" in contents