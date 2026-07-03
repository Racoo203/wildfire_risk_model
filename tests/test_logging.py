"""Regression test: a log line written by a deeply-nested module logger
must actually reach the configured file handler (guards against bug #1
recurring: logger-name mismatch silently swallowing records)."""

import logging
from wildfire_susceptibility.utils.logger import setup_logger

def test_child_logger_propagates_to_file_handler(tmp_path):
    log_file = tmp_path / "pipeline.log"
    setup_logger(log_file=log_file, level="INFO")

    # Simulate a deeply nested module doing logging.getLogger(__name__)
    child_logger = logging.getLogger("wildfire_susceptibility.features.topography")
    child_logger.info("synthetic test message 12345")

    for handler in logging.getLogger("wildfire_susceptibility").handlers:
        handler.flush()

    assert log_file.exists()
    contents = log_file.read_text()
    assert "synthetic test message 12345" in contents