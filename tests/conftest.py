"""Pytest configuration for distribution fitting tests."""
import os


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Enable fast fitting mode (fewer iterations and restarts).",
    )


def pytest_configure(config):
    if config.getoption("--fast"):
        os.environ["RP_FAST"] = "1"
