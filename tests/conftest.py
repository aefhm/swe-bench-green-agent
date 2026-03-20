import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL for integration tests (default: http://localhost:9009)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a running green agent server"
    )
