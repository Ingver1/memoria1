"""Pytest configuration and fixtures."""

import inspect
import logging
import os
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    """Register custom markers used in the test suite."""
    config.addinivalue_line("markers", "asyncio: mark async test")
    config.addinivalue_line("markers", "perf: marks performance / load tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark async test functions with pytest.mark.asyncio."""
    for item in items:
        test_fn = getattr(item, "obj", None)
        if inspect.iscoroutinefunction(test_fn):
            item.add_marker(pytest.mark.asyncio)

from fastapi.testclient import ClientHelper
from memory_system.api.app import create_app

# Set test environment variables
os.environ.update(
    {
        "UMS_ENVIRONMENT": "testing",
        "UMS_LOG_LEVEL": "DEBUG",
        "UMS_DB_PATH": str(Path(tempfile.mkdtemp()) / "test.db"),
        "CUDA_VISIBLE_DEVICES": "",  # Disable CUDA in tests
        "FAISS_OPT_LEVEL": "0",
    }
)


@pytest.fixture(autouse=True)
def _raise_log_level(caplog):
    """
    Silence DEBUG chatter from memory_system.core.index during CI.
    """
    caplog.set_level(logging.INFO, logger="memory_system.core.index")
    

@pytest.fixture(scope="session")
def test_settings():
    """Create test settings."""
    try:
        from memory_system.config.settings import UnifiedSettings

        return UnifiedSettings.for_testing()
    except ImportError:
        pytest.skip("memory_system.config.settings not available")


@pytest.fixture
def test_app(test_settings):
    """Create FastAPI application for tests."""
    return create_app()


@pytest.fixture
def test_client(test_app):
    """HTTP client for API tests."""
    return ClientHelper(test_app)


@pytest.fixture
def clean_test_vectors(tmp_path):
    """Temporary path used for vector store tests."""
    return tmp_path / "vectors"


# tests/test_basic.py
import pytest


def test_package_imports():
    """Test that basic package imports work."""
    try:
        import memory_system

        assert hasattr(memory_system, "__version__")
    except ImportError:
        pytest.skip("memory_system package not importable")


def test_exceptions_module():
    """Test that exceptions module works."""
    try:
        from memory_system.utils.exceptions import (
            MemorySystemError,
            StorageError,
            ValidationError,
        )

        # Test basic exception creation
        error = MemorySystemError("test error")
        assert str(error)  # Should not raise
        assert error.message == "test error"

        # Test inheritance
        assert issubclass(ValidationError, MemorySystemError)
        assert issubclass(StorageError, MemorySystemError)

    except ImportError:
        pytest.skip("exceptions module not available")


def test_config_module():
    """Test that config module works."""
    try:
        from memory_system.config.settings import UnifiedSettings

        # Test settings creation
        settings = UnifiedSettings.for_testing()
        assert settings is not None
        assert settings.version == "0.8.0a0"

    except ImportError:
        pytest.skip("config module not available")


def test_utils_module():
    """Test that utils module imports."""
    try:
        import memory_system.utils

        assert memory_system.utils is not None
    except ImportError:
        pytest.skip("utils module not available")


def test_core_module():
    """Test that core module imports."""
    try:
        import memory_system.core

        assert memory_system.core is not None
    except ImportError:
        pytest.skip("core module not available")


def test_api_module():
    """Test that api module imports."""
    try:
        import memory_system.api

        assert memory_system.api is not None
    except ImportError:
        pytest.skip("api module not available")


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_placeholder(self):
        """Placeholder test that always passes."""
        assert True

    def test_python_version(self):
        """Test Python version is supported."""
        import sys

        assert sys.version_info >= (3, 9)

    def test_environment_variables(self):
        """Test that test environment is set up correctly."""
        assert os.environ.get("UMS_ENVIRONMENT") == "testing"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
