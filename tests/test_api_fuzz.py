"""
Fuzzes the whole FastAPI surface against its OpenAPI schema.
Schemathesis autogenerates thousands of requests with random payloads.
"""
import schemathesis
from schemathesis import DataGenerationMethod
import pytest
from memory_system.api.app import create_app
from memory_system.config.settings import UnifiedSettings


schema = schemathesis.from_path(
    "tests/st_api_fuzz.yaml",
    data_generation_methods=[DataGenerationMethod.fuzzed],
)


@schema.parametrize()
def test_api_fuzz(case):
    """Run schema-driven fuzzing against the live ASGI app."""
    app = create_app(UnifiedSettings.for_testing())
    response = case.call_asgi(app)
    case.validate_response(response)
