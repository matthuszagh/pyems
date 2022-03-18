import subprocess
import os

os.environ["_PYEMS_PYTEST"] = "1"


def example_common(file_base_name):
    result = subprocess.run(["python3", f"examples/{file_base_name}.py"])
    assert result.returncode == 0


def test_example_coax():
    example_common("coax")


def test_example_coupler():
    example_common("coupler")


def test_example_differential_gcpw_blocking_cap():
    example_common("differential_gcpw_blocking_cap")


def test_example_differential_gcpw():
    example_common("differential_gcpw")


def test_example_gcpw_blocking_cap():
    example_common("gcpw_blocking_cap")


def test_example_gcpw_bypass_cap():
    example_common("gcpw_bypass_cap")


def test_example_gcpw():
    example_common("gcpw")


def test_example_gcpw_sma_transition():
    example_common("gcpw_sma_transition")


def test_example_horn_antenna():
    example_common("horn_antenna")


def test_example_microstrip():
    example_common("microstrip")


def test_example_microstrip_sma_transition():
    example_common("microstrip_sma_transition")


def test_example_miter():
    example_common("miter")


def test_example_no_miter():
    example_common("no_miter")


def test_example_rf_via():
    example_common("rf_via")


def test_example_rf_via_reverse():
    example_common("rf_via_reverse")
