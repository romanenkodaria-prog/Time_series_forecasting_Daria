"""Smoke tests for the installed package."""

import importlib
from importlib.metadata import version


def test_package_can_be_imported():
    package = importlib.import_module("sales_forecasting")

    assert package.__name__ == "sales_forecasting"


def test_distribution_is_installed():
    assert version("sales-forecasting") == "0.1.0"


def test_package_has_documentation():
    package = importlib.import_module("sales_forecasting")

    assert package.__doc__
