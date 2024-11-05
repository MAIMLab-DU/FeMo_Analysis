"""Provides utilities for SageMaker Pipeline CLI."""
from __future__ import absolute_import

import ast


def get_pipeline_driver(module_name, passed_args=None):
    """Gets the driver for generating your pipeline definition.

    Pipeline modules must define a get_pipeline() module-level method.

    Args:
        module_name: The module name of your pipeline.
        passed_args: Optional passed arguments that your pipeline may be templated by.

    Returns:
        The SageMaker Workflow pipeline.
    """
    _imports = __import__(module_name, fromlist=["get_pipeline"])
    kwargs = passed_args
    return _imports.get_pipeline(**kwargs)


def convert_struct(str_struct=None):
    # Parse the string structure
    result = ast.literal_eval(str_struct) if str_struct else {}
    
    # Convert "local_mode" to a boolean if it exists and is a string
    if "local_mode" in result and isinstance(result["local_mode"], str):
        result["local_mode"] = result["local_mode"].lower() == "true"
    
    return result