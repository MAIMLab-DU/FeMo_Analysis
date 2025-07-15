"""Setup script for building C++ extensions for the FeMo package."""

import sys
from pathlib import Path
from typing import Dict, List

from setuptools import setup, Extension
import pybind11
import numpy


def get_compiler_arguments() -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get compiler-specific compilation and linking arguments.
    
    Returns:
        Tuple of (compile_args_dict, link_args_dict) for different platforms
    """
    extra_compile_args: Dict[str, List[str]] = {
        'msvc': ['/EHsc', '/std:c++17'],
        'unix': ['-std=c++17', '-Wno-unused-variable', '-Wno-deprecated-declarations'],
    }

    extra_link_args: Dict[str, List[str]] = {
        'msvc': [],
        'unix': [],
    }

    # Special handling for macOS
    if sys.platform == 'darwin':
        extra_compile_args['unix'].extend(['-stdlib=libc++'])
        extra_link_args['unix'].extend(['-stdlib=libc++'])
    
    return extra_compile_args, extra_link_args


def create_extension_modules() -> List[Extension]:
    """
    Create the C++ extension modules for compilation.
    
    Returns:
        List of Extension objects to be built
    """
    extra_compile_args, extra_link_args = get_compiler_arguments()
    platform_key: str = 'msvc' if sys.platform == 'win32' else 'unix'
    
    # Define source files using pathlib for cross-platform compatibility
    cpp_source_directory: Path = Path('femo/data/transforms/formats/v2core')
    cpp_source_files: List[str] = [
        str(cpp_source_directory / 'parser.cpp'),
        str(cpp_source_directory / 'bindings.cpp')
    ]
    
    # Verify source files exist
    missing_files: List[str] = []
    for source_file in cpp_source_files:
        if not Path(source_file).exists():
            missing_files.append(source_file)
    
    if missing_files:
        raise FileNotFoundError(f"Required C++ source files not found: {missing_files}")
    
    include_directories: List[str] = [
        pybind11.get_include(),
        numpy.get_include()
    ]
    
    extension_modules: List[Extension] = [
        Extension(
            name='femo_parser_cpp',
            sources=cpp_source_files,
            include_dirs=include_directories,
            language='c++',
            extra_compile_args=extra_compile_args.get(platform_key, []),
            extra_link_args=extra_link_args.get(platform_key, []),
        ),
    ]
    
    return extension_modules


if __name__ == "__main__":
    extension_modules: List[Extension] = create_extension_modules()
    
    # Only define extension-specific setup parameters
    # Main package configuration comes from pyproject.toml
    setup(
        ext_modules=extension_modules,
        zip_safe=False,
    )