from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="femo",                        # Replace with your package name
    version="0.3.1",                          # Initial release version
    description="Package for analysis of Fetal Movement data",
    long_description=open("README.md").read(), # Use README.md for the long description
    long_description_content_type="text/markdown",
    author="MABatin",
    author_email="leonhsn18@gmail.com",
    packages=find_packages(),                 # Automatically find all packages in `my_project`
    install_requires=required,
    python_requires=">=3.10",                  # Minimum Python version required
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
