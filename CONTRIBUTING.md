
# Contributing to FeMo_Analysis Project

Welcome! We're excited that you're considering contributing to our [FeMo_Analysis](https://github.com/MAIMLab/FeMo_Analysis) project. Your involvement plays a crucial role in improving the quality of our analysis tools, benefiting both the research and medical communities. This guide provides clear guidelines and best practices to help you get started with contributing to the FeMo_Analysis repository, and we look forward to your valuable input in advancing fetal movement analysis.

## Table of Contents

- [Contributing to FeMo\_Analysis Project](#contributing-to-femo_analysis-project)
  - [Table of Contents](#table-of-contents)
  - [Contributing via Pull Requests](#contributing-via-pull-requests)
    - [Google-Style Docstrings](#google-style-docstrings)
      - [Example](#example)
      - [Example with type hints](#example-with-type-hints)
      - [Example Single-line](#example-single-line)
    - [GitHub Actions CI Tests](#github-actions-ci-tests)
  - [Reporting Bugs](#reporting-bugs)

## Contributing via Pull Requests

We greatly appreciate contributions in the form of pull requests. To make the review process as smooth as possible, please follow these steps:

1. **[Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo):** Start by forking the Ultralytics YOLO repository to your GitHub account.

2. **[Create a branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop):** Create a new branch in your forked repository with a clear, descriptive name that reflects your changes.

3. **[Make your changes]():** Ensure your code adheres to the project's style guidelines and does not introduce any new errors or warnings.

4. **[Test your changes](https://github.com/MAIMLab/FeMo_Analysis/tree/main/tests):** Before submitting, test your changes locally to confirm they work as expected and don't cause any new issues.

5. **[Commit your changes](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop):** Commit your changes with a concise and descriptive commit message. If your changes address a specific issue, include the issue number in your commit message.

6. **[Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request):** Submit a pull request from your forked repository to the main Ultralytics YOLO repository. Provide a clear and detailed explanation of your changes and how they improve the project.

### Google-Style Docstrings

When adding new functions or classes, please include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html). These docstrings provide clear, standardized documentation that helps other developers understand and maintain your code.

#### Example

This example illustrates a Google-style docstring. Ensure that both input and output `types` are always enclosed in parentheses, e.g., `(bool)`.

```python
def example_function(arg1, arg2=4):
    """
    Example function demonstrating Google-style docstrings.

    Args:
        arg1 (int): The first argument.
        arg2 (int): The second argument, with a default value of 4.

    Returns:
        (bool): True if successful, False otherwise.

    Examples:
        >>> result = example_function(1, 2)  # returns False
    """
    if arg1 == arg2:
        return True
    return False
```

#### Example with type hints

This example includes both a Google-style docstring and type hints for arguments and returns, though using either independently is also acceptable.

```python
def example_function(arg1: int, arg2: int = 4) -> bool:
    """
    Example function demonstrating Google-style docstrings.

    Args:
        arg1: The first argument.
        arg2: The second argument, with a default value of 4.

    Returns:
        True if successful, False otherwise.

    Examples:
        >>> result = example_function(1, 2)  # returns False
    """
    if arg1 == arg2:
        return True
    return False
```

#### Example Single-line

For smaller or simpler functions, a single-line docstring may be sufficient. The docstring must use three double-quotes, be a complete sentence, start with a capital letter, and end with a period.

```python
def example_small_function(arg1: int, arg2: int = 4) -> bool:
    """Example function with a single-line docstring."""
    return arg1 == arg2
```

### GitHub Actions CI Tests

All pull requests must pass the GitHub Actions [Continuous Integration](https://github.com/MAIMLab/FeMo_Analysis/actions/workflows/ci.yml) (CI) tests before they can be merged. These tests include linting, unit tests, and other checks to ensure that your changes meet the project's quality standards. Review the CI output and address any issues that arise.

## Reporting Bugs

We highly value bug reports as they help us maintain the quality of our projects. When reporting a bug, please provide a simple, clear code example that consistently reproduces the issue. This allows us to quickly identify and resolve the problem.
