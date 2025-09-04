# Contributing Guidelines

Thank you for considering contributing to Pydantic Prompter project! All types of contributions are encouraged and valued.

## How to Contribute

### Reporting Bugs

If you find bugs or issues with the code, please open an issue explaining the problem. Provide as much detail as possible including steps to reproduce the bug.

### Suggesting Enhancements

New feature ideas or enhancements are also welcome as GitHub issues. Explain your idea with specifics on why it would be useful to the project.

### Pull Requests

Pull requests are highly appreciated. Please follow these guidelines:

- Fork the repository and create a new branch for each feature or fix.
- Add tests for any new functionality or changes. Tests should pass locally before opening a PR.
- Follow the existing code style and standards. Use PEP8 style guidelines.
- Make sure git commit messages are clear and descriptive.
- Update any related documentation such as README or inline comments.
- Open a pull request against the `main` branch.

## Getting Started

This project uses `uv` for dependency and environment management. To set up your system locally:

1. Install `uv`: `pip install uv`
2. Clone the repo
3. Create a virtual environment: `uv venv`
4. Install dependencies: `uv pip sync uv.lock`
5. Install the project in editable mode: `uv pip install -e .`


## Testing

Tests are written using `pytest`. To run all tests, first activate the virtual environment: `source .venv/bin/activate`, then run:

```
pytest
```

## Licensing

This project is licensed under the MIT license. See [LICENSE](LICENSE) for details.

Let me know if you would like me to modify or add anything to this CONTRIBUTING guide!
