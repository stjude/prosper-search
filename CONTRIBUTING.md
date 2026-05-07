# Contributing to PROSPER

Thank you for your interest in contributing to PROSPER! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/stjude/prosper-search/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Relevant logs or error messages

### Suggesting Features

1. Check existing issues and discussions for similar suggestions
2. Create a new issue with the "enhancement" label
3. Describe the feature and its use case
4. Explain why it would benefit users

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`pytest tests/`)
6. Format code with black (`black prosper/`)
7. Commit with clear messages (`git commit -m 'Add amazing feature'`)
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/prosper-search.git
cd prosper-search

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and returns
- Write docstrings for all public functions and classes
- Use black for formatting
- Maximum line length: 100 characters

## Documentation

- Update docstrings when changing function signatures
- Update README.md for user-facing changes
- Add examples for new features

## Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for good test coverage

## Questions?

Feel free to open an issue for any questions about contributing!
