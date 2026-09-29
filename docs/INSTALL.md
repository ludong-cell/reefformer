# Installation

## Supported systems

The reference workflow is platform independent. It has been exercised on
Windows with Python 3.13 and PyTorch, and is designed for Python 3.10--3.13.
The archived training runs used CUDA; reproduction by checkpoint inference can
also run on CPU.

## Virtual environment

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For a CUDA-specific PyTorch wheel, follow the official PyTorch installation
selector first, then run the editable installation command above.

## Conda

```bash
conda env create -f environment.yml
conda activate reef-former
python -m pip install -e . --no-deps
```

## Installation check

```bash
python scripts/run_demo.py
pytest
```
