# Introduction
This repository holds code to interview task for Wattstor company.
This repository contains one Jupyter notebook, which contanins analysis of the data for better understanding and to choose proper methods.

It also comes with Pipfile containing all the requirements needed. To install them, you'll need Pipenv virtual environment.

# Initializing Pipenv

## Introduction
Pipenv is a dependency manager for Python projects. It creates and manages a virtual environment for your projects and adds/removes packages from your `Pipfile` as you install/uninstall packages.

## Prerequisites
- Python installed on your system
- Basic understanding of command-line operations

## Installation
1. **Install Pipenv**: If pipenv is not installed, you can install it using pip:
   ```bash
   pip install pipenv
   ```
2. **Change directory**: 
    ```bash
    cd path/to/your/project
    ```
3. **Activate pipenv**:
    ```bash
    pipenv shell
    ```
4. **Install necessary packages**:
    ```bash
    pipenv install
    ```
# Example startup

As part of the solution two methods were implemented. First of them is AutoRegressive model.

```python
python script.py --input ./data/SG.csv --quantity Consumption --method AR
```

The second option is to run Convolutional Neural Network. Example usage is:
```python
python script.py --input ./data/SG.csv --quantity Consumption --method CNN
```

# Test
This repository also contains unit tests using `pytest` module. This module will be installed while setupping virtual environment.

To run unit test, use following command
```bash
pytest -vv
```
to run all files whose names are starting with `test`.
To run specific test file, use command
```bash
pytest test_file.py -vv
```
To run specific test fucntion in specific test file, use command
```bash
pytest test_file.py::test_name -vv
```