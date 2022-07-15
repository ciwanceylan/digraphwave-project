# Digraphwave

[![tests-pip](https://github.com/ciwanceylan/digraphwave/actions/workflows/tests-pip.yml/badge.svg?branch=dev)](https://github.com/ciwanceylan/digraphwave/actions/workflows/tests-pip.yml)
[![tests-conda](https://github.com/ciwanceylan/digraphwave/actions/workflows/tests-conda.yml/badge.svg?branch=dev)](https://github.com/ciwanceylan/digraphwave/actions/workflows/tests-conda.yml)
[![codecov](https://codecov.io/gh/ciwanceylan/digraphwave/branch/dev/graph/badge.svg?token=EUOCUZ3B8U)](https://codecov.io/gh/ciwanceylan/digraphwave)

## Install instructions

Clone the repo.

Install the Digraphwave dependencies, using either conda or pip.

#### Using conda environments
Create a new environment using (either using cuda or cpu only)
```bash
conda env create --file <cpu/cuda>_environment.yml
```

#### Using pip
```bash
pip install -r requirements_<cpu/cuda>.txt
```

#### Install digraphwave
```bash
pip install ./digraphwave
```

### Note
If conda enviroments are used and you plan to install additional packages in the same environment, please the following to your `.condarc` file.
Otherwise package conflicts may occur when trying to install new packages. 
```
channels:
  - pyg
  - numba
  - pytorch
  - conda-forge
  - defaults
```

## Run tests
Install additional test dependencies

#### Using conda environments
```bash
conda install networkx=2.6 pytest
```


#### Using pip
```bash
pip install networkx==2.6 pytest
```

#### Run tests
```bash
pytest ./digraphwave/tests/
```