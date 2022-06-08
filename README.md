# WOBBLE: Solid Dynamics Simulation Tool for Large Translation and Rotations

### Extending EPFL's [Akantu](https://akantu.ch/) package, WOBBLE *(Waves Of Beams and other Bodies due to Loadings and Excitations)* offers efficient and accurate simulation capabilities by utilising modal analysis and rigid body mechanics

## Authors

- Oisín Morrison (<oisin.morrison@epfl.ch>)
- Leonhard Driever (<leonhard.driever@epfl.ch>)

## Purpose

WOBBLE is designed to deal with situations where large translations and rotations may be at play. This makes the package suitable for the simulation of satellites in space, which is important for applications such as [ClearSpace](https://clearspace.today/).

## Key Features

- Efficient Numerical Solvers
- Accurate Results
- Ability to Model Both Clamped and Non-Clamped Bodies

## Codebase

The project is structured using the following directory structure:

```
.
├── wobble
└── examples
     ├── sample_data_files
     └── sample_notebooks
```

where
- `wobble` contains the code for the WOBBLE package
- `sample_data_files` contains the files (mesh files, eigenmode files, force files, mask files, geometry files, material files) for running the examples
- `sample_notebooks` contains a number of sample notebooks showing how to run WOBBLE to obtain results shown in our report

Within `wobble`, there exists 5 main files, containing 5 key classes for the WOBBLE package:
```
wobble
├── ma_fundamentals.py
├── pure.py
├── rb_fundamentals.py
├── rb_simple.py
└── rb_coupled.py
```
where
- `ma_fundamentals.py` contains the abstract class `MAFundamentals`, which implements the main methods used across all other classes
- `pure.py` contains the `PureMA` class, which implements the solver using pure modal analysis (no rigid body motion). This is suitable for modelling scenarios where bodies are clamped
- `rb_fundamentals.py` contains the abstract class `RBFundamentals`, which implements the main methods used across all the rigid body mechanics solvers
- `rb_simple.py` contains the `SimpleRB` class, which implements the SimpleRB algorithm discussed in the report (implements rigid body mechanics and modal analysis in a decoupled fashion)
- `rb_coupled.py` contains the `CoupledRB` class, which implements the SimpleRB algorithm discussed in the report (implements rigid body mechanics and modal analysis in a coupled fashion)

## Usage

Examples of notebooks are provided in the `sample_notebooks` folder.

The authors recommend the following:
- Use `PureMA` if rigid body mechanics are not important for the application (e.g. clamped beam)
- Use `SimpleRB` if the body is not expected to be subject to a large amount of fictitious forces due to its rotation
- Use `CoupledRB` if the beam is expected to be subject to a large amount of fictitious forces due to its rotation

## Requirements

WOBBLE was tested with `Python 3.8.10` and uses the following packages:
```
akantu==4.0.1
numpy==1.20.3
matplotlib==3.4.3
scipy==1.8.0
```

## Other

*For further details, please see our report or contact the authors*