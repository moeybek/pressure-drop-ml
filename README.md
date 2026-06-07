# Pressure Drop Prediction with Python and Machine Learning

This project predicts pressure drop in internal flow using engineering-based synthetic data and a machine learning regression model.

The goal is to connect mechanical engineering, fluid mechanics, Python automation, and machine learning in one reproducible workflow.

## Schematic

![Pressure drop schematic](assets/pressure_drop_schematic.png)

## Problem

Given flow and geometry parameters, predict pressure drop across a pipe/orifice-like restriction.

## Inputs

- Fluid density
- Dynamic viscosity
- Pipe diameter
- Flow velocity
- Orifice diameter ratio
- Reynolds number

## Output

- Pressure drop

## Workflow

1. Generate engineering-based synthetic data.
2. Train a regression model.
3. Compare ML predictions against formula-based values.
4. Visualize errors.
5. Interpret the results from an engineering perspective.

## Tools

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib