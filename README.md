# Pressure Drop Prediction with Python and Machine Learning

This project predicts pressure drop in internal flow across a pipe/orifice-like restriction using engineering-based synthetic data and machine learning regression models.

The goal is to connect mechanical engineering, fluid mechanics, Python automation, data generation, model training, model comparison, visualization, and engineering interpretation in one reproducible workflow.

The synthetic dataset represents a water-like incompressible fluid. The sampled density range is 950–1050 kg/m³ and the dynamic viscosity range is 0.0007–0.0013 Pa·s.

Adding physics-informed features improved model performance, especially for Linear Regression. The R² score of Linear Regression increased from about 0.19 to about 0.67 for both clean and noisy targets. Random Forest remained the best-performing model overall, with the physics-informed feature set achieving the lowest errors and highest R² values. This shows that physics-based feature engineering can make the learning problem easier, especially for simpler models.

## Schematic

![Pressure drop schematic](assets/pressure_drop_schematic.png)

The modeled system is a pipe with a local restriction. The flow enters with velocity `v`, passes through an orifice with diameter `d`, and experiences a pressure drop `Δp`.

```text
Upstream flow                      Restriction                    Downstream flow

p1, v  ────────────────>     | smaller opening |     ───────────────>  p2

Pipe diameter:      D
Orifice diameter:   d
Diameter ratio:     beta = d / D

Pressure drop:      Δp = p1 - p2
```

## Physical Background

The simplified pressure-drop relation used in this project is:

```text
Δp = K · 0.5 · ρ · v²
```

where:

- `Δp` = pressure drop [Pa]
- `K` = loss coefficient [-]
- `ρ` = fluid density [kg/m³]
- `v` = flow velocity [m/s]

The loss coefficient is approximated as:

```text
K = 1 / β⁴ - 1
```

The diameter ratio is:

```text
β = d / D
```

where:

- `d` = orifice diameter [m]
- `D` = pipe diameter [m]
- `β` = orifice-to-pipe diameter ratio [-]

The Reynolds number is calculated as:

```text
Re = ρ · v · D / μ
```

where:

- `Re` = Reynolds number [-]
- `μ` = dynamic viscosity [Pa·s]

## Problem

Given flow and geometry parameters, the objective is to predict the pressure drop across a pipe/orifice-like restriction.

The project includes two target variables:

- Clean pressure drop calculated directly from the engineering equation
- Noisy pressure drop created by adding random noise to simulate measurement or simulation uncertainty

## Inputs

The machine learning models use the following input features:

- Fluid density
- Dynamic viscosity
- Pipe diameter
- Flow velocity
- Orifice diameter ratio
- Reynolds number

## Outputs

The generated dataset contains:

- Clean pressure drop [Pa]
- Added noise [Pa]
- Noisy pressure drop [Pa]

## Dataset

The dataset is synthetically generated using engineering equations.

Each row represents one operating condition with randomly sampled fluid and geometry parameters.

The clean pressure-drop target is calculated from the simplified physical equation:

```text
Δp_clean = K · 0.5 · ρ · v²
```

A noisy pressure-drop target is then created by adding random noise:

```text
Δp_noisy = Δp_clean + noise
```

The noise is added to make the dataset slightly more realistic. Real engineering data often contains uncertainty from measurement errors, numerical errors, operating fluctuations, or sensor noise.

This makes the dataset useful for building a first reproducible workflow before moving to CFD-generated or experimental data.

## Machine Learning Models

Three regression models are trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Each model is trained twice:

1. Once using the clean pressure-drop target
2. Once using the noisy pressure-drop target

This allows comparison between model behavior on ideal data and more realistic noisy data.

## Model Comparison Results

| Target | Model | MAE [Pa] | RMSE [Pa] | R² |
|---|---|---:|---:|---:|
| Clean pressure drop | Linear Regression | 97,708.95 | 158,881.64 | 0.1908 |
| Clean pressure drop | Random Forest | 2,094.03 | 5,645.07 | 0.9990 |
| Clean pressure drop | Gradient Boosting | 4,738.06 | 7,868.93 | 0.9980 |
| Noisy pressure drop | Linear Regression | 97,523.09 | 158,310.18 | 0.1903 |
| Noisy pressure drop | Random Forest | 6,061.49 | 16,002.85 | 0.9917 |
| Noisy pressure drop | Gradient Boosting | 7,814.26 | 16,452.57 | 0.9913 |

## Metric Interpretation

### MAE

MAE stands for Mean Absolute Error.

It describes the average absolute prediction error.

Example:

```text
MAE = 6061.49 Pa
```

This means that, on average, the model prediction differs from the target pressure-drop value by about 6061 Pa.

### RMSE

RMSE stands for Root Mean Squared Error.

It also measures prediction error, but it penalizes large errors more strongly than MAE.

A large difference between MAE and RMSE means that some individual predictions have much larger errors.

### R²

R² measures how much of the target variation is explained by the model.

Example:

```text
R² = 0.9917
```

This means that the model explains about 99.17% of the pressure-drop variation.

## Engineering Interpretation

The results show that Linear Regression performs poorly, while Random Forest and Gradient Boosting perform very well.

This is physically reasonable.

Linear Regression tries to fit a straight-line relationship:

```text
pressure_drop = a·density + b·viscosity + c·diameter + d·velocity + e·beta + ...
```

But the real pressure-drop relation is nonlinear:

```text
Δp = K · 0.5 · ρ · v²
K = 1 / β⁴ - 1
```

The pressure drop depends strongly on:

```text
v²
```

and on:

```text
1 / β⁴
```

A simple linear model cannot capture this nonlinear behavior well.

Tree-based models such as Random Forest and Gradient Boosting can capture nonlinear relationships more effectively. They can learn behavior such as:

```text
if beta is small and velocity is high, pressure drop becomes very high
```

This matches the physical behavior of the simplified pressure-drop equation.

## Feature Importance Interpretation

The most important features are:

1. Beta
2. Velocity

This is physically consistent with the pressure-drop equation.

Beta controls the loss coefficient:

```text
K = 1 / β⁴ - 1
```

Velocity controls the dynamic pressure term:

```text
0.5 · ρ · v²
```

A smaller beta value means a smaller orifice opening relative to the pipe diameter. This creates a stronger restriction and therefore a much higher pressure drop.

Velocity is also important because pressure drop increases approximately with the square of velocity.

Density contributes directly through the dynamic pressure term, but its influence is smaller because only a narrow water-like density range is sampled.

Viscosity and pipe diameter mainly affect the Reynolds number in this simplified setup, but they do not directly appear in the pressure-drop equation used to generate the target value. Therefore, their model importance is low.

## Clean vs Noisy Target Interpretation

The clean target is easier for the models to learn because it comes directly from a deterministic engineering equation.

The noisy target is harder because random noise is added to the clean pressure-drop values.

As expected:

- The noisy target has higher MAE.
- The noisy target has higher RMSE.
- The noisy target has lower R².

This behavior is realistic because real engineering data is rarely perfectly clean.

## Workflow

1. Generate engineering-based synthetic data.
2. Add noise to create a more realistic target.
3. Explore and visualize the dataset.
4. Train multiple regression models.
5. Compare model performance on clean and noisy targets.
6. Generate evaluation plots.
7. Interpret the results from an engineering perspective.

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python src/generate_data.py
```

Train and compare the models:

```bash
python src/train_model.py
```

Generate evaluation plots:

```bash
python src/plot_results.py
```

## Project Structure

```text
pressure-drop-ml/
│
├── assets/
│   └── pressure_drop_schematic.png
│
├── data/
│   └── generated/
│
├── figures/
│   └── generated/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   └── model_interpretation.md
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   └── train_models_plot_results.py
│
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Generated Outputs

The project can generate:

- Synthetic pressure-drop dataset
- Model comparison table
- Actual vs predicted pressure-drop plots
- Residual plots
- Feature-importance plots

## Generated Plots

The project generates plots such as:

- Actual vs predicted pressure drop for clean and noisy targets
- Residuals vs predicted pressure drop for clean and noisy targets
- Feature importance comparison between clean and noisy targets
- Velocity vs pressure drop

These plots help evaluate both the machine learning performance and the physical behavior of the generated dataset.

## Limitations

This project uses simplified synthetic data. It does not yet include:

- CFD simulation results
- experimental measurements
- turbulence model effects
- wall roughness
- compressibility
- discharge coefficient corrections
- complex geometry effects
- realistic sensor behavior
- uncertainty quantification
- model deployment

The current model is therefore not intended as a production-ready pressure-drop predictor. It is a reproducible first step for connecting fluid mechanics, Python, and machine learning.
