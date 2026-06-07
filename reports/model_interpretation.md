# Model Interpretation

## Objective

The objective of this project is to predict pressure drop across a pipe/orifice-like restriction using flow and geometry parameters.

The synthetic dataset is generated from a simplified engineering relation:

\[
\Delta p = K \cdot \frac{1}{2} \rho v^2
\]

with:

\[
K = \left(\frac{1}{\beta^4} - 1\right)
\]

where:

- \(\Delta p\) is the pressure drop
- \(\rho\) is the fluid density
- \(v\) is the flow velocity
- \(\beta = d/D\) is the orifice-to-pipe diameter ratio
- \(K\) is the loss coefficient

## Model

A Random Forest regression model was trained to predict pressure drop from:

- density
- viscosity
- pipe diameter
- velocity
- beta
- Reynolds number

## Evaluation

The first model achieved:

- MAE: 3670.92 Pa
- RMSE: 10267.12 Pa
- R²: 0.9971

The high R² score is expected because the dataset was generated from a deterministic engineering equation.

## Engineering Interpretation

Pressure drop increases strongly with velocity because dynamic pressure scales with:

\[
v^2
\]

The beta value has a strong nonlinear influence. Smaller beta values represent stronger restrictions and therefore lead to much higher pressure losses.

Density also contributes directly to pressure drop through the dynamic pressure term.

Viscosity and pipe diameter influence the Reynolds number, but in this simplified formulation they do not directly affect the pressure-drop equation.

## Limitations

This project uses simplified synthetic data. It does not yet include:

- CFD simulation results
- turbulence model effects
- wall roughness
- compressibility
- real measurement noise
- detailed discharge coefficient corrections
- geometry-dependent local flow effects

## Next Step

The next improvement is to compare this simplified ML workflow against CFD-generated data from OpenFOAM or another solver.