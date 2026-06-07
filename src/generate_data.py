from pathlib import Path

import numpy as np
import pandas as pd


def generate_pressure_drop_data(
    n_samples: int = 20000,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    density = rng.uniform(950, 1050, n_samples)          # kg/m³
    viscosity = rng.uniform(0.0007, 0.0013, n_samples)   # Pa·s
    pipe_diameter = rng.uniform(0.02, 0.20, n_samples)   # m
    velocity = rng.uniform(0.2, 5.0, n_samples)          # m/s
    beta = rng.uniform(0.3, 0.8, n_samples)              # d / D

    reynolds_number = density * velocity * pipe_diameter / viscosity

    # Physics-informed features
    dynamic_pressure = 0.5 * density * velocity**2
    loss_coefficient = (1 / beta**4) - 1

    # Clean pressure-drop target
    pressure_drop_clean = loss_coefficient * dynamic_pressure

    # Noisy pressure-drop target
    noise = rng.normal(
        loc=0.0,
        scale=0.07 * pressure_drop_clean,
        size=n_samples,
    )
    pressure_drop_noisy = pressure_drop_clean + noise

    return pd.DataFrame(
        {
            "density_kg_m3": density,
            "viscosity_pa_s": viscosity,
            "pipe_diameter_m": pipe_diameter,
            "velocity_m_s": velocity,
            "beta": beta,
            "reynolds_number": reynolds_number,
            "dynamic_pressure_pa": dynamic_pressure,
            "loss_coefficient": loss_coefficient,
            "pressure_drop_clean_pa": pressure_drop_clean,
            "noise_pa": noise,
            "pressure_drop_noisy_pa": pressure_drop_noisy,
        }
    )


def main() -> None:
    output_dir = Path("data/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_pressure_drop_data()
    output_path = output_dir / "pressure_drop_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()