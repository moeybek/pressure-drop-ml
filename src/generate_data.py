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
    loss_coefficient = (1 / beta**4) - 1
    #pressure_drop = loss_coefficient * 0.5 * density * velocity**2
    pressure_drop_clean = loss_coefficient * 0.5 * density * velocity**2
    noise = rng.normal(0,
                       scale=0.07 * pressure_drop_clean,  # 7% noise
                       size=n_samples)
    pressure_drop = pressure_drop_clean + noise

    return pd.DataFrame(
        {
            "density_kg_m3": density,
            "viscosity_pa_s": viscosity,
            "pipe_diameter_m": pipe_diameter,
            "velocity_m_s": velocity,
            "beta": beta,
            "reynolds_number": reynolds_number,
            "loss_coefficient": loss_coefficient,
            "pressure_drop_clean_pa": pressure_drop_clean,
            "pressure_drop_noisy_pa": pressure_drop,
            "noise_pa": noise,
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