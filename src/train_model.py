from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_label: str,
) -> dict[str, float | str]:
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "target": target_label,
        "mae_pa": mae,
        "rmse_pa": rmse,
        "r2": r2,
    }


def main() -> None:
    data_path = Path("data/generated/pressure_drop_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python src/generate_data.py` first."
        )

    df = pd.read_csv(data_path)

    feature_columns = [
        "density_kg_m3",
        "viscosity_pa_s",
        "pipe_diameter_m",
        "velocity_m_s",
        "beta",
        "reynolds_number",
    ]

    results = [
        train_and_evaluate(
            df=df,
            feature_columns=feature_columns,
            target_column="pressure_drop_clean_pa",
            target_label="Clean pressure drop",
        ),
        train_and_evaluate(
            df=df,
            feature_columns=feature_columns,
            target_column="pressure_drop_noisy_pa",
            target_label="Noisy pressure drop",
        ),
    ]

    results_df = pd.DataFrame(results)

    print("\nModel comparison")
    print("----------------")
    print(results_df.to_string(index=False))

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved model comparison to: {output_path}")


if __name__ == "__main__":
    main()