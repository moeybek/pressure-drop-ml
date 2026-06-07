from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_model(
    model,
    model_name: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    feature_set_name: str,
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

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "target": target_label,
        "feature_set": feature_set_name,
        "model": model_name,
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

    raw_features = [
        "density_kg_m3",
        "viscosity_pa_s",
        "pipe_diameter_m",
        "velocity_m_s",
        "beta",
        "reynolds_number",
    ]

    physics_informed_features = [
        "density_kg_m3",
        "viscosity_pa_s",
        "pipe_diameter_m",
        "velocity_m_s",
        "beta",
        "reynolds_number",
        "dynamic_pressure_pa",
        "loss_coefficient",
    ]

    feature_sets = [
        {
            "name": "Raw features",
            "columns": raw_features,
        },
        {
            "name": "Raw + physics-informed features",
            "columns": physics_informed_features,
        },
    ]

    targets = [
        {
            "column": "pressure_drop_clean_pa",
            "label": "Clean pressure drop",
        },
        {
            "column": "pressure_drop_noisy_pa",
            "label": "Noisy pressure drop",
        },
    ]

    models = [
        {
            "name": "Linear Regression",
            "model": LinearRegression(),
        },
        {
            "name": "Random Forest",
            "model": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        },
        {
            "name": "Gradient Boosting",
            "model": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        },
    ]

    results = []

    for target in targets:
        for feature_set in feature_sets:
            for model_config in models:
                result = evaluate_model(
                    model=model_config["model"],
                    model_name=model_config["name"],
                    df=df,
                    feature_columns=feature_set["columns"],
                    feature_set_name=feature_set["name"],
                    target_column=target["column"],
                    target_label=target["label"],
                )
                results.append(result)

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