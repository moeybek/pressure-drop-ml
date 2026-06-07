from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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

    target_column = "pressure_drop_pa"

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

    print("Model evaluation")
    print("----------------")
    print(f"MAE:  {mae:.2f} Pa")
    print(f"RMSE: {rmse:.2f} Pa")
    print(f"R²:   {r2:.4f}")


if __name__ == "__main__":
    main()