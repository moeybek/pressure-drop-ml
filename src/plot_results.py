from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main() -> None:
    data_path = Path("data/generated/pressure_drop_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python src/generate_data.py` first."
        )

    figures_dir = Path("figures/generated")
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    residuals = y_test - y_pred

    # Actual vs predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, s=10, alpha=0.5)
    plt.xlabel("Actual pressure drop [Pa]")
    plt.ylabel("Predicted pressure drop [Pa]")
    plt.title("Actual vs Predicted Pressure Drop")

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png", dpi=150)
    plt.show()

    # Residual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, s=10, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted pressure drop [Pa]")
    plt.ylabel("Residual [Pa]")
    plt.title("Residuals vs Predicted Pressure Drop")
    plt.tight_layout()
    plt.savefig(figures_dir / "residuals_vs_predicted.png", dpi=150)
    plt.show()

    # Feature importance
    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    feature_importance.plot(kind="bar")
    plt.ylabel("Feature importance")
    plt.title("Random Forest Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.show()

    print("Saved plots to figures/generated/")
    print(feature_importance)


if __name__ == "__main__":
    main()