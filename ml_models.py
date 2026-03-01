import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_models(file_path):

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    X = df[["month", "year"]]
    y = df["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "model": name,
            "R2": round(r2_score(y_test, predictions), 3),
            "MAE": round(mean_absolute_error(y_test, predictions), 3)
        })

    # Create R2 graph
    model_names = [r["model"] for r in results]
    r2_scores = [r["R2"] for r in results]
    mae_scores = [r["MAE"] for r in results]

    plt.figure(figsize=(8,5))
    plt.bar(model_names, r2_scores)
    plt.title("Model Comparison - R2 Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/ml_r2.png")
    plt.close()

    # Create MAE graph
    plt.figure(figsize=(8,5))
    plt.bar(model_names, mae_scores)
    plt.title("Model Comparison - MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/ml_mae.png")
    plt.close()

    return results