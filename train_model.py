import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from data_preprocessing import pipeline

data = './data/Algerian_forest_fires_dataset.csv'

# Dictionary to store R² scores
r2_scores = {}

# Load preprocessed data
X_train_scaled, X_test_scaled, y_train, y_test = pipeline(data)

def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Actual vs Predicted")
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m*y_test + b, color='red', label="Best Fit Line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.show()

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def train_ridge(X_train, X_test, y_train, y_test):
    print("=== Ridge Regression ===")
    ridge = Ridge(alpha=2.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)
    r2_scores["Ridge"] = r2
    plot_actual_vs_predicted(y_test, y_pred, title="Ridge Regression: Actual vs Predicted")
    save_model(ridge, "ridge_model.pkl")

    print("\n=== RidgeCV Regression ===")
    ridgecv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    ridgecv.fit(X_train, y_train)
    y_pred_cv = ridgecv.predict(X_test)
    mae_cv = mean_absolute_error(y_test, y_pred_cv)
    r2_cv = r2_score(y_test, y_pred_cv)
    print("Mean Absolute Error (CV):", mae_cv)
    print("R2 Score (CV):", r2_cv)
    r2_scores["RidgeCV"] = r2_cv
    plot_actual_vs_predicted(y_test, y_pred_cv, title="RidgeCV Regression: Actual vs Predicted")
    save_model(ridgecv, "ridgecv_model.pkl")

def train_lasso(X_train, X_test, y_train, y_test):
    print("=== Lasso Regression ===")
    lasso = Lasso(alpha=0.05, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)
    r2_scores["Lasso"] = r2
    plot_actual_vs_predicted(y_test, y_pred, title="Lasso Regression: Actual vs Predicted")
    save_model(lasso, "lasso_model.pkl")
    
    print("\n=== LassoCV Regression ===")
    lassocv = LassoCV(alphas=[0.01, 0.05, 0.1, 1.0], cv=5, random_state=42)
    lassocv.fit(X_train, y_train)
    y_pred_cv = lassocv.predict(X_test)
    mae_cv = mean_absolute_error(y_test, y_pred_cv)
    r2_cv = r2_score(y_test, y_pred_cv)
    print("Mean Absolute Error (CV):", mae_cv)
    print("R2 Score (CV):", r2_cv)
    r2_scores["LassoCV"] = r2_cv
    plot_actual_vs_predicted(y_test, y_pred_cv, title="LassoCV Regression: Actual vs Predicted")
    save_model(lassocv, "lassocv_model.pkl")

def plot_r2_comparison(scores_dict):
    plt.figure(figsize=(8,6))
    models = list(scores_dict.keys())
    scores = list(scores_dict.values())
    plt.bar(models, scores, color=['blue','orange','green','red'], alpha=0.7)
    plt.ylabel("R² Score")
    plt.title("Model Comparison (R² Scores)")
    plt.ylim(0, 1)  # since R² usually lies between 0 and 1
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    plt.show()

def run_all_models():
    train_ridge(X_train_scaled, X_test_scaled, y_train, y_test)
    train_lasso(X_train_scaled, X_test_scaled, y_train, y_test)
    print("\n=== R² Score Comparison ===")
    print(r2_scores)
    plot_r2_comparison(r2_scores)

if __name__ == "__main__":
    run_all_models()
