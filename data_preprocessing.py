import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load and create Region column
def load_and_label_region(filepath):
    df = pd.read_csv(filepath, header=1)

    # Create Region column: 0=Bejaia, 1=Sidi-Bel Abbes
    n_rows = len(df)
    df["Region"] = [0 if i < n_rows // 2 else 1 for i in range(n_rows)]

    # Drop empty unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df

def preprocess_data(df):
    # Convert target column to numeric and handle NaNs
    df["FWI"] = pd.to_numeric(df["FWI"], errors="coerce")

    # Drop rows where target is NaN
    df = df.dropna(subset=["FWI"])

    X = df.drop(columns=["FWI"])
    y = df["FWI"].astype(float)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Separate numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Fit + transform
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def pipeline(filepath):
    df = load_and_label_region(filepath)
    return preprocess_data(df)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = pipeline("./data/Algerian_forest_fires_dataset.csv")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
