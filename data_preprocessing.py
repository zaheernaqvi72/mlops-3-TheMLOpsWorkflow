import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load and create Region column
def load_and_label_region(filepath):
    df = pd.read_csv(filepath, header=1)
    
    # Create Region column: 0=Bejaia, 1=Sidi-Bel Abbes
    df.loc[:122, "Region"] = 0
    df.loc[122:, "Region"] = 1
    df[['Region']] = df[['Region']].astype(int)
    
    # Drop null values
    df = df.dropna().reset_index(drop=True)
    
    # Drop 122nd row
    df = df.drop(122).reset_index(drop=True)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    return df

# Convert data types
def convert_types(df):
    numeric_cols = ['month', 'day', 'year', 'Temperature', 'RH', 'Ws', 
                    'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Region']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert object columns except 'Classes' to float
    object_cols = [col for col in df.columns if df[col].dtype == 'O' and col != 'Classes']
    for col in object_cols:
        df[col] = df[col].astype(float)
    
    return df

# Create preprocessing pipeline
def create_preprocessing_pipeline(df):
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove('Region')  # Region can be treated separately if desired
    numeric_cols.remove('FWI')  # Example of target-related feature (optional)
    
    # Categorical columns (objects)
    categorical_cols = [col for col in df.columns if df[col].dtype == 'O' and col != 'Classes']
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

# Prepare features and target
def prepare_data(df):
    X = df.drop('Classes', axis=1)
    y = df['Classes']
    return X, y

# Putting it all together
def main(filepath):
    df = load_and_label_region(filepath)
    df = convert_types(df)
    X, y = prepare_data(df)
    
    pipeline = create_preprocessing_pipeline(df)
    X_processed = pipeline.fit_transform(X)
    
    print(f"Processed feature shape: {X_processed.shape}")
    return X_processed, y, pipeline
