# Algerian Forest Fires ML Project 🌲🔥

## 📌 Overview

This project demonstrates a complete **MLOps workflow** for predicting **burned area (ha)** of Algerian forest fires using machine learning regression models.
It covers:

* Data preprocessing (cleaning, encoding, scaling)
* Model training (Ridge, RidgeCV, Lasso, LassoCV)
* Model evaluation (MAE, R² Score)
* Model persistence (`pickle`)
* Release management using Git

---

## 📂 Dataset

We use the **Algerian Forest Fires Dataset**, which contains meteorological and fire data collected from:

* **Bejaia Region** (northeastern Algeria)
* **Sidi-Bel-Abbes Region** (northwestern Algeria)

👉 Stored in: `./data/Algerian_forest_fires_dataset.csv`

---

## ⚙️ Workflow

1. **Data Ingestion**: Load the dataset from CSV.
2. **Data Preprocessing**: Clean and prepare the data for modeling.
3. **Feature Engineering**: Create new features from existing ones.
4. **Model Training**: Train multiple regression models.
5. **Model Evaluation**: Assess model performance using metrics.
6. **Model Persistence**: Save the trained models for later use.
7. **Release Management**: Use Git for version control.

---

## 🚀 Training

Run the training script:

```bash
python train_model.py
```

It will:

* Train **Ridge**, **RidgeCV**, **Lasso**, **LassoCV**
* Print **MAE & R² Scores**
* Save models (`.pkl` files)
* Generate comparison plots

---

## 📊 Results

### R² Score Comparison

#### Before Hyperparameter Update

```
{'Ridge': 0.7466, 'RidgeCV': 0.7521, 'Lasso': 0.4528, 'LassoCV': 0.7258}
```

#### After Hyperparameter Update

```
{'Ridge': 0.7406, 'RidgeCV': 0.7521, 'Lasso': 0.6753, 'LassoCV': 0.7259}
```

### Comparison Plot

The script generates a **bar chart** comparing model R² scores.

---

## 📦 Model Files

Saved under the project root:

* `ridge_model.pkl`
* `ridgecv_model.pkl`
* `lasso_model.pkl`
* `lassocv_model.pkl`

---

## 🔖 Release Management

We follow **semantic versioning** for model releases:

```bash
# Tag the release
git tag -a v1.0.0 -m "Release v1.0.0"

# Push the tag
git push origin v1.0.0

# Create deployment branch
git checkout -b deployment/v1.0.0
```

---

## ✅ Future Work

* Try **ElasticNet Regression**
* Add **MLflow for experiment tracking**
* Deploy model with **FastAPI / Flask**
* Create a **Docker container** for the application
* Implement **CI/CD pipeline** for automated testing and deployment

