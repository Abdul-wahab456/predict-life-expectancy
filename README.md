# Life Expectancy Prediction using WHO Dataset

## Overview
This project is based on a dataset collected by the **World Health Organization (WHO)**. The data comes from a global survey conducted by WHO, with the support of local sources and various NGOs, to analyze and predict the life expectancy of people in different regions around the world. Using this dataset, the project follows the Data Science pipeline to preprocess the data, perform exploratory data analysis (EDA), and build a predictive machine learning model.

## Objective
The goal is to build a machine learning model that predicts the average life expectancy of people in a specific region based on the provided features.

## Dataset
- The dataset contains features related to health, socio-economic factors, and regional data that influence life expectancy.
- This is a sample of the original data collected by WHO, tailored for this project.

## Features
- **Data Preprocessing**: 
  - Handles missing values using KNN Imputation.
  - Scales and transforms features to prepare the data for modeling.
  - Reduces dimensionality with PCA for better performance.
- **Exploratory Data Analysis (EDA)**:
  - Visualizes and summarizes key attributes, correlations, and distributions.
- **Machine Learning**:
  - Utilizes `RandomForestRegressor` to predict life expectancy.
  - Optimizes the model using hyperparameter tuning with `GridSearchCV`.
  - Evaluates model performance using R² and other metrics.

## Workflow

### 1. Data Loading and Cleaning
- Loaded the dataset and addressed missing values using KNN Imputation.
- Scaled numerical features to prepare the data for machine learning.
- Applied dimensionality reduction using PCA to eliminate redundant information.

### 2. Exploratory Data Analysis (EDA)
- Explored the dataset's features to understand their influence on life expectancy.
- Visualized data distributions and identified correlations using plots and heatmaps.

### 3. Model Development
- Trained a `RandomForestRegressor` on the preprocessed dataset.
- Performed hyperparameter tuning with `GridSearchCV` to optimize model performance.

### 4. Evaluation
- Evaluated the model's performance using metrics such as R² score.
- Validated accuracy and interpreted results based on the test dataset.

### 5. Results
- Successfully predicted average life expectancy using socio-economic and health-related factors.
- Documented key visualizations and performance metrics throughout the notebook.

## Libraries Used
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (Pipelines, KNN Imputer, Random Forest, PCA, GridSearchCV)
- **Utilities**: `warnings`
