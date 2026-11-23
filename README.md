# Walmart Store Sales Forecasting

A time-series forecasting project focused on predicting weekly sales across Walmart stores and departments using statistical baselines, feature engineering, and machine learning.

## Project Overview

This project explores weekly sales forecasting using the Walmart Recruiting - Store Sales Forecasting dataset.

The objective is to predict department-level sales across multiple Walmart stores while accounting for seasonal patterns, holidays, and historical sales behaviour.

The project emphasizes time-aware validation, feature engineering, and comparison between traditional forecasting approaches and machine learning models.

## Dataset

**Source:** [Walmart Recruiting - Store Sales Forecasting (Kaggle)](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

The dataset contains information from 45 Walmart stores.

| Dataset | Description |
|---|---|
| train.csv | Historical weekly sales data |
| features.csv | External features such as temperature, fuel price, CPI, unemployment, and markdowns |
| stores.csv | Store metadata and store types |
| test.csv | Records for which sales must be forecast |
| sampleSubmission.csv | Example submission format |

### Dataset Dimensions

| Dataset | Rows | Columns |
|---|---:|---:|
| Train | 421,570 | 5 |
| Features | 8,190 | 12 |
| Stores | 45 | 3 |
| Test | 115,064 | 4 |

The training data contains 3,331 store-department time series.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git, GitHub

## Project Workflow

### 1. Data Loading and Exploration

- Loaded and inspected the training, testing, store, and feature datasets.
- Examined missing values, data types, and data distributions.
- Investigated weekly sales patterns across stores and departments.
- Identified zero and negative weekly sales observations.

### 2. Data Preprocessing and Feature Engineering

- Combined sales records with store and external feature information.
- Extracted time-based information from weekly dates.
- Developed historical lag features to capture previous sales behaviour.
- Incorporated seasonal and holiday-related information.
- Prepared features for machine learning forecasting.

### 3. Time-Series Validation

A chronological validation strategy was used instead of a random train-test split to preserve the temporal ordering of observations.

The revised evaluation used:

- Training cutoff: October 28, 2011
- Evaluation period: November 4, 2011 – July 27, 2012
- Forecast horizon: 39 weeks

This approach helps evaluate model performance on future observations without introducing information from the evaluation period into training.

### 4. Model Development

Two forecasting approaches were compared:

**Seasonal Naive Forecasting**

Used historical seasonal sales behaviour as a benchmark.

**HistGradientBoosting Regressor**

Developed a machine learning forecasting model using engineered lag-based predictors and historical sales information.

### 5. Model Evaluation

Models were evaluated using Weighted Mean Absolute Error (WMAE).

Holiday observations received five times the weight of non-holiday observations.

\[
WMAE = \frac{\sum_{i=1}^{n} w_i |y_i-\hat{y}_i|}{\sum_{i=1}^{n} w_i}
\]

Where:

- \(y_i\) = Actual sales
- \(\hat{y}_i\) = Predicted sales
- \(w_i\) = Observation weight

## Results

### Model Comparison

| Model | WMAE |
|---|---:|
| Seasonal Naive | 2,043.61 |
| HistGradientBoosting (Lag Features) | 3,113.42 |

<ins>Lower WMAE indicates better forecasting performance.</ins>

### Key Findings

- The Seasonal Naive baseline outperformed the initial lag-based machine learning model.
- The machine learning model struggled to generalize across the evaluated store-department series.
- The comparison demonstrated the importance of establishing strong forecasting baselines before introducing complex models.
- Time-aware validation provided a more realistic assessment of forecasting performance.

## Key Learnings

- Handling multi-series time-series forecasting problems.
- Understanding the importance of chronological validation.
- Developing lag-based predictive features.
- Evaluating forecasting models using weighted error metrics.
- Comparing statistical baselines against machine learning approaches.
- Understanding why a more complex model does not necessarily produce better predictions.

## Future Improvements

- Improve lag and rolling-window feature engineering.
- Investigate store and department-specific forecasting strategies.
- Experiment with seasonal statistical models.
- Explore alternative machine learning algorithms.
- Improve holiday-related feature handling.
- Perform systematic hyperparameter optimization.

## Repository Structure

The repository includes the project's notebooks, Python source files, datasets, and supporting documentation.

- `notebooks/` – Exploratory analysis and modelling notebooks.
- `src/` – Python scripts for feature engineering and forecasting.
- `data/` – Dataset files.
- `README.md` – Project documentation.

## Author

**Mark Reji**

M.Sc. Statistics Student | Data Science & Analytics

- GitHub: [markreji2027](https://github.com/markreji2027)
- LinkedIn: [Mark Reji](https://linkedin.com/in/mark-reji-29488b246)

---

*This project was developed as an independent data science and forecasting project using the Walmart Recruiting Store Sales Forecasting dataset.*