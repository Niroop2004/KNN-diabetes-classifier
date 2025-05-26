# KNN-diabetes-classifier
This project uses the K-Nearest Neighbors (KNN) algorithm for two classification tasks:
- Diabetes Prediction using the Pima Indians dataset
- Wine Quality Classification using the Wine Quality dataset.
It includes data preprocessing, normalization, model evaluation across various k-values, and a manual input interface for real-time predictions. Implemented in Python with scikit-learn and visualized using Matplotlib and Seaborn.

# Problem Statement 
## Using sample dataset, apply the k-nearest neighbor classifier and analyse its performance with different values of k.
- **Deliverable:** Jupyter notebook/Excel file with all steps and brief documentation. 

# Dataset
Here we have used the same model on 2 different datasets 
- Pima Indians Diabetes Dataset
- Wine Quality Dataset( Introductory Paper : Modeling wine preferences by data mining from physicochemical properties )
  
1. Source: Pima Indians Diabetes Database
Description: Contains diagnostic measurements from female patients of Pima Indian heritage, including:
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

   Target: Outcome (0 = Not Diabetic, 1 = Diabetic)

2. Source : UC Irvine Machine Learning Repository
The dataset used is `winequality-red.csv`.
using this dataset model classifies red wine samples into **Low**, **Medium**, and **High** quality categories using the **K-Nearest Neighbors (K-NN)** algorithm

# Diabetes Dataset Feature Interpretations
## Pregnancies
**Number of times the patient has been pregnant**  
- **Typical Range:** 0 to ~17  
- **Insight:** Higher values may suggest increased risk, as gestational diabetes is a known factor.

## Glucose
**Plasma glucose concentration after 2 hours in an oral glucose tolerance test**  
- **Typical Range:**  
  - Normal: 70–99 mg/dL  
  - Prediabetes: 100–125 mg/dL  
  - Diabetes: 126+ mg/dL  
- **Insight:** High glucose is a strong indicator of diabetes.

## BloodPressure
**Diastolic blood pressure (mm Hg)**  
- **Normal Range:** ~60–80 mm Hg  
- **Insight:** High or very low diastolic pressure may contribute to health risks.

## SkinThickness
**Triceps skinfold thickness (in mm)**  
- **Typical Range:** 10–50 mm  
- **Insight:** Measures subcutaneous fat; correlates with BMI and obesity.

## Insulin
**2-Hour serum insulin (mu U/ml)**  
- **Typical Range:** 16–166 mu U/ml (can vary)  
- **Insight:** Elevated or very low insulin levels can be indicators of insulin resistance.

## BMI (Body Mass Index)
**Weight (kg) / Height (m)^2**  
- **Interpretation:**  
  - Underweight: < 18.5  
  - Normal: 18.5–24.9  
  - Overweight: 25–29.9  
  - Obese: 30+  
- **Insight:** Obesity is a major risk factor for type 2 diabetes.

## DiabetesPedigreeFunction (Pedigree)
**A function that scores likelihood of diabetes based on family history**  
- **Typical Range:** 0.0 – 2.5+  
- **Insight:** Higher values mean a stronger genetic influence.

## Age
**Age in years**  
- **Typical Range in Dataset:** 21 – ~80  
- **Insight:** Risk increases significantly with age, especially over 45.


# Project Features
Data cleaning (replacing invalid zeros with median values)

Feature normalization using MinMaxScaler

Training and evaluating KNN for different values of k (1–20)

Accuracy visualization across k values

Confusion matrix and classification report

Manual input prediction with human-readable output (Diabetic or Not Diabetic)


# Tech Stack

```
Python
Pandas, NumPy, Matplotlib, Seaborn
scikit-learn
```
## How to Use
**Clone the repository:**
git clone https://github.com/your-username/knn-diabetes-predictor.git
cd knn-diabetes-predictor

**Run the notebook:**
Open KNN_Diabetes_Analysis.ipynb in Jupyter or Google Colab and run all cells.
Enter new patient data in the input cell to predict diabetes status.

# Performance 
 ## 1. Confusion Matrix
 [[89 10]
 
 [21 34]]
 
 - Confusion matrix is for test-dataset
 
**What this means:**

                     Predicted: 0         Predicted: 1
                 
**Actual: 0** 	     89 (True Negatives)	  10 (False Positives)

**Actual: 1**        21 (False Negatives)	34 (True Positives)

**True Negative (TN)** = 89 → Correctly predicted class 0

**False Positive (FP)** = 10 → Incorrectly predicted class 1 when it was 0

**False Negative (FN)** = 21 → Incorrectly predicted class 0 when it was 1


# Overall Metrics
**Accuracy** = 0.80 → 80% of total predictions were correct.

**Macro avg:**
Averages precision, recall, and F1 equally across both classes, regardless of how many samples each has.
Shows fairness across classes.

**Weighted avg:**
Averages precision, recall, and F1 weighted by support (i.e., more weight to class 0 due to more samples).
More reflective of overall model performance.

# Sample Output
**Enter the following values:**
Pregnancies: 2
Glucose: 130
BloodPressure: 70
SkinThickness: 28
Insulin: 88
BMI: 32.5
Pedigree: 0.45
Age: 36

**Prediction Result:** Not Diabetic

# 2 Wine Quality Dataset

## Dataset Overview

- **Dataset:** `winequality-red.csv`
- **Samples:** 1,599
- **Features:** 11 physicochemical inputs
- **Target Variable:** Wine quality (original score from 3 to 8)


## Features Description

The dataset includes the following input features:

| Feature Name             | Description                            |
|--------------------------|----------------------------------------|
| Fixed Acidity            | Tartaric acid content                  |
| Volatile Acidity         | Acetic acid content                    |
| Citric Acid              | Citric acid content                    |
| Residual Sugar           | Remaining sugar after fermentation     |
| Chlorides                | Salt content                           |
| Free Sulfur Dioxide      | Free SO₂ in wine                       |
| Total Sulfur Dioxide     | Total SO₂ in wine                      |
| Density                  | Wine density                           |
| pH                       | Acidity level                          |
| Sulphates                | Sulfate concentration                  |
| Alcohol                  | Alcohol content (%)                    |


## Target Variable Transformation

The `quality` score is transformed into 3 categories(Binning):

- **Low (0):** Scores 3 to 5  
- **Medium (1):** Score 6  
- **High (2):** Scores 7 to 8  

This converts the problem into a **multi-class classification** task.


## Preprocessing Steps

- **Feature Scaling**:
  - Applied both `StandardScaler` and `MinMaxScaler`
- **Train-Test Split**:
  - 80% training and 20% testing
  - Stratified to maintain class distribution


## Model: K-Nearest Neighbors

- **Hyperparameter Tuning**:
  - Explored `k` values from **4 to 32**
- **Evaluation Metric**:
  - Accuracy on the test set for each value of `k`


### Best Performing `k`
- Best k: 8 with Accuracy: 0.6625
  

### Confusion Matrix

| Actual → / Predicted ↓ | Low | Medium | High |
|------------------------|-----|--------|------|
| **Low**                | 111 |   30   |  0   |
| **Medium**             | 54  |   69   |  9   |
| **High**               | 4   |   24   |  19  |

- Confusion matrix is for test-dataset


### Classification Report
|         | precision  |  recall | f1-score  | support  |
|---------|------------|---------|-----------|----------|
|     Low |     0.66   |   0.79  |   0.72    |  141     |
|  Medium |      0.56  |   0.52  |   0.54    |  132     |
|    High |      0.68  |   0.40  |   0.51    |   47     |



## Manual Prediction Example

### Input: 7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0
- Predicted Wine Quality: Medium


