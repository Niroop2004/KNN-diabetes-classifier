# KNN-diabetes-classifier
This project implements a machine learning pipeline using the K-Nearest Neighbours (KNN) algorithm to classify whether a patient is diabetic or not, based on medical attributes. It includes data pre-processing, feature normalization, model training, evaluation for different values of k, and a user input interface for real-time prediction.

# Dataset
Source: Pima Indians Diabetes Database
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

# Feature Interpretations
## Pregnancies
🔍 **Number of times the patient has been pregnant**  
- **Typical Range:** 0 to ~17  
- **Insight:** Higher values may suggest increased risk, as gestational diabetes is a known factor.

## Glucose
🔍 **Plasma glucose concentration after 2 hours in an oral glucose tolerance test**  
- **Typical Range:**  
  - Normal: 70–99 mg/dL  
  - Prediabetes: 100–125 mg/dL  
  - Diabetes: 126+ mg/dL  
- **Insight:** High glucose is a strong indicator of diabetes.

## BloodPressure
🔍 **Diastolic blood pressure (mm Hg)**  
- **Normal Range:** ~60–80 mm Hg  
- **Insight:** High or very low diastolic pressure may contribute to health risks.

## SkinThickness
🔍 **Triceps skinfold thickness (in mm)**  
- **Typical Range:** 10–50 mm  
- **Insight:** Measures subcutaneous fat; correlates with BMI and obesity.

## Insulin
🔍 **2-Hour serum insulin (mu U/ml)**  
- **Typical Range:** 16–166 mu U/ml (can vary)  
- **Insight:** Elevated or very low insulin levels can be indicators of insulin resistance.

## BMI (Body Mass Index)
🔍 **Weight (kg) / Height (m)^2**  
- **Interpretation:**  
  - Underweight: < 18.5  
  - Normal: 18.5–24.9  
  - Overweight: 25–29.9  
  - Obese: 30+  
- **Insight:** Obesity is a major risk factor for type 2 diabetes.

## DiabetesPedigreeFunction (Pedigree)
🔍 **A function that scores likelihood of diabetes based on family history**  
- **Typical Range:** 0.0 – 2.5+  
- **Insight:** Higher values mean a stronger genetic influence.

## Age
🔍 **Age in years**  
- **Typical Range in Dataset:** 21 – ~80  
- **Insight:** Risk increases significantly with age, especially over 45.


# 🚀 Project Features
✅ Data cleaning (replacing invalid zeros with median values)
✅ Feature normalization using MinMaxScaler
✅ Training and evaluating KNN for different values of k (1–20)
✅ Accuracy visualization across k values
✅ Confusion matrix and classification report
✅ Manual input prediction with human-readable output (Diabetic or Not Diabetic)


# 🧪 Tech Stack

```
Python
Pandas, NumPy, Matplotlib, Seaborn
scikit-learn
```
## 🔍 How to Use
**Clone the repository:**
git clone https://github.com/your-username/knn-diabetes-predictor.git
cd knn-diabetes-predictor
**Run the notebook:**
Open KNN_Diabetes_Analysis.ipynb in Jupyter or Google Colab and run all cells.
Enter new patient data in the input cell to predict diabetes status.

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

# Results
Best accuracy achieved at k = X
Performance metrics: Precision, Recall, F1-Score, and Confusion Matrix provided
