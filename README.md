# KNN-diabetes-classifier
This project implements a machine learning pipeline using the K-Nearest Neighbours (KNN) algorithm to classify whether a patient is diabetic or not, based on medical attributes. It includes data pre-processing, feature normalization, model training, evaluation for different values of k, and a user input interface for real-time prediction.

📊 Dataset
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

🚀 Project Features
✅ Data cleaning (replacing invalid zeros with median values)

✅ Feature normalization using MinMaxScaler

✅ Training and evaluating KNN for different values of k (1–20)

✅ Accuracy visualization across k values

✅ Confusion matrix and classification report

✅ Manual input prediction with human-readable output (Diabetic or Not Diabetic)

🧪 Tech Stack
Python

Pandas, NumPy, Matplotlib, Seaborn

scikit-learn

🔍 How to Use
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/knn-diabetes-predictor.git
cd knn-diabetes-predictor
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:
Open KNN_Diabetes_Analysis.ipynb in Jupyter or Google Colab and run all cells.

Enter new patient data in the input cell to predict diabetes status.

📈 Sample Output
yaml
Copy
Edit
Enter the following values:
Pregnancies: 2
Glucose: 130
BloodPressure: 70
SkinThickness: 28
Insulin: 88
BMI: 32.5
Pedigree: 0.45
Age: 36

✅ Prediction Result: Not Diabetic
📌 Results
Best accuracy achieved at k = X

Performance metrics: Precision, Recall, F1-Score, and Confusion Matrix provided
