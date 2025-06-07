# Loan_default_prediction
 Loan Default Prediction A machine learning project to predict loan defaulters using models like Random Forest and Decision Tree. Includes EDA, feature scaling, model evaluation, and hyperparameter tuning with GridSearchCV and RandomizedSearchCV.  Tools: pandas, numpy, sklearn, seaborn, matplotlib Notebook: LoanDefault_project pjt.ipynb

Here's a polished and **attractive README** suitable for your GitHub repository based on the uploaded notebook `LoanDefault_project pjt.ipynb`. This includes sections for project overview, data, methodology, visualizations, results, and more.

---

# 📊 Loan Default Prediction Using Machine Learning

## 🧠 Project Overview

This project is focused on predicting the likelihood of a loan applicant defaulting on their loan using machine learning techniques. It leverages a structured dataset and explores data preprocessing, feature engineering, visualization, and model building to derive insights and build a predictive model.

The aim is to help financial institutions make better lending decisions and minimize credit risk.

---

## 📁 Dataset Description

The dataset includes detailed loan information such as:

* **Loan\_ID**
* **Gender, Marital Status, Dependents**
* **Education, Self-Employed**
* **Applicant and Coapplicant Income**
* **Loan Amount and Loan Term**
* **Credit History**
* **Property Area**
* **Loan\_Status** (Target variable: Y/N)

---

## 🔍 Problem Statement

> Predict whether a loan will be approved or not (Loan\_Status) based on applicant details and financial data.

This is a **binary classification** problem.

---

## 📊 Exploratory Data Analysis (EDA)

Key visual insights include:

* **Count plots** for categorical features (Gender, Married, Education, etc.)
* **Distribution plots** for numerical variables (Applicant Income, Loan Amount)
* **Box plots** highlighting income influence on loan approval
* **Heatmap** showing feature correlation
* **Bar graphs** demonstrating Credit History’s influence on Loan Status

📌 **Observation Highlights**:

* Applicants with a **credit history** are far more likely to get loan approval.
* **Self-employed** and **lower-income applicants** tend to have higher rejection rates.
* **Property area** and **education level** also have a visible impact on loan status.

---

## 🧼 Data Preprocessing

Performed the following steps:

* Handling **missing values**
* **Label encoding** of categorical variables
* **Outlier treatment**
* **Feature selection** based on EDA and correlation

---

## 🧪 Model Building

Models implemented:

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **K-Nearest Neighbors**
* **Support Vector Machine**

🔧 Hyperparameter tuning done using **GridSearchCV** and **Cross-validation**.

---

## 📈 Model Evaluation

Metrics used:

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**

Random Forest and SVM performed best, with **accuracy scores > 80%**.

---

## 📚 Tech Stack

* Python (Jupyter Notebook)
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🚀 How to Run

1. Clone this repo:

   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook LoanDefault_project\ pjt.ipynb
   ```

---

## 💡 Future Work

* Integration with real-time loan processing systems
* Use of advanced models (e.g., XGBoost, LightGBM)
* Deployment using Flask/Django and Streamlit dashboard

---

## 🙌 Acknowledgements

Thanks to the open-source community and the contributors of the dataset.

---

If you'd like me to extract plots and clean up code from the notebook for a cleaner repo structure, I can help with that too. Would you like a `requirements.txt`, `.gitignore`, or code refactor for modularity?
