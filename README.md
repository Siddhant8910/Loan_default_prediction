# Loan_default_prediction
 Loan Default Prediction A machine learning project to predict loan defaulters using models like Random Forest and Decision Tree. Includes EDA, feature scaling, model evaluation, and hyperparameter tuning with GridSearchCV and RandomizedSearchCV.  Tools: pandas, numpy, sklearn, seaborn, matplotlib Notebook: LoanDefault_project pjt.ipynb


 


## 🏦 Loan Default Risk Prediction: Home Equity Credit Scoring

### 📘 Context

Retail banks derive a substantial portion of their revenue from home loans, especially those offered to stable or high-income customers. However, **loan defaults (non-performing assets)** significantly impact profitability. Therefore, it is critical for banks to assess creditworthiness accurately before approving loans.

Traditionally, this evaluation is conducted manually, relying on human judgment to analyze customer profiles. While effective to an extent, this manual approach is time-consuming and prone to **errors and biases**. With the evolution of **data science and machine learning**, there is now an opportunity to **automate and improve** this process—making it **faster, fairer, and more consistent**.

---

### 🎯 Problem Statement

The Consumer Credit Department of a bank wants to streamline its decision-making process for home equity line approvals. In compliance with the **Equal Credit Opportunity Act (ECOA)**, the goal is to build a **statistically sound and interpretable credit scoring model**.

This model should:

* Predict whether an applicant is likely to **default**.
* Provide **clear justifications** for any rejection (interpretability is key).
* Help the bank make informed, bias-free credit decisions.

---

### 🧠 Objective

Develop a **classification model** to:

* Accurately identify applicants who are likely to **default on their loans**.
* Recommend the most influential features that the bank should prioritize in the approval process.

---

### 📊 Dataset Description

The dataset used is the **HMEQ (Home Equity) dataset** from Kaggle. It includes information on recent applicants and their loan performance.

Key features:

| Variable    | Description                                             |
| ----------- | ------------------------------------------------------- |
| **BAD**     | Target: 1 = defaulted, 0 = repaid                       |
| **LOAN**    | Approved loan amount                                    |
| **MORTDUE** | Outstanding amount on existing mortgage                 |
| **VALUE**   | Current value of the property                           |
| **REASON**  | Reason for loan (Home Improvement / Debt Consolidation) |
| **JOB**     | Applicant's job title                                   |
| **YOJ**     | Years in current job                                    |
| **DEROG**   | No. of major derogatory reports                         |
| **DELINQ**  | No. of delinquent credit lines                          |
| **CLAGE**   | Age of oldest credit line (in months)                   |
| **NINQ**    | Recent credit inquiries                                 |
| **CLNO**    | Number of existing credit lines                         |
| **DEBTINC** | Debt-to-Income ratio                                    |

---

### ⚙️ Models Developed

Three supervised classification models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest** (baseline and tuned versions)

Each model was assessed using:

* **Accuracy**
* **Recall** (priority metric — minimizing missed defaulters)
* **Precision**

---

### ✅ Final Model Selection & Results

The **Tuned Decision Tree Classifier** outperformed other models based on both performance and interpretability.

📌 **Best Model: Tuned Decision Tree**

* **Accuracy**: 86%
* **Recall**: 74%
* **Precision**: 62%

This model was chosen for:

* Its **high recall**, which helps the bank catch most potential defaulters.
* Its **transparency**, making it suitable for justifying adverse decisions to regulators and applicants.

---

### 💡 Conclusion

This project demonstrates how machine learning can enhance the credit approval process by making it:

* **More efficient** (automated decisions),
* **Fairer** (reduced human bias),
* **Transparent** (interpretable predictions).


