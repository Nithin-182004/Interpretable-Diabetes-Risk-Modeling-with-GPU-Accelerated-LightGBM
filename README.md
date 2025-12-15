# Interpretable-Diabetes-Risk-Modeling-with-GPU-Accelerated-LightGBM

## 📌 Project Overview
This project presents a robust and interpretable machine learning pipeline to predict the **risk of diabetes** using demographic, lifestyle, and clinical health indicators.

The solution emphasizes:
- EDA-driven modeling decisions
- Proper metric alignment
- Cross-validation stability and generalization

The final model achieves a **mean ROC-AUC of ~0.727** using **GPU-accelerated LightGBM** with **stratified cross-validation**.

---

## 🎯 Problem Statement
Given structured tabular data containing health metrics such as BMI, blood pressure, cholesterol levels, lifestyle habits, and medical history, the objective is to **predict the probability of diabetes diagnosis** for unseen individuals.

This is a **binary classification problem**, where:
- `0` → No diabetes
- `1` → Diagnosed with diabetes

The evaluation metric is **ROC-AUC**, emphasizing **probability ranking** rather than fixed-threshold classification.

---

## 🗂️ Dataset Description
- **Type:** Synthetic, high-signal tabular healthcare dataset
- **Total Rows:** ~700,000 (train + test)

### Feature Types
- **Numerical:**  
  Age, BMI, blood pressure, cholesterol, triglycerides, physical activity levels, etc.
- **Categorical:**  
  Gender, ethnicity, smoking status, employment status
- **Ordinal:**  
  Education level, income level
- **Binary Flags:**  
  Family history of diabetes, hypertension, cardiovascular history

**Target Variable:**  
`diagnosed_diabetes` (0 = No, 1 = Yes)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from EDA:
- Mild class imbalance (~60–65% positive class)
- Strong non-linear relationships between diabetes risk and:
  - BMI
  - Waist-to-hip ratio
  - Triglycerides
  - Hypertension history
- Ordinal features (education, income) exhibit meaningful ordering
- No missing values or data leakage detected

EDA findings directly informed feature handling, encoding strategy, and model selection.

---

## 🧠 Feature Engineering & Encoding Strategy
- Dropped ID column to eliminate non-informative noise
- **Numerical features:** Used as-is (tree-based models do not require scaling)
- **Ordinal features:** Manually ordinal-encoded to preserve ordering
- **Nominal categorical features:** Handled using LightGBM’s native categorical support
- **Binary medical flags:** Retained without transformation due to strong predictive signal

This strategy minimizes unnecessary preprocessing while preserving domain relevance.

---

## 🔁 Model Training & Validation
- **Model:** LightGBM (Gradient Boosted Decision Trees)
- **Hardware:** GPU-enabled training
- **Cross-validation:** Stratified 5-Fold CV
- **Early stopping:** Enabled to prevent overfitting
- **Evaluation metric:** ROC-AUC

### Cross-Validation Performance
- **Mean ROC-AUC:** ~0.727
- Low fold-to-fold variance → strong generalization
- Controlled train–validation gap → effective regularization

---

## 📈 Why ROC-AUC?
ROC-AUC was selected because:
- The dataset is mildly imbalanced
- The objective focuses on **risk ranking**, not hard classification
- It is **threshold-independent**
- It provides stable and comparable model evaluation

Classification metrics were intentionally deferred to post-model-selection phases where decision thresholds are business-defined.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM (GPU)
- Matplotlib, Seaborn

---

## 📚 References
1. Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).
2. Friedman, J. H. *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics.
3. Kuhn, M., & Johnson, K. *Applied Predictive Modeling*. Springer.
4. Fawcett, T. *An Introduction to ROC Analysis*. Pattern Recognition Letters.
5. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
