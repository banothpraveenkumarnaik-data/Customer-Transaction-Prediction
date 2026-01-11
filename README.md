# 🏦 Customer Transaction Prediction  
### Banking Domain | Binary Classification | End-to-End Machine Learning Project

---

## 📌 Executive Summary
This project presents a **production-grade machine learning solution** for predicting **future customer transaction behavior** in the banking domain. Using a **large-scale, fully anonymized, high-dimensional dataset**, the model determines whether a customer is likely to perform a transaction in the future, regardless of transaction amount.

The implementation strictly follows **industry-standard ML practices**, ensuring robustness, reproducibility, and deployment readiness.

---

## 🎯 Business Objective
From a banking analytics standpoint, this solution enables:

- Targeted marketing and personalization
- Customer behavior and propensity modeling
- Risk and engagement profiling
- Data-driven customer retention strategies

### Technical Goals
- Solve a **binary classification problem** with severe class imbalance
- Efficiently manage **200+ numerical features**
- Apply **dimensionality reduction** to improve efficiency
- Optimize model performance through **hyperparameter tuning**
- Persist the best-performing model for deployment

---

## 📊 Dataset Overview
| Attribute | Description |
|--------|------------|
| **Domain** | Banking |
| **Total Records** | 200,000 |
| **Features** | 200 numerical (`var_1` to `var_200`) |
| **Identifier** | `ID_code` |
| **Target Variable** | `target` (Binary) |

### Target Definition
- `0` → No future transaction  
- `1` → Future transaction expected  

### Data Characteristics
- Fully anonymized and privacy-compliant
- No missing or null values
- High-dimensional numerical feature space
- Strong class imbalance
- Clean and modeling-ready dataset

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from exploratory analysis:

- Feature distributions are **approximately Gaussian**
- Minimal skewness and noise
- No significant outliers detected
- **Low multicollinearity**, supporting dimensionality reduction
- Dataset suitable for PCA without information loss

---

## ⚙️ Machine Learning Pipeline
A **leakage-free, modular pipeline** was implemented using **scikit-learn Pipelines**, ensuring consistency across training and evaluation.

### Pipeline Components
- **Stratified Train-Test Split**  
  - Preserves original class distribution
- **Feature Scaling**
- **SMOTE (Synthetic Minority Oversampling Technique)**  
  - Addresses class imbalance
- **PCA (Principal Component Analysis)**  
  - Reduces dimensionality  
  - Retains ≥95% explained variance  
  - Improves computational efficiency
- **Cross-Validation**
  - Ensures robust and unbiased evaluation

✔️ No data leakage  
✔️ Fully reproducible pipeline  

---

## 🔧 Hyperparameter Tuning
To maximize model performance, **systematic hyperparameter optimization** was conducted using **GridSearchCV and RandomizedSearchCV**.

### Tuning Strategy
- Cross-validated search over model-specific parameter grids
- Optimization focused on **F1-score** due to class imbalance
- Balanced trade-off between performance and generalization

### Tuned Parameters (Examples)
- **XGBoost**
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`
  - `scale_pos_weight`
- **LightGBM**
  - `num_leaves`
  - `max_depth`
  - `learning_rate`
  - `n_estimators`

Hyperparameter tuning significantly improved minority-class detection and overall model stability.

---

## 🤖 Models Evaluated
The following models were trained, tuned, and evaluated:

- Logistic Regression (with regularization)
- Linear Support Vector Machine (SVM)
- LightGBM (tuned)
- XGBoost (tuned)

---

## 📏 Evaluation Metrics
Given the imbalanced dataset, the following metrics were prioritized:

- Accuracy
- Precision
- Recall
- F1-Score (Primary Metric)

---

## 🏆 Model Selection & Performance
After extensive tuning and evaluation:

- **XGBoost emerged as the top-performing model**
- Achieved an **F1-score of 0.9034**
- Demonstrated strong generalization on unseen data
- Delivered excellent minority-class recall and precision

📌 **Final Selected Model:** `Tuned XGBoost Classifier`

---

## 💾 Model Persistence
- Final tuned model saved using `pickle`
- PCA transformer and preprocessing steps stored together
- Metrics logged for reproducibility
- Ready for:
  - Batch predictions
  - REST API deployment
  - Enterprise-scale inference pipelines

---

## 🛠️ Technology Stack
- **Language:** Python  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Imbalance Handling:** Imbalanced-learn (SMOTE)  
- **Boosting Models:** XGBoost, LightGBM  

---

## 🚀 Key Achievements
- Handled **high-dimensional banking data** effectively
- Reduced dimensionality using **PCA with minimal information loss**
- Improved performance via **hyperparameter tuning**
- Built a **leakage-free, production-ready ML pipeline**
- Delivered a **scalable and optimized classification system**

---

## 📌 Conclusion
This project demonstrates a **real-world, enterprise-ready machine learning workflow** for banking analytics. The solution is optimized, interpretable, and deployment-ready, making it a strong demonstration of applied **data science and machine learning expertise**.

---

## 👤 Author
**Praveen Kumar Naik Banoth**  
*Data Science & Machine Learning Enthusiast*  

🔗 LinkedIn:  
https://www.linkedin.com/in/banoth-praveen-kumar-naik-8017032a4
