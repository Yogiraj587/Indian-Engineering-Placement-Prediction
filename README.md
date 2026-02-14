# 🎓 Indian Engineering Placement Prediction

## Problem Statement
Predict whether an Indian engineering student will get placed or not using machine learning models.

---

## Dataset Description

This dataset contains synthetically generated data representing engineering students from Indian colleges. This dataset contains data regarding the student such as coding skill rating, gender, backlogs count etc. The target variable is "placement_status" which indicates whether the student got placed or not.

This dataset contains 5000 rows and 23 columns.

This dataset consists of 2 csv files:
1. 'indian engineering_students_placement_prediction.csv' - This file contains the main dataset with all the features - 23 features
2. 'placement_targets.csv' - This file contains the target variable "placement_status", "salary" for each student - 3 features

Combining both we have got 25 features since student id is the duplicate column on both

For this problem, I have removed the salary part and considered only the "placement_status" as the target variable for classification.

## Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Gaussian Naive Bayes  
- XGBoost  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  
- Matthews Correlation Coefficient (MCC)  

---

## Model Performance Metrics

The table below shows the performance of different machine learning models on the Indian Engineering Placement dataset:

| ML Model Name        | Accuracy | Precision | Recall  | F1-Score | AUC-ROC | MCC      |
|---------------------|---------|-----------|--------|----------|---------|----------|
| Logistic Regression  | 0.818   | 0.9095    | 0.9802 | 0.8051   | 0.8841  | 0.5338   |
| Decision Tree        | 0.845   | 0.7214    | 0.9254 | 0.8921   | 0.9084  | 0.4076   |
| KNN                  | 0.726   | 0.8361    | 0.9682 | 0.7053   | 0.8161  | 0.3999   |
| Gaussian Naive Bayes | 0.772   | 0.8773    | 0.9717 | 0.7575   | 0.8514  | 0.4554   |
| Random Forest        | 0.869   | 0.8907    | 0.9168 | 0.9327   | 0.9247  | 0.4240   |
| XGBoost              | 0.862   | 0.8892    | 0.9161 | 0.9246   | 0.9203  | 0.4057   |


## Model Performance Observations

| ML Model Name        | Observation about model performance |
|---------------------|------------------------------------|
| Logistic Regression  | Did pretty well with 81.8% accuracy. Predicts placed students very well (recall 0.98) but sometimes gives false positives. F1-score is a bit lower. Good as a simple starting model. |
| Decision Tree        | Slightly better accuracy (84.5%) than logistic regression. Catches most placed students (recall 0.925) but has lower precision (0.72), sometimes wrongly predicts students as placed. Can overfit if not careful. |
| kNN                  | Not very strong here (accuracy 72.6%). Finds most placed students (recall 0.968) but also mistakes non-placed students a lot, so F1-score is low (0.70). Sensitive to scaling and data noise. |
| Gaussian Naive Bayes | Decent accuracy (77.2%) and very high recall (0.97). F1-score is moderate because of lower precision. Simple and fast but assumes features are independent. |
| Random Forest        | Best overall. High accuracy (86.9%) and balanced precision and recall. F1-score is the highest (0.93) and AUC is very good. Handles complex patterns well and doesn’t overfit easily. |
| XGBoost              | Very strong model with 86.2% accuracy. High precision (0.889) and recall (0.916) result in a high F1-score (0.925). Performs almost as well as Random Forest and handles complex patterns efficiently. |

---

## How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Yogiraj587/Indian-Engineering-Placement-Prediction.git
cd Indian-Engineering-Placement-Prediction
```

### Install requirements
```bash
pip install -r requirements.txt
```
### Run
```bash
streamlit run app.py
```
#### You can also see it here on live
https://indian-engineering-placement-prediction.streamlit.app
