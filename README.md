# 🎓 Indian Engineering Placement Prediction

Predict whether an Indian engineering student will get placed or not using machine learning models.

---

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
## Model Performance Observations

| ML Model Name        | Observation about model performance |
|---------------------|------------------------------------|
| Logistic Regression  | Did pretty well with 81.8% accuracy. Predicts placed students very well (recall 0.98) but sometimes gives false positives. F1-score is a bit lower. Good as a simple starting model. |
| Decision Tree        | Slightly better accuracy (84.5%) than logistic regression. Catches most placed students (recall 0.925) but has lower precision (0.72), sometimes wrongly predicts students as placed. Can overfit if not careful. |
| kNN                  | Not very strong here (accuracy 72.6%). Finds most placed students (recall 0.968) but also mistakes non-placed students a lot, so F1-score is low (0.70). Sensitive to scaling and data noise. |
| Gaussian Naive Bayes | Decent accuracy (77.2%) and very high recall (0.97). F1-score is moderate because of lower precision. Simple and fast but assumes features are independent. |
| Random Forest        | Best overall. High accuracy (86.9%) and balanced precision and recall. F1-score is the highest (0.93) and AUC is very good. Handles complex patterns well and doesn’t overfit easily. |
| XGBoost              | Not in the table, but usually performs similar or slightly better than Random Forest. Can handle complex patterns and gives high accuracy, precision, and recall. |

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
