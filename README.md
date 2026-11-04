# Customer Churn Prediction: ML Assignment 2

**Live app:** https://ml-assignment2-bf8ar9aygvepffyum4drwa.streamlit.app/

## a. Problem statement

A company loses revenue every time a customer cancels their service. The goal of this project is to predict whether a customer will **churn** (leave) from their account details, contract type, services and billing information. This is a **binary classification** problem (Churn = Yes / No). Five classifiers are trained and compared on the same data, and the results are shown in an interactive Streamlit app.

## b. Dataset description

| Item               | Detail                                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| File               | `IT_customer_churn.csv`                                                                                        |
| Source             | [Telecom Customer Churn (Kaggle, alipakzad)](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn) |
| Rows               | 7,043 (7,021 after removing 22 duplicates)                                                                     |
| Columns            | 20 (19 input features + target `Churn`)                                                                        |
| Task               | Binary classification: `Churn` = Yes / No                                                                      |
| Class balance      | About 73.5% No, 26.5% Yes (imbalanced)                                                                         |
| Train / test split | 80 / 20, stratified, `random_state = 2026` (5,616 train, 1,405 test)                                           |

**Features:** gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.

**Preprocessing:**

- `TotalCharges` was stored as text; 11 blank values (all customers with tenure 0) were set to 0.
- Duplicate rows were removed.
- Categorical columns are one-hot encoded; numeric columns are standardised for Logistic Regression, KNN, SVM and Naive Bayes (not needed for the Decision Tree).
- Three engineered features were added: `addon_count` (number of add-on services), `avg_monthly_spend` (TotalCharges / tenure) and `charge_gap` (MonthlyCharges minus average monthly spend).
- Preprocessing and model are saved together as one scikit-learn Pipeline per model, so the scaler is fitted on training data only.

## c. GitHub repository link

[https://github.com/ruchinagrwal/ml-assignment2]

```
├── app.py                  Streamlit app
├── churn_utils.py          shared cleaning, feature engineering and metrics code
├── train_models.py         trains and saves the five models
├── requirements.txt
├── test_data.csv           held-out test set (1,405 rows) used by the app
├── IT_customer_churn.csv   original dataset
└── model/                  five saved .joblib pipelines + metrics_summary.csv
```

## d. Models used

### Comparison table (test set, 1,405 customers)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression    | 0.7986   | 0.8408 | 0.6440    | 0.5349 | 0.5844 | 0.4564 |
| Decision Tree          | 0.7907   | 0.8221 | 0.6154    | 0.5591 | 0.5859 | 0.4473 |
| KNN                    | 0.7794   | 0.8048 | 0.5969    | 0.5134 | 0.5520 | 0.4088 |
| Naive Bayes            | 0.7025   | 0.8142 | 0.4654    | 0.8306 | 0.5965 | 0.4304 |
| Support Vector Machine | 0.8021   | 0.7917 | 0.6691    | 0.5000 | 0.5723 | 0.4551 |

Model settings: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=6`, `min_samples_leaf=20`), KNN (`n_neighbors=15`, distance-weighted), Gaussian Naive Bayes, SVM (RBF kernel, `C=1.0`, `probability=True`). Settings were chosen as reasonable defaults and were not tuned.

### Observations on model performance

| ML Model Name                       | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression                 | Best AUC (0.841) and best MCC (0.456), with accuracy of 79.9%, only slightly behind SVM. It is the most consistent model across all six metrics. Its recall is moderate (0.535), so it still misses almost half of the customers who churn. A linear model works well here because churn depends strongly on a few clear factors such as contract type, tenure and monthly charges.                              |
| Decision Tree                       | Accuracy 0.791 and AUC 0.822, a little below Logistic Regression, with slightly higher recall (0.559). Limiting the depth to 6 and requiring at least 20 samples per leaf kept it from memorising the training data, so it generalises reasonably. Its advantage is that the rules are easy to read and explain.                                                                                                 |
| KNN                                 | Weakest overall, with the lowest F1 (0.552) and lowest MCC (0.409). After one-hot encoding there are many columns, and distance-based methods get less reliable as the number of dimensions grows, because points start to look equally far from each other. It is also slower at prediction time.                                                                                                               |
| Naive Bayes                         | Lowest accuracy (0.703) and precision (0.465), but the highest recall (0.831) and highest F1 (0.597). It catches most churners, but also flags many loyal customers by mistake. This is likely because it assumes features are independent, which is not true here (for example, tenure and total charges are closely related). It suits cases where missing a churner costs more than a wasted retention offer. |
| Support Vector Machine              | Best accuracy (0.802) and best precision (0.669), but the lowest AUC (0.792) and a recall of only 0.500. Its accuracy is partly helped by the class imbalance, since about 73% of customers do not churn. Its probability scores come from a separate calibration step, which may be part of why its AUC is weaker.                                                                                              |
| **Overall winner for this dataset** | **Logistic Regression.** It has the best AUC and MCC, which are more reliable than accuracy on imbalanced data, and its accuracy is within half a percentage point of the top model. It is also simple, fast and explainable. If catching as many churners as possible were the priority, Naive Bayes would be the alternative because of its recall.                                                            |

_Note: results come from a single train/test split, so small differences between the top models should not be over-interpreted._

## Streamlit app features

- Upload a test CSV (must contain all input columns and the `Churn` column)
- Model selection dropdown (all five models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC, plus ROC curve
- Confusion matrix and classification report
- All-model comparison on the uploaded file
- Decision-threshold slider (0.50 uses each model's default predictions)
- Customer risk table ranked by churn probability, downloadable as CSV

The app loads the saved models and never retrains on uploaded data.

## installation steps

```
pip install -r requirements.txt
python train_models.py      # trains and saves models, writes test_data.csv
streamlit run app.py
```
