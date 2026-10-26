# Cellphone Price Range Prediction

A machine learning classification project that predicts the **price range** of a mobile phone
(0 = low, 1 = medium, 2 = high, 3 = very high cost) from its technical specifications, and
shows how the model can help a new phone company price its products.

> The model predicts a price *range*, not an exact price. It is based on historical patterns
> in the supplied dataset and does not account for brand value, marketing or market timing.

## Business Problem
Bob has started a mobile company and wants to compete with big brands. He does not know how
to price the phones his company makes. This project finds which specifications drive price and
builds a model that estimates the price tier of a phone from its specs.

## Project Tasks
1. Complete exploratory data analysis (EDA) of the mobile phone dataset
2. Build a model to predict the price range from phone specifications
3. Business report on how the model helps expand the business, including feature importance
4. Compare multiple models and recommend the best one for production
5. Report on data challenges and the techniques used to address them

## Dataset
- **File:** `datasets_11167_15520_train.csv`
- **Size:** 2,000 phones, 20 features, 1 target (`price_range`)
- **Target classes:** perfectly balanced (500 phones per class)
- **Features include:** battery power, RAM, internal memory, screen size and resolution,
  camera megapixels, clock speed, number of cores, weight, and yes/no features
  (Bluetooth, Wi-Fi, 3G, 4G, dual SIM, touch screen)

## Approach
1. **Data cleaning:** found impossible zeros in `sc_w` (180 rows) and `px_height` (2 rows) and
   replaced them with the median; kept valid zeros in camera columns
2. **EDA:** target balance, distributions, feature vs price-range plots, correlation analysis,
   multicollinearity and outlier checks
3. **Feature engineering:** added total pixels (`px_total`) and screen area (`screen_area`)
4. **Modelling:** stratified 80/20 split; 8 classifiers inside scikit-learn pipelines
   (Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest,
   Gradient Boosting, XGBoost)
5. **Tuning:** GridSearchCV with 5-fold stratified cross-validation
6. **Evaluation:** accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix
7. **Business analysis:** feature importance (three methods), pricing tool, RAM thresholds
   per tier, competitor benchmarking

## Results
| Model | CV Accuracy | Test Accuracy | Macro F1 |
|---|---|---|---|
| **Tuned Logistic Regression** | **0.963** | **0.980** | **0.980** |
| Tuned SVM (linear kernel) | 0.960 | 0.970 | 0.970 |
| Logistic Regression | 0.947 | 0.962 | 0.962 |
| XGBoost | 0.908 | 0.930 | 0.930 |
| Tuned Gradient Boosting | 0.900 | 0.930 | 0.930 |
| Random Forest | 0.866 | 0.888 | 0.887 |
| KNN | 0.486 | 0.510 | 0.516 |

**Best model: Tuned Logistic Regression.** It has the highest cross-validated accuracy, a small
gap between train and test scores, trains in a fraction of a second, and its coefficients are
easy to interpret. All 8 test errors (out of 400) were between neighbouring price classes.

## Key Findings
- **RAM is by far the strongest driver of price** (correlation 0.92). Shuffling it drops model
  accuracy by about 70 points.
- **Battery power and screen resolution** come next; all other features have little influence.
- Bluetooth, Wi-Fi, 3G, 4G, dual SIM and touch screen show almost no relationship with
  price tier in this data.
- For a typical phone, the price tiers start at roughly **1.2 GB, 2.2 GB and 3.1 GB of RAM**.
- About 97% of phones are priced consistently with their specs, which makes competitor
  benchmarking possible.
- Linear models beat tree models here because the class boundaries are close to linear and most
  features are noise; removing the weak features lifted Random Forest from 88.8% to 95.3%.

## Limitations
- The data has no brand, cost, launch date or region
- The model reflects what the market has priced, not what customers would pay
- The test set has 400 phones, so differences of about 1% between top models are within noise

## Repository Contents
| File | Description |
|---|---|
| `PRCP-1009-CellphonePrice.ipynb` | Full analysis: EDA, modelling, comparison, business report, challenges |
| `datasets_11167_15520_train.csv` | Dataset |
| `requirements.txt` | Python packages needed |
| `README.md` | This file |

## How to Run
```bash
git clone https://github.com/<your-username>/CellphonePrice.git
cd CellphonePrice
pip install -r requirements.txt
jupyter notebook PRCP-1009-CellphonePrice.ipynb
```

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost
