# Customer Intelligence System

An end-to-end Machine Learning project built on the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — combining **customer segmentation**, **churn prediction**, and a **product recommender system** into a single, connected pipeline.

This project was built to apply the full DeepLearning.AI Machine Learning Specialization (Andrew Ng) — supervised learning, neural networks, and unsupervised learning/recommenders — on one real-world dataset, rather than three disconnected exercises.

## The Story

A UK-based online retailer has ~4,300 customers and no clear picture of who they are or who is about to leave. This project answers three connected business questions:

1. **Who are our customers?** → Segment them into meaningful groups based on purchase behavior.
2. **Who is likely to leave?** → Predict churn so the business can act before it happens.
3. **What can we recommend to keep them engaged?** → Suggest products, whether the customer has purchase history or not.

## Dataset

| Property | Value |
|---|---|
| Source | UCI Online Retail dataset |
| Raw rows | ~541,909 transactions |
| Time span | Dec 2010 – Dec 2011 |
| Customers (after cleaning) | 4,335 |
| Products | ~3,659 (after removing non-product codes) |

**Cleaning decisions** (documented in `notebooks/01_data_cleaning.ipynb`):
- Removed ~135K rows with missing `CustomerID` (can't attribute to a customer).
- Removed negative-quantity rows (returns/cancellations).
- Removed administrative stock codes (`M`, `POST`, `DOT`, `C2`, `BANK CHARGES`) that are not real products.

## Methodology & Results

### 1. Customer Segmentation (Unsupervised Learning)

Built RFM (Recency, Frequency, Monetary) features per customer, applied log transformation to handle the natural right-skew of retail spending data, scaled the features, and ran K-Means.

- **K selection**: Compared the Elbow Method against the Silhouette Score — they disagreed (Elbow favored K=4, Silhouette clearly favored **K=3**). Chose K=3 based on Silhouette Score, since it directly measures cluster separation quality.
- **Validation**: PCA (2 components, 93.7% variance explained) confirmed visually distinct clusters.

| Segment | Customers | Avg. Recency | Avg. Frequency | Avg. Monetary |
|---|---|---|---|---|
| Champions | 1,325 | 29.8 days | 9.8 | £5,318 |
| Normal Customers | 2,024 | 54.6 days | 2.0 | £603 |
| Lost | 986 | 255.1 days | 1.4 | £405 |

### 2. Churn Prediction (Supervised Learning)

Defined churn as `Recency > median (51 days)`, then compared four models using the same two features (`frequency_log`, `monetary_log`) to isolate the effect of model choice itself:

| Model | F1-score (Test) |
|---|---|
| Logistic Regression (tuned) | 0.699 |
| Random Forest (tuned) | 0.704 |
| **XGBoost (tuned)** | **0.711** |
| Neural Network | 0.704 |

**Key finding**: the four models converge within ~1% F1-score of each other, indicating the two RFM-derived features are close to their information ceiling — further improvement would require new features, not a more complex model.

The Neural Network was trained with a proper train/validation/test split and diagnosed using its learning curve: no overfitting or underfitting was observed — the model plateaus after ~20 epochs, consistent with the feature-ceiling finding above.

### 3. Recommender System

Three unified strategies behind a single `unified_recommend()` function, chosen based on how much is known about the customer:

| Customer Type | Strategy |
|---|---|
| New, no budget given | Best Sellers (validated to exclude products dominated by a single outlier customer) |
| New, budget given | Products priced near the stated budget, ranked by popularity |
| Existing customer | Item-based Collaborative Filtering (cosine similarity over co-purchase patterns) |

All three return the same output shape (`StockCode`, `Description`, `UnitPrice`), so downstream consumers (e.g. a future front-end) don't need to know which strategy produced the result.

## Tech Stack

- **Core**: Python, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Classical ML**: Scikit-learn (KMeans, LogisticRegression, RandomForest, StandardScaler, PCA, GridSearchCV)
- **Gradient Boosting**: XGBoost
- **Deep Learning**: TensorFlow / Keras
- **Sparse Data**: SciPy
- **Model Persistence**: Joblib

## Repository Structure

```
customer-intelligence-system/
├── README.md
├── requirements.txt
├── data/
│   ├── Online_Retail.xlsx
│   ├── clean_online_retail.csv
│   └── customer_segments.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_rfm_segmentation.ipynb
│   ├── 03_churn_prediction.ipynb
│   ├── 04_neural_network.ipynb
│   └── 05_recommender.ipynb
├── models/
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── churn_model.pkl
│   ├── item_similarity.pkl
│   ├── product_lookup.pkl
│   └── best_sellers.pkl
└── images/
```

## How to Run

```bash
git clone <repo-url>
cd customer-intelligence-system
pip install -r requirements.txt
```

Then run the notebooks in order (01 → 05), as each one saves outputs consumed by the next.

## Key Takeaways

- Choosing K for K-Means isn't always a single clean "elbow" — validating with a second metric (Silhouette Score) can change the decision, and it did here.
- Comparing multiple models on the *same* features (rather than tuning one model in isolation) reveals whether the bottleneck is the model or the data — here, it was the data.
- A recommender system doesn't need one algorithm; routing between strategies based on what's known about the customer (cold start vs. history) is a standard, practical pattern.

---

*Built as a personal machine learning project applying the DeepLearning.AI Machine Learning Specialization.*
