# Customer Intelligence System

An end-to-end Machine Learning project built on the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — combining **customer segmentation**, **churn prediction**, and a **product recommender system** into a single, connected pipeline.

This project was built to apply the full DeepLearning.AI Machine Learning Specialization (Andrew Ng) — supervised learning, neural networks, and unsupervised learning/recommenders — on one real-world dataset, rather than three disconnected exercises. It goes one step further than the coursework itself: the trained models are served through a live API and a working front-end, so the project is a usable tool, not just a set of notebooks.

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

- **Cleaning decisions** (documented in `notebooks/data_cleaning.ipynb`):
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

All three return the same output shape (`StockCode`, `Description`, `UnitPrice`, `image_path`), so downstream consumers don't need to know which strategy produced the result.

### 4. Deployment: API + Front-End

The three trained systems (segmentation, churn, recommender) are wrapped behind a single **FastAPI** endpoint (`app.py`) so a caller sends customer data once and gets everything back together:

- **Existing customer** (`customer_id`): returns their `Segment`, `churn_risk`, and personalized recommendations from Collaborative Filtering.
- **New customer**: returns Best Sellers, or budget-ranked products if a `budget` is provided — no segment or churn risk, since there's no purchase history to compute them from.

A lightweight **HTML/CSS/JS front-end** (`index.html`) consumes this API directly: a toggle for new vs. existing customer, an ID/budget form, and a color-coded product grid with segment and churn-risk cards.

**Product images**: generated on first request via the Hugging Face Inference API (`image_generator.py`) and cached to disk (`images/products/`), so each product is only generated once regardless of how many times it's recommended afterward. If image generation is unavailable (e.g. provider credits exhausted), the front-end falls back to a colored category icon instead of breaking the recommendation itself.

## Tech Stack

- **Core**: Python, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Classical ML**: Scikit-learn (KMeans, LogisticRegression, RandomForest, StandardScaler, PCA, GridSearchCV)
- **Gradient Boosting**: XGBoost
- **Deep Learning**: TensorFlow / Keras
- **Sparse Data**: SciPy
- **Model Persistence**: Joblib
- **Deployment**: FastAPI, Uvicorn
- **Image Generation**: Hugging Face Inference API (`huggingface_hub`), python-dotenv
- **Front-End**: HTML, CSS, vanilla JavaScript

## Repository Structure

```
customer-intelligence-system/
├── README.md
├── requirements.txt
├── .env                          ← not committed (see Setup below)
├── .gitignore
├── app.py                        ← FastAPI backend (Phase 8)
├── image_generator.py            ← product image generation + caching
├── index.html                    ← front-end
├── data/
│   ├── clean_online_retail.csv   ← not committed (regenerate, see below)
│   ├── customer_segments.csv
│   └── customer_segments_with_churn.csv
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── rfm_segmentation.ipynb
│   ├── churn_prediction.ipynb
│   ├── neural_network.ipynb
│   └── recommender.ipynb
├── models/
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── churn_model.pkl
│   ├── product_lookup.pkl
│   ├── avg_price.pkl
│   ├── best_sellers.pkl
│   ├── item_similarity.pkl       ← not committed (regenerate, see below)
│   └── conf_matrix.pkl           ← not committed (regenerate, see below)
└── images/
    └── products/                 ← cached generated product images
```

**Note on excluded files**: `item_similarity.pkl` and `conf_matrix.pkl` exceed GitHub's 100 MB file size limit and are excluded via `.gitignore`. Both are deterministic outputs of `05_recommender.ipynb` — running that notebook once regenerates them locally. The raw dataset (`Online_Retail.xlsx`) and its cleaned version are excluded for the same size-related reason.

## Setup & How to Run

**1. Clone and install dependencies**

```bash
git clone https://github.com/AhmedEldeeb8805/customer-intelligence-system.git
cd customer-intelligence-system
pip install -r requirements.txt
```

**2. Get the raw dataset**

Download `Online Retail.xlsx` from the [UCI repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it in `data/`.

**3. Run the notebooks in this order**

```
data_cleaning.ipynb → rfm_segmentation.ipynb → churn_prediction.ipynb → neural_network.ipynb → recommender.ipynb
```

Each notebook saves outputs consumed by the next, and `recommender.ipynb` regenerates the two large model files excluded from the repo (`item_similarity.pkl`, `conf_matrix.pkl`).

**4. Set up image generation (optional)**

Create a `.env` file in the project root with a Hugging Face token that has *"Make calls to Inference Providers"* permission:

```
HF_TOKEN=hf_your_token_here
```

Without this step, the API still works — product cards simply fall back to a colored category icon instead of a generated image.

**5. Run the API**

```bash
uvicorn app:app --reload
```

**6. Open the front-end**

Open `index.html` in a browser (or serve it, e.g. via VS Code Live Server) while the API is running on `127.0.0.1:8000`.

## Key Takeaways

- Choosing K for K-Means isn't always a single clean "elbow" — validating with a second metric (Silhouette Score) can change the decision, and it did here.
- Comparing multiple models on the *same* features (rather than tuning one model in isolation) reveals whether the bottleneck is the model or the data — here, it was the data.
- A recommender system doesn't need one algorithm; routing between strategies based on what's known about the customer (cold start vs. history) is a standard, practical pattern.
- Deployment surfaces problems notebooks hide: a 100 MB+ similarity matrix works fine locally but breaks a `git push`, and a single failed image-generation call can silently take down an otherwise-successful API response if it isn't isolated with its own error handling.

---

*Built as a personal machine learning project applying the DeepLearning.AI Machine Learning Specialization.*
