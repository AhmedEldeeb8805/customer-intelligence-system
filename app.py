from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import traceback
from image_generator import get_or_generate_image
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Customer Intelligence System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

churn_model = joblib.load("models/churn_model.pkl")
item_similarity_df = joblib.load("models/item_similarity.pkl")
product_lookup = joblib.load("models/product_lookup.pkl")
best_sellers = joblib.load("models/best_sellers.pkl")
avg_price = joblib.load("models/avg_price.pkl")
conf_matrix = joblib.load("models/conf_matrix.pkl")
customer_segments = pd.read_csv("data/customer_segments.csv")


class CustomerRequest(BaseModel):
    is_new_customer: bool
    customer_id: Optional[int] = None
    budget: Optional[float] = None


def safe_get_image(stock_code, description):
    # حماية من القيم الفاضية (NaN) أو الأنواع الغريبة قبل استدعاء الـ API
    if pd.isna(description) or pd.isna(stock_code):
        return None
    return get_or_generate_image(str(stock_code), str(description))


def unified_recommend(is_new_customer, customer_id=None, budget=None, n=10):
    if is_new_customer:
        if budget is not None:
            raw_result = recommend(avg_price, best_sellers, budget)
        else:
            raw_result = best_sellers.head(n)
        product_codes = raw_result['StockCode'].tolist()
    else:
        raw_result = recommend_from_history(customer_id, conf_matrix, item_similarity_df, n)
        product_codes = raw_result.index.tolist()

    codes_df = pd.DataFrame({'StockCode': product_codes})
    final_result = codes_df.merge(product_lookup, on='StockCode', how='left')
    final_result = final_result.head(n)

    final_result['image_path'] = final_result.apply(
        lambda row: safe_get_image(row['StockCode'], row['Description']),
        axis=1
    )
    return final_result


def recommend(avg_price, best_sellers, budget, n=10):
    cheaper_or_equal = avg_price[avg_price['UnitPrice'] <= budget].copy()
    more_expensive = avg_price[avg_price['UnitPrice'] > budget].copy()
    cheaper_or_equal = cheaper_or_equal.sort_values('UnitPrice', ascending=False)
    more_expensive = more_expensive.sort_values('UnitPrice', ascending=True)
    combined = pd.concat([cheaper_or_equal, more_expensive], ignore_index=True)
    return combined.head(n)


def recommend_from_history(customer_id, conf_matrix, item_similarity_df, n=10):
    purchased_products = conf_matrix.loc[customer_id]
    purchased_products = purchased_products[purchased_products > 0].index
    similarity_scores = item_similarity_df.loc[purchased_products].sum(axis=0)
    similarity_scores = similarity_scores.drop(purchased_products)
    return similarity_scores.sort_values(ascending=False).head(n)


@app.post("/predict")
def predict_customer(request: CustomerRequest):
    try:
        if not request.is_new_customer:
            if request.customer_id is None:
                return {"error": "customer_id is required for existing customers"}

            customer_row = customer_segments[customer_segments['CustomerID'] == request.customer_id]
            if customer_row.empty:
                return {"error": "Customer ID not found"}

            segment = customer_row['Segment'].values[0]
            features = customer_row[['frequency_log', 'monetary_log']]
            churn_probability = churn_model.predict_proba(features)[0][1]

            recommendations = unified_recommend(
                is_new_customer=False,
                customer_id=request.customer_id
            )

            return {
                "customer_type": "existing",
                "segment": segment,
                "churn_risk": round(float(churn_probability), 3),
                "recommended_products": recommendations.to_dict(orient="records")
            }

        else:
            recommendations = unified_recommend(
                is_new_customer=True,
                budget=request.budget
            )

            return {
                "customer_type": "new",
                "recommended_products": recommendations.to_dict(orient="records")
            }

    except Exception as e:
        # طباعة الخطأ الكامل في الـ Terminal
        traceback.print_exc()
        # إرجاع رسالة الخطأ في الـ Response نفسه (مؤقتاً، عشان نشخص المشكلة)
        return {"error": str(e)}