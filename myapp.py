from fastapi import FastAPI, HTTPException
import joblib
import duckdb
import pandas as pd
from lightgbm import LGBMClassifier

# ===============================
# 1️⃣ 啟動時載入模型（只載一次）
# ===============================
print("🚀 Loading model...")

MODEL_PATH = "./lgbm_model.pkl"
FEATURE_PATH = "./features.pkl"
DATA_PATH = "./training.parquet"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)
df = duckdb.read_parquet(DATA_PATH).df()
print("✅ Model Ready")

# ===============================
# 2️⃣ 建立 FastAPI
# ===============================
app = FastAPI(
    title="LightGBM Recommendation API",
    version="1.0"
)

# ===============================
# 3️⃣ Health Check
# ===============================
@app.get("/health")
def health():
    return {"status": "ok"}

# ===============================
# 4️⃣ 取得使用者特徵
# ===============================
def load_feature_by_vid(vid: str):

    user_df = df[df["user_id"] == int(vid)].copy()

    return user_df


# ===============================
# 5️⃣ 排序推薦
# ===============================
def rank_items(feature_df):

    if feature_df.empty:
        return feature_df

    # 只拿模型需要的欄位
    X = feature_df[feature_columns]

    # LightGBM 預測機率
    feature_df["score"] = model.predict_proba(X)[:, 1]

    ranked = feature_df.sort_values(
        "score",
        ascending=False
    )

    return ranked


# ===============================
# 6️⃣ 推薦 API
# ===============================
@app.get("/recommend/{vid}")
def recommend(vid: str, top_k: int = 20):

    feature_df = load_feature_by_vid(vid)

    if feature_df.empty:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    ranked = rank_items(feature_df)

    result = ranked.head(top_k)[
        ["user_id", "product_id", "score"]
    ].to_dict("records")

    return {
        "vid": vid,
        "top_k": top_k,
        "recommendations": result
    }