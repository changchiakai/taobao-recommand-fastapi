import duckdb
import pandas as pd
from lightgbm import LGBMClassifier
import joblib

print("正在從 Parquet 讀取資料...")
df = duckdb.read_parquet("training.parquet").df()

# ===============================
# Feature / Label
# ===============================
X = df.drop(columns=["label"])
y = df["label"]

print(f"開始訓練模型... 總資料量: {len(df)} 筆")

model = LGBMClassifier(n_estimators=100)
model.fit(X, y)

# ===============================
# ✅ 儲存模型
# ===============================
joblib.dump(model, "model.pkl")

# ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
# ⭐ 新增這行（超重要）
# ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ model.pkl + features.pkl 已儲存")