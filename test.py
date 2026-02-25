import joblib
import pandas as pd
import duckdb

# 1. 載入練好的大腦 (模型)
model_path = "./lgbm_model.pkl"
model = joblib.load(model_path)
print(f"✅ 已載入模型: {model_path}")

# 2. 讀取需要預測的新資料 (這裡假設是 new_data.parquet)
# 注意：新資料的欄位名稱和順序，必須跟訓練時的 X 一模一樣
print("🚀 讀取待預測資料...")
new_df = duckdb.read_parquet("./training.parquet").df()

# 如果有 label 欄位要先去掉，只留特徵
X_new = new_df.drop(columns=["label"]) if "label" in new_df.columns else new_df

# 3. 執行預測
# predict() 會直接給 0 或 1
predictions = model.predict(X_new)

# predict_proba() 會給機率 (例如：0.98 表示非常有可能是 1)
probabilities = model.predict_proba(X_new)[:, 1]

# 4. 將結果合併回原始資料並儲存
new_df['prediction'] = predictions
new_df['score'] = probabilities

print("📊 預測完成！前 5 筆結果：")
print(new_df[['prediction', 'score']].head())

# 5. 匯出結果
new_df.to_csv("predictions_result.csv", index=False)


print("💾 預測結果已存至 predictions_result.csv")

threshold = 0.01  # 只要機率大於 1%，我們就視為潛在客戶
new_df['potential_buyer'] = (new_df['score'] > threshold).astype(int)

top_potential = new_df.sort_values('score', ascending=False).head(100)

new_df.to_csv("top_potential.csv", index=False)


print("💾 預測結果已存至 top_potential.csv")