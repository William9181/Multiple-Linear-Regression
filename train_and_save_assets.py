import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import joblib # 用於儲存模型和縮放器
import warnings

warnings.filterwarnings('ignore')

print("開始執行模型訓練與資產儲存腳本...")

# --- 1. 載入資料 ---
file_path = 'data.csv'
df = pd.read_csv(file_path)

# --- 2. 資料清理 & 準備 (與階段 3 相同) ---
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 填充缺失值
df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].median())
df['engine_cylinders'] = df['engine_cylinders'].fillna(df['engine_cylinders'].median())
df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].mode()[0])

# 刪除不使用欄位並去重
df_cleaned = df.drop(['model', 'market_category'], axis=1)
df_cleaned = df_cleaned.drop_duplicates()

# 儲存用於 UI 的選項 (在 get_dummies 之前)
# 這些是我們需要讓使用者在 Streamlit 下拉選單中選擇的
categorical_features_list = ['make', 'engine_fuel_type', 'transmission_type', 
                             'driven_wheels', 'vehicle_size', 'vehicle_style']
ui_options = {}
for col in categorical_features_list:
    # 轉換為 list 以便 joblib 儲存
    ui_options[col] = list(df_cleaned[col].dropna().unique())

print(f"  - UI 選項已擷取 (例如: {len(ui_options['make'])} 個品牌)")

# --- 3. 特徵工程 & 轉換 (與階段 3 相同) ---
df_cleaned['log_msrp'] = np.log1p(df_cleaned['msrp'])
df_prepared = df_cleaned.drop('msrp', axis=1)

# One-Hot Encoding
df_prepared = pd.get_dummies(df_prepared, columns=categorical_features_list, drop_first=True)

# 建立 'vehicle_age'
current_year = 2025 # 保持與訓練時一致
df_prepared['vehicle_age'] = current_year - df_prepared['year']
df_prepared = df_prepared.drop('year', axis=1)

# --- 4. 分割 & 縮放 (與階段 3 相同) ---
X = df_prepared.drop('log_msrp', axis=1)
y = df_prepared['log_msrp']

# 儲存模型欄位
# 這是「最關鍵」的一步：我們必須儲存模型訓練時的 87 個欄位順序
model_columns = list(X.columns)
print(f"  - 模型欄位已儲存 (共 {len(model_columns)} 個特徵)")

# 我們將使用「所有」資料來訓練最終的 scaler 和 model
# 既然要佈署了，我們就用 100% 的資料來訓練，以獲得最佳效能
numerical_cols = ['engine_hp', 'engine_cylinders', 'number_of_doors', 
                  'highway_mpg', 'city_mpg', 'popularity', 'vehicle_age']

# 建立並 Fit Scaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("  - 特徵縮放器 (Scaler) 已在完整資料上訓練完成。")

# --- 5. 建模 (與階段 4 相同) ---
# 使用在階段 4 找到的最佳 alpha
lasso_model = Lasso(alpha=0.001, max_iter=2000, random_state=42)
lasso_model.fit(X, y) # 在 100% 的資料上訓練
print("  - Lasso 模型已在完整資料上訓練完成。")

# --- 6. 儲存資產 ---
joblib.dump(lasso_model, 'lasso_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')
joblib.dump(numerical_cols, 'numerical_cols.pkl') # 儲存數值欄位列表
joblib.dump(ui_options, 'ui_options.pkl') # 儲存 UI 下拉選單

print("\n模型訓練與資產儲存完成！")
print("已生成 5 個檔案:")
print("  - lasso_model.pkl (已訓練的模型)")
print("  - scaler.pkl (已訓練的縮放器)")
print("  - model_columns.pkl (模型所需的 87 個欄位列表)")
print("  - numerical_cols.pkl (需要縮放的 7 個數值欄位)")
print("  - ui_options.pkl (Streamlit UI 所需的選項)")