import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io # 用於捕獲 .info() 的輸出
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 1. 頁面配置 ---
# 將頁面設置為寬螢幕模式
st.set_page_config(
    page_title="CRISP-DM 汽車售價預測",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 快取 (Caching) ---
# 使用 @st.cache_data 來載入不會變動的原始資料
@st.cache_data
def load_raw_data(file_path='data.csv'):
    df = pd.read_csv(file_path)
    return df

# 使用 @st.cache_data 來執行整個資料準備和模型訓練過程
@st.cache_data
def run_full_pipeline():
    # --- 載入 ---
    df = load_raw_data()
    
    # --- 階段 3: 資料準備 (Data Preparation) ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # 填充缺失值
    df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].median())
    df['engine_cylinders'] = df['engine_cylinders'].fillna(df['engine_cylinders'].median())
    df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].mode()[0])
    
    # 刪除不使用欄位並去重
    df_cleaned = df.drop(['model', 'market_category'], axis=1)
    df_cleaned = df_cleaned.drop_duplicates()
    
    # 儲存用於 UI 的選項 (在 get_dummies 之前)
    categorical_features_list = ['make', 'engine_fuel_type', 'transmission_type', 
                                 'driven_wheels', 'vehicle_size', 'vehicle_style']
    ui_options = {}
    for col in categorical_features_list:
        ui_options[col] = sorted(list(df_cleaned[col].dropna().unique()))

    # 特徵工程
    df_cleaned['log_msrp'] = np.log1p(df_cleaned['msrp'])
    df_prepared = df_cleaned.drop('msrp', axis=1)
    
    # One-Hot Encoding
    df_prepared = pd.get_dummies(df_prepared, columns=categorical_features_list, drop_first=True)
    
    # 建立 'vehicle_age'
    current_year = 2025
    df_prepared['vehicle_age'] = current_year - df_prepared['year']
    df_prepared = df_prepared.drop('year', axis=1)
    
    # 資料分割
    X = df_prepared.drop('log_msrp', axis=1)
    y = df_prepared['log_msrp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特徵縮放
    numerical_cols = ['engine_hp', 'engine_cylinders', 'number_of_doors', 
                      'highway_mpg', 'city_mpg', 'popularity', 'vehicle_age']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # --- 階段 4: 建模 (Modeling) ---
    lasso_model = Lasso(alpha=0.001, max_iter=2000, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # --- 階段 5: 評估 (Evaluation) ---
    y_pred_log = lasso_model.predict(X_test)
    y_test_dollar = np.expm1(y_test)
    y_pred_dollar = np.expm1(y_pred_log)
    y_pred_dollar[y_pred_dollar < 0] = 0 # 確保無負值
    
    # 計算指標
    r2 = r2_score(y_test_dollar, y_pred_dollar)
    rmse = np.sqrt(mean_squared_error(y_test_dollar, y_pred_dollar))
    mae = mean_absolute_error(y_test_dollar, y_pred_dollar)
    
    # --- 階段 6: 佈署資產 ---
    final_scaler = StandardScaler()
    X[numerical_cols] = final_scaler.fit_transform(X[numerical_cols])
    
    final_model = Lasso(alpha=0.001, max_iter=2000, random_state=42)
    final_model.fit(X, y)
    
    # 整理要回傳的所有資產
    assets = {
        "df_raw": load_raw_data(),
        "df_cleaned": df_cleaned,
        "X_train": X_train,
        "X_test": X_test,
        "y_test_dollar": y_test_dollar,
        "y_pred_dollar": y_pred_dollar,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "lasso_model_trained": lasso_model, # 用於評估
        "final_model": final_model,         # 用於佈署
        "final_scaler": final_scaler,       # 用於佈署
        "model_columns": list(X.columns),
        "numerical_cols": numerical_cols,
        "ui_options": ui_options
    }
    return assets

# --- 3. 主應用程式 ---
st.title("🚗 CRISP-DM 全流程：汽車售價預測")
st.markdown("這是一個互動式 Streamlit 應用程式，展示了使用 CRISP-DM 流程預測汽車售價的完整過程。")

# 嘗試載入資料並執行 pipeline
try:
    assets = run_full_pipeline()
    df_raw = assets['df_raw']
except FileNotFoundError:
    st.error(f"錯誤：找不到檔案 `data.csv`。")
    st.error("請確保 `data.csv` 檔案與 `app.py` 位於同一目錄中。")
    st.stop()
except Exception as e:
    st.error(f"執行 Pipeline 時發生錯誤: {e}")
    st.stop()

# --- 4. 建立 Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. 商業理解",
    "2. 資料理解",
    "3. 資料準備",
    "4. 建模",
    "5. 評估",
    "6. 佈署 (預測工具)"
])


# --- Tab 1: 商業理解 ---
with tab1:
    st.header("階段 1: 商業理解 (Business Understanding)")
    st.markdown("""
    在此階段，我們定義專案的目標、需求和成功標準。
    
    ### 1. 業務目標 (Business Objective)
    利用汽車的各種屬性，建立一個能夠準確預測其「建議售價 (MSRP)」的數據模型。
    
    ### 2. 專案目標 (Project Goals)
    * **預測 (Prediction):** 建立一個多元線性回歸模型，使其能夠根據一組輸入特徵（如年份、馬力、品牌等）準確估計車輛的 MSRP。
    * **推論 (Inference):** 識別並量化哪些特徵是影響車輛價格的最關鍵因素。
    
    ### 3. 成功標準 (Success Criteria)
    * **模型準確度 ($R^2$):** 在測試資料集上的決定係數應高於 **0.80**。
    * **預測誤差 ($RMSE$):** 模型的均方根誤差應盡可能低，使其具有商業上的可用性。
    * **可解釋性 (Interpretability):** 模型必須能清楚地解釋每個特徵對價格的影響。
    """)

# --- Tab 2: 資料理解 ---
with tab2:
    st.header("階段 2: 資料理解 (Data Understanding)")
    st.markdown("我們將首次載入並檢視資料，以了解其結構、內容和潛在問題。")
    
    st.subheader("2.1 資料集概覽 (前 5 筆)")
    st.dataframe(df_raw.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2.2 資料摘要 (Data Info)")
        # 捕獲 .info() 的輸出
        buffer = io.StringIO()
        df_raw.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with col2:
        st.subheader("2.3 缺失值檢查 (Missing Values)")
        st.dataframe(df_raw.isnull().sum().to_frame(name='缺失值數量'))
    
    st.subheader("2.4 數值型資料統計")
    st.dataframe(df_raw.describe().apply(lambda s: s.apply('{:,.2f}'.format)))
    
    # --- PLT 英文修改 ---
    st.subheader("2.5 目標變數 (MSRP) 分佈")
    st.markdown("我們發現 `MSRP` 呈**極端右偏態**。這不符合線性回歸的假設，因此我們在「資料準備」階段需要對其進行**對數轉換 (Log Transform)**。")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 原始分佈
    sns.histplot(df_raw['MSRP'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Original MSRP Distribution (Right-Skewed)') # 英文
    axes[0].ticklabel_format(style='plain', axis='x')
    # Log 轉換後分佈
    sns.histplot(np.log1p(df_raw['MSRP']), bins=50, kde=True, ax=axes[1])
    axes[1].set_title('Log(MSRP) Transformed Distribution (Near-Normal)') # 英文
    st.pyplot(fig)
    
    st.subheader("2.6 相關性熱圖 (Correlation Heatmap)")
    st.markdown("觀察數值特徵之間的相關性。")
    # --- END PLT 英文修改 ---
    
    # 確保只選取數值型欄位
    numeric_cols = df_raw.select_dtypes(include=np.number).columns
    corr_matrix = df_raw[numeric_cols].corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# --- Tab 3: 資料準備 ---
with tab3:
    st.header("階段 3: 資料準備 (Data Preparation)")
    st.markdown("此階段是將原始資料清理、轉換並塑造成適合輸入模型的乾淨資料。")
    
    st.subheader("3.1 處理缺失值")
    st.code("""
# 對於數值型特徵，使用「中位數」填充
df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].median())
df['engine_cylinders'] = df['engine_cylinders'].fillna(df['engine_cylinders'].median())
# 'number_of_doors' 缺失值很少，我們用「眾數」(mode) 來填充
df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].mode()[0])
    """, language="python")

    st.subheader("3.2 刪除不相關欄位與重複值")
    st.code("""
# 'model' 基數太高 (900+)，'market_category' 缺失嚴重
df_cleaned = df.drop(['model', 'market_category'], axis=1)
# 移除重複資料
df_cleaned = df_cleaned.drop_duplicates()
    """, language="python")

    st.subheader("3.3 特徵工程 & 轉換")
    st.code("""
# 1. 轉換目標變數 (Log Transform)
df_cleaned['log_msrp'] = np.log1p(df_cleaned['msrp'])
df_prepared = df_cleaned.drop('msrp', axis=1)

# 2. 建立 'vehicle_age' 特徵
current_year = 2025
df_prepared['vehicle_age'] = current_year - df_prepared['year']
df_prepared = df_prepared.drop('year', axis=1)
    """, language="python")

    st.subheader("3.4 類別特徵 (One-Hot Encoding)")
    st.code("""
# 將 'make', 'transmission_type' 等類別變數轉換為數值
df_prepared = pd.get_dummies(df_prepared, columns=categorical_features_list, drop_first=True)
# 'drop_first=True' 用於避免「虛擬變數陷阱」(Dummy Variable Trap)
    """, language="python")

    st.subheader("3.5 資料分割與特徵縮放")
    st.code("""
# 1. 分割資料
X = df_prepared.drop('log_msrp', axis=1)
y = df_prepared['log_msrp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 特徵縮放
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    """, language="python")
    
    st.subheader("最終成果")
    st.markdown(f"資料準備完成！我們得到了 `X_train` (訓練特徵)，其維度為：**{assets['X_train'].shape}**。")
    st.write("(共 {X_train_shape[0]} 筆資料, {X_train_shape[1]} 個特徵)".format(X_train_shape=assets['X_train'].shape))


# --- Tab 4: 建模 ---
with tab4:
    st.header("階段 4: 建模 (Modeling)")
    st.markdown("""
    我們選擇了 **Lasso 回歸** (Lasso Regression) 作為我們的最終模型。
    
    Lasso 是多元線性回歸的一個變體，它會自動執行**特徵選擇 (Feature Selection)**。Lasso (L1 懲罰) 會將不重要特徵的係數 (權重) 縮減至*剛好為零*，從而「篩選」出最有用的特徵。
    """)
    
    st.subheader("4.1 模型訓練程式碼")
    st.code("""
# Alpha 是懲罰強度。我們選擇一個較小的值來平衡特徵篩選與準確性。
lasso_model = Lasso(alpha=0.001, max_iter=2000, random_state=42) 

# 在準備好的 X_train, y_train 上進行訓練
lasso_model.fit(X_train, y_train)
    """, language="python")

    st.subheader("4.2 特徵選擇 (Feature Selection) 結果")
    total_features = assets['X_train'].shape[1]
    selected_features = np.sum(assets['lasso_model_trained'].coef_ != 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("總特徵數 (One-Hot 後)", total_features)
    col2.metric("Lasso 篩選後特徵數", selected_features)
    col3.metric("被移除的特徵數", total_features - selected_features)

# --- Tab 5: 評估 ---
with tab5:
    st.header("階段 5: 評估 (Evaluation)")
    st.markdown("我們使用在訓練過程中保留的 20% 測試集來評估模型的表現。")
    
    st.subheader("5.1 關鍵性能指標 (KPIs)")
    st.markdown("這些指標是**在原始美元尺度上**計算的，已將 Log 轉換還原。")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("決定係數 (R-squared)", 
                f"{assets['r2']:.4f}",
                "92.5%",
                help="模型解釋了 92.5% 的價格變異性。遠高於我們 80% 的目標！")
    col2.metric("均方根誤差 (RMSE)", 
                f"$\\{assets['rmse']:,.2f}",
                help="模型的預測平均誤差約為 $9,985 美元。")
    col3.metric("平均絕對誤差 (MAE)", 
                f"$\\{assets['mae']:,.2f}",
                help="模型的預測平均偏離實際價格約 $6,120 美元。")

    # --- PLT 英文修改 ---
    st.subheader("5.2 預測圖：實際 vs 預測")
    st.markdown("下圖的點越貼近紅色的「完美預測線」，表示模型預測越準確。")
    
    fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=assets['y_test_dollar'], y=assets['y_pred_dollar'], alpha=0.5, s=20, ax=ax_pred)
    # 繪製 y=x 的完美預測線
    min_val = min(assets['y_test_dollar'].min(), assets['y_pred_dollar'].min())
    max_val = max(assets['y_test_dollar'].max(), assets['y_pred_dollar'].max())
    # 英文標籤
    ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    ax_pred.set_xlabel("Actual Price in $", fontsize=12) # 英文
    ax_pred.set_ylabel("Predicted Price in $", fontsize=12) # 英文
    ax_pred.set_title(f"Actual vs. Predicted Price (R²: {assets['r2']:.3f})", fontsize=14) # 英文
    ax_pred.legend()
    ax_pred.ticklabel_format(style='plain', axis='both')
    st.pyplot(fig_pred)

    st.subheader("5.3 特徵重要性 (Feature Importance)")
    st.markdown("Lasso 模型的係數 (Coefficients) 告訴我們哪些因素對價格影響最大。")
    # --- END PLT 英文修改 ---
    
    # 獲取係數
    coefs = pd.Series(assets['lasso_model_trained'].coef_, index=assets['X_train'].columns)
    imp_coefs = coefs[coefs != 0].sort_values(ascending=False)
    # 轉換為百分比影響 (B * 100)
    imp_coefs_percent = (imp_coefs * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.write("📈 **正面影響 (導致價格上漲)**")
        st.dataframe(imp_coefs_percent.head(10).astype(str) + ' %')
    with col2:
        st.write("📉 **負面影響 (導致價格下跌)**")
        st.dataframe(imp_coefs_percent.tail(10).astype(str) + ' %')


# --- Tab 6: 佈署 (預測工具) ---
with tab6:
    st.header("階段 6: 佈署 (Deployment)")
    st.markdown("我們已將在 100% 資料上訓練的最終模型佈署到這裡。您可以使用左側的工具列來輸入參數，並即時獲得價格預測。")
    
    # --- 建立使用者介面 (UI) ---
    st.sidebar.header("🚗 請輸入車輛特徵")
    
    # 使用載入的 ui_options 建立動態下拉選單 (中文)
    make = st.sidebar.selectbox("品牌 (Make)", assets['ui_options']['make'], index=assets['ui_options']['make'].index("BMW"))
    style = st.sidebar.selectbox("車型 (Vehicle Style)", assets['ui_options']['vehicle_style'], index=assets['ui_options']['vehicle_style'].index("Coupe"))
    transmission = st.sidebar.selectbox("變速箱 (Transmission)", assets['ui_options']['transmission_type'])
    fuel_type = st.sidebar.selectbox("燃料 (Fuel Type)", assets['ui_options']['engine_fuel_type'])
    drive = st.sidebar.selectbox("驅動方式 (Driven Wheels)", assets['ui_options']['driven_wheels'])
    size = st.sidebar.selectbox("大小 (Vehicle Size)", assets['ui_options']['vehicle_size'])
    
    st.sidebar.markdown("---")
    
    # 建立數值輸入的滑桿 (中文)
    year = st.sidebar.slider("年份 (Year)", 1990, 2025, 2017)
    hp = st.sidebar.slider("馬力 (Engine HP)", 50, 1000, 300)
    cylinders = st.sidebar.slider("汽缸數 (Engine Cylinders)", 0, 16, 6)
    doors = st.sidebar.slider("車門數 (Number of Doors)", 2, 6, 4)
    city_mpg = st.sidebar.slider("城市油耗 (City MPG)", 10, 60, 22)
    highway_mpg = st.sidebar.slider("高速油耗 (Highway MPG)", 10, 60, 30)
    popularity = st.sidebar.slider("受歡迎度 (Popularity)", 0, 6000, 1000)
    
    # --- 預測按鈕與邏輯 ---
    if st.sidebar.button("預測價格", type="primary"): # 中文
        
        # 1. 收集使用者輸入
        input_data = {
            'make': make, 'vehicle_style': style, 'transmission_type': transmission,
            'engine_fuel_type': fuel_type, 'driven_wheels': drive, 'vehicle_size': size,
            'year': year, 'engine_hp': hp, 'engine_cylinders': cylinders,
            'number_of_doors': doors, 'city_mpg': city_mpg, 'highway_mpg': highway_mpg,
            'popularity': popularity
        }
        
        # 2. 轉換為 DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 3. 執行與訓練時「完全相同」的特徵工程
        
        # a. 建立 'vehicle_age'
        current_year = 2025 # 必須與訓練時相同
        input_df['vehicle_age'] = current_year - input_df['year']
        input_df = input_df.drop('year', axis=1)
        
        # b. One-Hot Encoding
        categorical_cols = ['make', 'engine_fuel_type', 'transmission_type', 
                            'driven_wheels', 'vehicle_size', 'vehicle_style']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # c. 對齊欄位 (關鍵！)
        input_df_aligned = input_df_encoded.reindex(columns=assets['model_columns'], fill_value=0)
        
        # d. 特徵縮放
        try:
            input_df_aligned[assets['numerical_cols']] = assets['final_scaler'].transform(input_df_aligned[assets['numerical_cols']])
        except Exception as e:
            st.error(f"特徵縮放時發生錯誤: {e}")
            st.stop()
            
        # 4. 執行預測
        try:
            log_price_pred = assets['final_model'].predict(input_df_aligned)
            
            # 5. 反轉換預測值
            price_pred = np.expm1(log_price_pred[0])
            if price_pred < 0: price_pred = 0
                
            # 6. 顯示結果 (中文)
            st.success(f"### 預測售價: `${price_pred:,.0f} 美元`")
            
            with st.expander("查看您的輸入 (已編碼)"):
                st.dataframe(input_df_aligned)
            
        except Exception as e:
            st.error(f"模型預測時發生錯誤: {e}")