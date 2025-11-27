import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="多目标 Logit 模型", layout="wide")
st.title("🔍 多目标 Logit（逻辑回归）模型自动建模")

st.sidebar.header("1️⃣ 上传数据")

# 上传 CSV
uploaded_file = st.sidebar.file_uploader("上传已编码好的 CSV 文件", type=["csv"])

if uploaded_file is None:
    st.info("请在左侧上传一个 CSV 文件。")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ 数据上传并读取成功！")
st.subheader("数据预览")
st.dataframe(df.head())

# 找出所有二分类变量
st.sidebar.header("2️⃣ 目标变量设置")
binary_cols = [col for col in df.columns if df[col].nunique() == 2]

if not binary_cols:
    st.error("数据中没有找到二分类变量，无法建立 Logit 模型。")
    st.stop()

st.write("**可作为目标变量（Y）的二分类变量：**")
st.write(binary_cols)

# 允许多选 Y
selected_targets = st.sidebar.multiselect(
    "选择要建模的目标变量（可多选）：",
    options=binary_cols,
    default=binary_cols,
)

if not selected_targets:
    st.warning("请至少选择一个目标变量。")
    st.stop()

# 数值型特征列作为 X
st.sidebar.header("3️⃣ 特征列设置")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.error("没有找到数值型特征列，无法训练逻辑回归。")
    st.stop()

st.write("**数值型特征列（默认作为 X）：**")
st.write(numeric_cols)

feature_cols = st.sidebar.multiselect(
    "选择特征列（X）", options=numeric_cols, default=numeric_cols
)

if not feature_cols:
    st.error("请至少选择一个特征列。")
    st.stop()

# 训练参数
st.sidebar.header("4️⃣ 训练参数")
test_size = st.sidebar.slider("测试集比例", 0.1, 0.4, 0.3, step=0.05)
random_state = st.sidebar.number_input("随机种子 random_state", value=42, step=1)

# 训练模型按钮
if st.sidebar.button("开始训练所有模型"):
    results = []

    for target in selected_targets:
        st.markdown(f"---\n### 🎯 目标变量：`{target}`")

        X = df[feature_cols].copy()
        if target in X.columns:
            X = X.drop(columns=[target])  # 避免泄漏

        y = df[target]
        uniq = y.dropna().unique()

        if len(uniq) != 2:
            st.warning(f"变量 `{target}` 当前不是二分类（唯一值: {uniq}），跳过。")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        st.write(f"- **Accuracy**：{acc:.4f}")
        st.write(f"- **ROC-AUC**：{roc_auc:.4f}")

        results.append({
