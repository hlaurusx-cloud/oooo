import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="多目标 Logit 模型", layout="wide")
st.title("🔍 多目标 Logit（逻辑回归）模型自动建模")

st.sidebar.header("1️⃣ 上传数据")
uploaded_file = st.sidebar.file_uploader("上传已编码好的 CSV 文件", type=["csv"])

if uploaded_file is None:
    st.info("请在左侧上传一个 CSV 文件。")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ 数据上传并读取成功！")
st.subheader("数据预览")
st.dataframe(df.head())

# 2️⃣ 找所有二分类变量作为目标候选
st.sidebar.header("2️⃣ 选择目标 Y（可多选）")
binary_cols = [col for col in df.columns if df[col].nunique() == 2]

if not binary_cols:
    st.error("❌ 数据中没有二分类变量，无法训练 Logit 模型。")
    st.stop()

st.write("**可用于预测的目标变量（Y）：**")
st.write(binary_cols)

selected_targets = st.sidebar.multiselect(
    "请选择用于预测的目标（可多选）：",
    options=binary_cols,
    default=binary_cols,
)

if not selected_targets:
    st.warning("⚠ 请至少选择一个目标变量。")
    st.stop()

# 3️⃣ 选择特征变量 X（默认全数值型）
st.sidebar.header("3️⃣ 选择特征 X")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.error("❌ 没有数值型特征列。")
    st.stop()

feature_cols = st.sidebar.multiselect(
    "选择特征列（X）", options=numeric_cols, default=numeric_cols
)

if not feature_cols:
    st.error("⚠ 请至少选择一个特征变量。")
    st.stop()

# 4️⃣ 设置训练参数
st.sidebar.header("4️⃣ 训练参数设置")
test_size = st.sidebar.slider("测试集比例", 0.1, 0.4, 0.3, step=0.05)
random_state = st.sidebar.number_input("随机种子", value=42, step=1)

# 🚀 训练按钮
if st.sidebar.button("开始训练所有模型"):
    results = []

    for target in selected_targets:
        st.markdown(f"---\n### 🎯 目标变量：`{target}`")

        X = df[feature_cols].copy()
        if target in X.columns:
            X = X.drop(columns=[target])

        y = df[target]

        if y.nunique() != 2:
            st.warning(f"`{target}` 不是二分类，跳过。")
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
        auc = roc_auc_score(y_test, y_proba)

        st.write(f"- Accuracy：**{acc:.4f}**")
        st.write(f"- ROC-AUC：**{auc:.4f}**")

        # 🔥 核心修复：字典完整闭合 ⬇⬇⬇
        results.append({
            "Target (Y)": target,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(auc, 4)
        })

    if results:
        st.subheader("📊 所有模型表现对比")
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("⚠ 没有成功训练任何模型。")

else:
    st.info("👈 设置完成后，点击按钮开始训练。")
