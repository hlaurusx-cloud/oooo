import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

st.set_page_config(page_title="Logit信用评估模型", layout="wide")

st.title("🧮 Logit（逻辑回归）信用评估模型 Demo")

st.markdown(
    """
这个小应用用于从你已经编码好的数据中，构建一个**二分类 Logit 模型**（例如：0=正常，1=不良）。
- 左侧上传 CSV 数据
- 选择目标列（违约标记）
- 模型会自动进行 15000 / 15000 的样本抽样（如果样本数量足够）
- 输出模型表现和主要指标
"""
)

# ========== 1. 数据上传 ==========
st.sidebar.header("1. 上传或选择数据")

uploaded_file = st.sidebar.file_uploader("上传已编码好的 CSV 文件", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 数据上传成功！")
else:
    st.info("📂 请在左侧上传一个 CSV 文件。")
    st.stop()

st.subheader("原始数据预览")
st.dataframe(df.head())

# ========== 2. 选择目标列和特征列 ==========
st.sidebar.header("2. 模型设置")

# 选择目标列
target_col = st.sidebar.selectbox(
    "选择目标（Y）列（例如: target / bad_flag / default）",
    options=df.columns,
)

# 默认把除目标列以外的所有数值型列作为特征
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols_default = [c for c in numeric_cols if c != target_col]

feature_cols = st.sidebar.multiselect(
    "选择特征列（X）", options=df.columns, default=feature_cols_default
)

if not feature_cols:
    st.error("❗ 你至少需要选择一个特征列。")
    st.stop()

# ========== 3. 15000 / 15000 抽样 ==========
st.sidebar.header("3. 抽样与训练参数")

balance_sample = st.sidebar.checkbox("对0/1样本做 15000 / 15000 抽样（如果可能）", value=True)

test_size = st.sidebar.slider("测试集比例", 0.1, 0.4, 0.3, step=0.05)
random_state = st.sidebar.number_input("随机种子 random_state", value=42, step=1)

# 提取 X, y
X = df[feature_cols].copy()
y = df[target_col].copy()

# 确保是二分类
unique_y = sorted(y.dropna().unique())
if len(unique_y) != 2:
    st.error(f"❗ 目标变量 {target_col} 不是二分类（当前唯一值: {unique_y}）。")
    st.stop()

st.write(f"目标变量 **{target_col}** 的取值分布：")
st.write(y.value_counts())

if balance_sample:
    # 假设较小的那个类别是正类/负类都无所谓，只做平衡
    class0 = unique_y[0]
    class1 = unique_y[1]

    df_all = pd.concat([X, y], axis=1)

    df_0 = df_all[df_all[target_col] == class0]
    df_1 = df_all[df_all[target_col] == class1]

    n0 = len(df_0)
    n1 = len(df_1)
    st.write(f"类 {class0} 样本数: {n0}, 类 {class1} 样本数: {n1}")

    # 这里按题意：如果足够则各取15000，否则取 min(15000, 实际数量)
    n_sample_each = 15000
    n0_sample = min(n_sample_each, n0)
    n1_sample = min(n_sample_each, n1)

    df_0_sample = df_0.sample(n=n0_sample, random_state=random_state)
    df_1_sample = df_1.sample(n=n1_sample, random_state=random_state)

    df_balanced = pd.concat([df_0_sample, df_1_sample], axis=0)
    st.success(
        f"已进行样本抽样：类 {class0} = {n0_sample}，类 {class1} = {n1_sample}，总计 {len(df_balanced)} 条。"
    )

    X = df_balanced[feature_cols]
    y = df_balanced[target_col]

# ========== 4. 划分训练/测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

st.subheader("样本信息")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("训练集样本数", len(X_train))
with col2:
    st.metric("测试集样本数", len(X_test))
with col3:
    st.metric("特征数", X.shape[1])

# ========== 5. 训练 Logit 模型 ==========
st.header("模型训练：Logistic Regression (Logit)")

penalty = st.sidebar.selectbox("正则化方式 penalty", ["l2", "l1", "none"])
solver_map = {
    "l2": "lbfgs",
    "l1": "liblinear",
    "none": "lbfgs",
}
solver = solver_map[penalty]

C = st.sidebar.number_input("正则化强度 C（越小越强）", value=1.0, step=0.1)

if st.sidebar.button("开始训练模型"):
    with st.spinner("模型训练中..."):
        # sklearn 的 LogisticRegression
        model = LogisticRegression(
            penalty=None if penalty == "none" else penalty,
            C=C,
            solver=solver,
            max_iter=1000,
        )
        model.fit(X_train, y_train)

    st.success("✅ 模型训练完成！")

    # ========== 6. 模型系数（Logit） ==========
    st.subheader("模型系数（Log-Odds & Odds Ratio）")

    coef = model.coef_.flatten()
    intercept = model.intercept_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef (log-odds)": coef,
            "odds_ratio = exp(coef)": np.exp(coef),
        }
    ).sort_values("odds_ratio = exp(coef)", ascending=False)

    st.write(f"Intercept (截距) = {intercept:.4f}")
    st.dataframe(coef_df, use_container_width=True)

    # ========== 7. 模型评估 ==========
    st.subheader("模型评估")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy 准确率", f"{acc:.4f}")
    with col2:
        st.metric("ROC-AUC", f"{auc:.4f}")

    st.markdown("**分类结果报告（classification report）**")
    st.text(classification_report(y_test, y_pred))

    # ========== 混淆矩阵 ==========
    st.markdown("**混淆矩阵（Confusion Matrix）**")
    cm = confusion_matrix(y_test, y_pred, labels=unique_y)
    cm_df = pd.DataFrame(cm, index=[f"True {v}" for v in unique_y],
                         columns=[f"Pred {v}" for v in unique_y])
    st.dataframe(cm_df)

    # ========== ROC 曲线 ==========
    st.markdown("**ROC 曲线**")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": thresholds})

    st.line_chart(roc_df, x="FPR", y="TPR")

else:
    st.info("👈 在侧边栏设置好参数后，点击 **开始训练模型**。")
