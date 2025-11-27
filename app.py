import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="多目标 Logit 模型", layout="wide")
st.title("🔍 多目标 Logit（逻辑回归）模型自动建模")

# 1️⃣ 上传数据
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
    default=binary_cols,  # 默认全选所有二分类变量
)

if not selected_targets:
    st.warning("⚠ 请至少选择一个目标变量。")
    st.stop()

# 3️⃣ 选择特征变量 X（默认使用所有数值型列）
st.sidebar.header("3️⃣ 选择特征 X")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.error("❌ 没有数值型特征列，无法训练 Logit 模型。")
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

        # 1. 构造 X, y
        X = df[feature_cols].copy()
        if target in X.columns:
            X = X.drop(columns=[target])  # 避免目标变量混入特征

        y = df[target]

        # 再次确认是二分类
        if y.nunique() != 2:
            st.warning(f"`{target}` 不是二分类，跳过。")
            continue

        # 2. 合并后统一清洗 NaN / Inf
        data_xy = pd.concat([X, y], axis=1)

        # 替换无穷为 NaN
        data_xy = data_xy.replace([np.inf, -np.inf], np.nan)

        # 丢弃含 NaN 的行
        before = len(data_xy)
        data_xy = data_xy.dropna()
        after = len(data_xy)

        if after < 50:
            st.warning(f"`{target}` 清洗后只剩 {after} 条样本，样本太少，跳过。")
            continue

        st.write(f"✅ 目标 `{target}`：已删除含缺失值或无穷值的样本 {before - after} 条，剩余 {after} 条。")

        # 拆回 X, y
        X_clean = data_xy.drop(columns=[target])
        y_clean = data_xy[target]

        # 确保两类样本都还在
        if y_clean.nunique() != 2:
            st.warning(f"`{target}` 清洗后只剩一个类别，无法训练，跳过。")
            continue

        # 3. 划分训练/测试集
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean,
                test_size=test_size,
                random_state=random_state,
                stratify=y_clean
            )
        except ValueError as e:
            st.warning(f"`{target}` 在划分训练/测试集时出错：{e}，跳过。")
            continue

        # 4. 训练逻辑回归（Logit）
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            st.warning(f"`{target}` 在训练模型时出错：{e}，跳过。")
            continue

        # 5. 预测与评估
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        st.write(f"- Accuracy：**{acc:.4f}**")

        if y_proba is not None and y_test.nunique() == 2:
            try:
                auc = roc_auc_score(y_test, y_proba)
                st.write(f"- ROC-AUC：**{auc:.4f}**")
            except ValueError:
                auc = np.nan
                st.write("- ROC-AUC：无法计算（可能是预测概率异常）")
        else:
            auc = np.nan
            st.write("- ROC-AUC：无法计算")

        # 6. 保存结果用于汇总
        results.append({
            "Target (Y)": target,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(auc, 4) if not np.isnan(auc) else None
        })

    # 7. 汇总结果表
    if results:
        st.subheader("📊 所有模型表现对比")
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("⚠ 没有成功训练任何模型。")
else:
    st.info("👈 设置完成后，点击左侧按钮开始训练所有模型。")
