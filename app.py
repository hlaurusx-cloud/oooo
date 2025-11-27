import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.title("🔍 多目标 Logit 模型自动建模")

# 读取数据
file_path = "/mnt/data/Accepted_data (1).csv"
df = pd.read_csv(file_path)
st.write(df.head())

# 找出所有二分类变量
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
st.subheader("可作为目标变量（Y）的二分类变量：")
st.write(binary_cols)

# 选择特征列（默认：全部数值型）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

results = []

for target in binary_cols:
    X = df[numeric_cols].drop(columns=[target], errors="ignore")
    y = df[target]

    # 训练 / 测试集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Target (Y)": target,
        "Accuracy": round(acc, 4),
        "ROC-AUC": round(auc, 4)
    })

# 输出结果
st.subheader("📊 各 Logit 模型表现对比")
st.dataframe(pd.DataFrame(results))

