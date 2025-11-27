import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="信用评估模型 (HTML 数据版)", layout="wide")

st.title("💳 智能型信用评分模型（逻辑回归）")
st.write("直接使用上传的 **HTML 文件中的表格数据**，进行建模和可视化。")


# ------------------------------------------------------------
# 1. 上传 HTML 文件并解析为多个 DataFrame
# ------------------------------------------------------------
st.sidebar.header("1️⃣ 上传 HTML 数据文件")

uploaded_file = st.sidebar.file_uploader(
    "请选择已经处理好的 HTML 文件",
    type=["html", "htm"]
)

if uploaded_file is None:
    st.warning("请在左侧上传 HTML 文件（.html 或 .htm）")
    st.stop()

# 读 HTML 里的所有 <table> 标签
try:
    tables = pd.read_html(uploaded_file)
except Exception as e:
    st.error(f"读取 HTML 失败：{e}")
    st.stop()

st.sidebar.success(f"已从 HTML 中解析出 {len(tables)} 个表格")

# 选择用于建模的表格 index
table_index = st.sidebar.number_input(
    "选择用于建模的数据表索引（从 0 开始）",
    min_value=0,
    max_value=len(tables)-1,
    value=0,
    step=1
)

df = tables[table_index]
st.write(f"### 使用的表格 (index = {table_index}) 数据预览")
st.dataframe(df.head())


# ------------------------------------------------------------
# 2. 选择目标变量（好/坏客户标签）
# ------------------------------------------------------------
st.sidebar.header("2️⃣ 设置目标变量（好/坏标签）")

# 选择目标列
target_col = st.sidebar.selectbox(
    "请选择目标变量列（例如 target / loan_status 等）",
    options=df.columns
)

# 查看目标列的唯一取值
unique_vals = df[target_col].dropna().unique()
st.sidebar.write("该列的唯一取值：", unique_vals)

if len(unique_vals) < 2:
    st.error("目标列的取值种类少于 2，无法进行二分类建模。请换一个目标列。")
    st.stop()

# 选择“好 / 坏”对应的值
bad_value = st.sidebar.selectbox("请选择『坏客户 / 违约』的标签值", options=unique_vals)
good_value = st.sidebar.selectbox(
    "请选择『好客户 / 正常』的标签值",
    options=[v for v in unique_vals if v != bad_value]
)

st.write(f"**目标列：** `{target_col}`，坏 = `{bad_value}`，好 = `{good_value}`")


# ------------------------------------------------------------
# 3. 按好/坏各抽样 15000 条
# ------------------------------------------------------------
st.sidebar.header("3️⃣ 抽样与模型训练")
sample_n = 15000
test_size = st.sidebar.slider("测试集比例", 0.1, 0.4, 0.3, 0.05)

if st.sidebar.button("开始抽样 + 训练模型"):
    # 分成好/坏两部分
    bad_df = df[df[target_col] == bad_value]
    good_df = df[df[target_col] == good_value]

    st.write("### 原始数据中标签分布")
    st.write(f"- 坏客户({bad_value})：{len(bad_df)} 条")
    st.write(f"- 好客户({good_value})：{len(good_df)} 条")

    # 抽样（如果不足 15000 就放回采样）
    bad_sample = bad_df.sample(
        n=sample_n,
        replace=len(bad_df) < sample_n,
        random_state=42
    )
    good_sample = good_df.sample(
        n=sample_n,
        replace=len(good_df) < sample_n,
        random_state=42
    )

    sampled_df = pd.concat([bad_sample, good_sample], axis=0)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    st.success(f"已从好/坏各抽取 {sample_n} 条样本，共 {len(sampled_df)} 条。")
    st.write("抽样后数据预览：")
    st.dataframe(sampled_df.head())

    # --------------------------------------------------------
    # 4. 特征 / 目标拆分（只用数值型特征）
    # --------------------------------------------------------
    y = sampled_df[target_col]
    X = sampled_df.drop(columns=[target_col])

    # 只取数值型列（HTML 已编码好的话，这里一般都是数值 + 少量字符串列）
    X_num = X.select_dtypes(include=[np.number])
    st.write(f"用于建模的数值型特征个数：{X_num.shape[1]}")

    if X_num.shape[1] == 0:
        st.error("没有检测到数值型特征列，无法进行逻辑回归。请确认数据是否已经编码/数值化。")
        st.stop()

    # --------------------------------------------------------
    # 5. 训练 / 测试集划分 & 标准化
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # 6. 逻辑回归模型训练
    # --------------------------------------------------------
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs"
    )
    model.fit(X_train_scaled, y_train)

    st.success("✅ 逻辑回归模型训练完成！")

    # --------------------------------------------------------
    # 7. 模型评估
    # --------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    if len(np.unique(y)) == 2:
        # 把“坏客户”的概率作为预测概率（需要找对应的列）
        bad_index = list(model.classes_).index(bad_value)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, bad_index]
    else:
        y_pred_proba = None

    st.subheader("📊 分类报告 (Classification Report)")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📉 混淆矩阵 (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"真实_{c}" for c in model.classes_],
        columns=[f"预测_{c}" for c in model.classes_]
    )
    st.dataframe(cm_df)

    if y_pred_proba is not None:
        try:
            auc = roc_auc_score((y_test == bad_value).astype(int), y_pred_proba)
            st.subheader("📈 ROC-AUC")
            st.write(f"ROC-AUC：**{auc:.4f}**（以坏客户 `{bad_value}` 为正类）")
        except Exception as e:
            st.info(f"计算 ROC-AUC 时出错：{e}")

    # --------------------------------------------------------
    # 8. 查看特征系数（重要性）
    # --------------------------------------------------------
    st.subheader("🔍 特征系数（绝对值越大影响越大）")
    coef_df = pd.DataFrame({
        "feature": X_num.columns,
        "coef": model.coef_[0]
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    st.dataframe(coef_df[["feature", "coef"]].head(30))

else:
    st.info("在侧边栏设置好 **HTML 文件 + 目标列 + 好/坏标签** 后，点击「开始抽样 + 训练模型」。")
