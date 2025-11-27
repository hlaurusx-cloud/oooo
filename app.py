import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="Loan_status 예측 모델", layout="wide")
st.title("🔍 Logit (로지스틱 회귀) 모델 - `Loan_status` 예측")

# 1️⃣ 데이터 업로드
st.sidebar.header("1️⃣ 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("인코딩된 CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is None:
    st.info("왼쪽에서 CSV 파일을 업로드해 주세요.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ 데이터 업로드 및 읽기가 완료되었습니다!")
st.subheader("📌 데이터 미리보기")
st.dataframe(df.head())

# 2️⃣ 목표 변수(Y)는 자동으로 Loan_status 로 고정
TARGET = "Loan_status"

if TARGET not in df.columns:
    st.error(f"❌ 데이터에 `{TARGET}` 변수가 없습니다. 존재하는 컬럼명을 다시 확인하세요.")
    st.stop()

st.write(f"**예측 대상(Y) 변수:** `{TARGET}`")

# 3️⃣ 특징(X)는 Loan_status 를 제외한 모든 수치형 변수
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != TARGET]

if not feature_cols:
    st.error("❌ 수치형 특징 변수가 부족합니다. (Loan_status 제외)")
    st.stop()

st.write(f"**자동 선택된 X 변수 목록 ({len(feature_cols)}개):**")
st.write(feature_cols)

# 4️⃣ 훈련 설정
st.sidebar.header("2️⃣ 훈련 설정")
test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.4, 0.3, step=0.05)
RANDOM_STATE = 42  # 내부 고정

# 🚀 모델 훈련 버튼
if st.sidebar.button("Loan_status 예측 모델 훈련 시작"):
    # 🎯 X, y 구성
    X = df[feature_cols]
    y = df[TARGET]

    # 이진 분류인지 확인
    if y.nunique() != 2:
        st.error(f"❌ `{TARGET}` 변수는 이진 분류가 아닙니다. 현재 값: {y.unique()}")
        st.stop()

    # 🧹 결측치 & 무한대 정리
    data_xy = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan)
    before = len(data_xy)
    data_xy = data_xy.dropna()
    after = len(data_xy)

    st.write(f"🧹 결측치/무한대 제거: **{before - after}개 삭제 → 현재 {after}개 샘플 유지**")

    X_clean = data_xy.drop(columns=[TARGET])
    y_clean = data_xy[TARGET]

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_clean
    )

    # 모델 훈련
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 📊 성능 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    st.subheader("📊 모델 성능 지표")
    st.write(f"- 정확도 (Accuracy) : **{acc:.4f}**")
    st.write(f"- 정밀도 (Precision) : **{prec:.4f}**")
    st.write(f"- 재현율 (Recall) : **{rec:.4f}**")
    st.write(f"- F1-score : **{f1:.4f}**")
    st.write(f"- ROC-AUC : **{auc:.4f}**")

    # ROC 곡선 출력
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr}).set_index("FPR")
    st.line_chart(roc_df)

    # 회귀 계수도 보여주기
    coef_df = pd.DataFrame({
        "Feature": X_clean.columns,
        "Coefficient": model.coef_[0],
        "Odds_Ratio (exp(coef))": np.exp(model.coef_[0])
    }).sort_values("Odds_Ratio (exp(coef))", ascending=False)

    st.subheader("📌 회귀 계수 (변수 영향력)")
    st.dataframe(coef_df)

else:
    st.info("👈 CSV 업로드 후 'Loan_status 예측 모델 훈련 시작' 버튼을 클릭하십시오.")
