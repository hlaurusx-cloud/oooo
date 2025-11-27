import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

st.set_page_config(
    page_title="지능형 신용평가모형 데모",
    layout="wide"
)

st.title("지능형 신용평가모형 (Logistic Regression)")
st.write("건전 15,000건 / 부실 15,000건 샘플링 후 로지스틱 회귀로 모형을 학습합니다.")


# ------------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------------
st.sidebar.header("1. 데이터 불러오기")

uploaded_file = st.sidebar.file_uploader(
    "처리·인코딩된 데이터 파일 업로드 (CSV 권장)",
    type=["csv", "txt"]
)

if uploaded_file is not None:
    # 필요에 따라 sep, encoding 바꾸세요
    df = pd.read_csv(uploaded_file)
    st.success("데이터 업로드 완료!")
    st.write("데이터 미리보기:")
    st.dataframe(df.head())
else:
    st.warning("왼쪽 사이드바에서 데이터를 업로드해주세요.")
    st.stop()


# ------------------------------------------------------------------
# 2. 타겟 변수/라벨 선택
# ------------------------------------------------------------------
st.sidebar.header("2. 타겟 변수 설정")

# 실제 타겟 컬럼명을 선택하게 함
target_col = st.sidebar.selectbox(
    "부실/건전 라벨이 들어있는 타겟 컬럼 선택",
    options=df.columns
)

# 타겟 값(범주) 보여주기
unique_vals = df[target_col].dropna().unique()
st.sidebar.write(f"타겟 고유값: {unique_vals}")

# 사용자가 '건전' 라벨 값, '부실' 라벨 값 선택
good_value = st.sidebar.selectbox(
    "건전(정상) 대출 라벨 값 선택",
    options=unique_vals
)
bad_value = st.sidebar.selectbox(
    "부실(연체) 대출 라벨 값 선택",
    options=[v for v in unique_vals if v != good_value]
)

# ------------------------------------------------------------------
# 3. 건전 15,000 / 부실 15,000 샘플링
# ------------------------------------------------------------------
st.sidebar.header("3. 샘플링 옵션")
sample_n = 15000

if st.sidebar.button("건전/부실 데이터 샘플링 및 모형 학습"):
    # 건전 / 부실 분리
    good_df = df[df[target_col] == good_value]
    bad_df = df[df[target_col] == bad_value]

    st.write("### 타겟 분포")
    st.write(f"- 건전({good_value}) 데이터 수: {len(good_df)}")
    st.write(f"- 부실({bad_value}) 데이터 수: {len(bad_df)}")

    # 부족하면 replace=True로 중복 허용 샘플
    good_sample = good_df.sample(
        n=sample_n,
        replace=len(good_df) < sample_n,
        random_state=42
    )
    bad_sample = bad_df.sample(
        n=sample_n,
        replace=len(bad_df) < sample_n,
        random_state=42
    )

    sampled_df = pd.concat([good_sample, bad_sample], axis=0)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    st.success(f"건전/부실 각각 {sample_n}건씩 샘플링 완료! (총 {len(sampled_df)}건)")
    st.write("샘플링된 데이터 미리보기:")
    st.dataframe(sampled_df.head())

    # ------------------------------------------------------------------
    # 4. 피처/타겟 분리 + 숫자형 피처만 사용 (이미 인코딩 되어있다고 가정)
    # ------------------------------------------------------------------
    y = sampled_df[target_col]

    # 타겟 컬럼 제거
    X = sampled_df.drop(columns=[target_col])

    # 숫자형 변수만 사용 (원핫 인코딩 완료 상태라는 가정)
    X_num = X.select_dtypes(include=[np.number])

    st.write(f"사용되는 숫자형 피처 수: {X_num.shape[1]}개")

    # ------------------------------------------------------------------
    # 5. 학습/검증 데이터 분할
    # ------------------------------------------------------------------
    test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.4, 0.3, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=test_size, random_state=42, stratify=y
    )

    # 스케일링 (옵션)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 6. 로지스틱 회귀 모형 학습
    # ------------------------------------------------------------------
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs"
    )
    model.fit(X_train_scaled, y_train)

    st.success("로지스틱 회귀 모형 학습 완료!")

    # ------------------------------------------------------------------
    # 7. 성능 평가
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    st.subheader("분류 리포트")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"실제_{good_value}", f"실제_{bad_value}"],
        columns=[f"예측_{good_value}", f"예측_{bad_value}"]
    )
    st.dataframe(cm_df)

    # ROC-AUC (이진 분류일 때만 의미 있음)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        st.subheader("ROC-AUC")
        st.write(f"ROC-AUC: **{auc:.4f}**")
    except Exception as e:
        st.info(f"ROC-AUC 계산 중 오류: {e}")

    # ------------------------------------------------------------------
    # 8. 계수(Feature Importance 비슷하게) 보기
    # ------------------------------------------------------------------
    st.subheader("로지스틱 회귀 계수 (Feature Importance 느낌으로 보기)")
    coef_df = pd.DataFrame({
        "feature": X_num.columns,
        "coef": model.coef_[0]
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    st.dataframe(coef_df[["feature", "coef"]].head(30))
else:
    st.info("사이드바에서 **[건전/부실 데이터 샘플링 및 모형 학습]** 버튼을 눌러주세요.")
