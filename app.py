import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="다중 Logit 모델", layout="wide")
st.title("🔍 다중 Logit (로지스틱 회귀) 모델 자동 구축")

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

# 2️⃣ 이진 분류 변수(Y) 탐색
st.sidebar.header("2️⃣ 예측 대상 변수(Y) 선택 (다중 선택 가능)")
binary_cols = [col for col in df.columns if df[col].nunique() == 2]

if not binary_cols:
    st.error("❌ 이진 분류 변수(Y)가 없어 Logit 모델을 구축할 수 없습니다.")
    st.stop()

st.write("**예측 가능한 이진 분류 변수 목록(Y):**")
st.write(binary_cols)

selected_targets = st.sidebar.multiselect(
    "예측할 Y 변수를 선택하세요 (여러 개 선택 가능):",
    options=binary_cols,
    default=binary_cols,  # 기본적으로 전체 선택
)

if not selected_targets:
    st.warning("⚠ 최소 한 개 이상의 Y 변수를 선택해야 합니다.")
    st.stop()

# 3️⃣ X (특징 변수) 선택
st.sidebar.header("3️⃣ 특징 변수(X) 선택")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.error("❌ 수치형 특징 변수가 없어 모델 훈련이 불가능합니다.")
    st.stop()

feature_cols = st.sidebar.multiselect(
    "X 변수(특징)를 선택하세요.", options=numeric_cols, default=numeric_cols
)

if not feature_cols:
    st.error("⚠ 최소 한 개 이상의 X 변수를 선택해야 합니다.")
    st.stop()

# 4️⃣ 훈련 설정
st.sidebar.header("4️⃣ 훈련 설정")
test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.4, 0.3, step=0.05)
random_state = st.sidebar.number_input("랜덤 시드(random_state)", value=42, step=1)

# 🚀 모델 훈련 실행
if st.sidebar.button("모든 모델 훈련 시작"):
    results = []

    for target in selected_targets:
        st.markdown(f"---\n### 🎯 예측 대상(Y): `{target}`")

        X = df[feature_cols].copy()
        if target in X.columns:
            X = X.drop(columns=[target])  # 타겟 변수는 X에서 제거

        y = df[target]

        if y.nunique() != 2:
            st.warning(f"`{target}` 변수는 이진 분류가 아니므로 건너뜁니다.")
            continue

        # 데이터 정리 (결측치 & 무한대 제거)
        data_xy = pd.concat([X, y], axis=1)
        data_xy = data_xy.replace([np.inf, -np.inf], np.nan)
        before = len(data_xy)
        data_xy = data_xy.dropna()
        after = len(data_xy)

        if after < 50:
            st.warning(f"`{target}` 정리 후 샘플 수가 {after}개로 너무 적습니다. 건너뜁니다.")
            continue

        st.write(f"🧹 `{target}` : 결측치/무한대 제거 후 **{before - after}개 삭제**, 남은 샘플 **{after}개**")

        X_clean = data_xy.drop(columns=[target])
        y_clean = data_xy[target]

        if y_clean.nunique() != 2:
            st.warning(f"`{target}` 정리 후 한 개의 클래스만 남아 모델 훈련 불가. 건너뜁니다.")
            continue

        # 훈련/테스트 데이터 분리
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean,
                test_size=test_size,
                random_state=random_state,
                stratify=y_clean
            )
        except ValueError as e:
            st.warning(f"`{target}` 훈련/테스트 분리 오류: {e}")
            continue

        # 모델 훈련
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            st.warning(f"`{target}` 모델 훈련 중 오류 발생: {e}")
            continue

        # 예측
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        st.write(f"- 정확도 (Accuracy): **{acc:.4f}**")

        if y_proba is not None and y_test.nunique() == 2:
            try:
                auc = roc_auc_score(y_test, y_proba)
                st.write(f"- ROC-AUC: **{auc:.4f}**")
            except ValueError:
                auc = np.nan
                st.write("- ROC-AUC 계산 불가")
        else:
            auc = np.nan
            st.write("- ROC-AUC 제공 불가")

        # 결과 저장
        results.append({
            "Target (Y)": target,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(auc, 4) if not np.isnan(auc) else None
        })

    # 결과 요약 테이블
    if results:
        st.subheader("📊 모든 Logit 모델 성능 비교")
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("⚠ 성공적으로 훈련된 모델이 없습니다.")
else:
    st.info("👈 왼쪽 설정을 완료한 후 **모든 모델 훈련 시작** 버튼을 눌러주세요.")
