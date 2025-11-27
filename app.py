import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

import matplotlib.pyplot as plt

st.set_page_config(page_title="다중 Logit 모델", layout="wide")
st.title("🔍 다중 Logit (로지스틱 회귀) + DT + Hybrid 모델 자동 구축")

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
    st.error("❌ 이진 분류 변수(Y)가 없어 Logit/DT 모델을 구축할 수 없습니다.")
    st.stop()

st.write("**예측 가능한 이진 분류 변수 목록(Y):**")
st.write(binary_cols)

selected_targets = st.sidebar.multiselect(
    "예측할 Y 변수를 선택하세요 (여러 개 선택 가능):",
    options=binary_cols,
    default=binary_cols,  # 기본: 전부 선택
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

# 4️⃣ 훈련 설정 (랜덤 시드는 내부에서 고정)
st.sidebar.header("4️⃣ 훈련 설정")
test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.4, 0.3, step=0.05)
RANDOM_STATE = 42  # 👈 랜덤 시드 고정 (입력창 제거)


# 🚀 모델 훈련 실행
if st.sidebar.button("모든 모델 훈련 시작"):
    # 요약용 결과 (Logit 기준만 모아서 볼 수도 있음)
    summary_rows = []

    for target in selected_targets:
        st.markdown(f"---\n## 🎯 예측 대상(Y): `{target}`")

        # 1. X, y 구성
        X = df[feature_cols].copy()
        if target in X.columns:
            X = X.drop(columns=[target])  # 타깃이 X에 섞여 있으면 제거

        y = df[target]

        if y.nunique() != 2:
            st.warning(f"`{target}` 변수는 이진 분류가 아니므로 건너뜁니다.")
            continue

        # 2. 데이터 정리 (결측치 & 무한대 제거)
        data_xy = pd.concat([X, y], axis=1)
        data_xy = data_xy.replace([np.inf, -np.inf], np.nan)

        before = len(data_xy)
        data_xy = data_xy.dropna()
        after = len(data_xy)

        if after < 50:
            st.warning(f"`{target}` 정리 후 샘플 수가 {after}개로 너무 적습니다. 건너뜁니다.")
            continue

        st.write(
            f"🧹 `{target}` : 결측치/무한대 제거 후 **{before - after}개 삭제**, "
            f"남은 샘플 **{after}개**"
        )

        X_clean = data_xy.drop(columns=[target])
        y_clean = data_xy[target]

        if y_clean.nunique() != 2:
            st.warning(f"`{target}` 정리 후 한 개의 클래스만 남아 모델 훈련 불가. 건너뜁니다.")
            continue

        # 3. 훈련/테스트 분리
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean,
                y_clean,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=y_clean,
            )
        except ValueError as e:
            st.warning(f"`{target}` 훈련/테스트 분리 오류: {e}")
            continue

        # 4. 세 가지 모델 정의
        logit_model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
        dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

        # 5. 모델 훈련
        try:
            logit_model.fit(X_train, y_train)
            dt_model.fit(X_train, y_train)
        except ValueError as e:
            st.warning(f"`{target}` 모델 훈련 중 오류 발생: {e}")
            continue

        # 6. 예측 (확률 포함)
        y_pred_logit = logit_model.predict(X_test)
        y_proba_logit = logit_model.predict_proba(X_test)[:, 1]

        y_pred_dt = dt_model.predict(X_test)
        y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

        # Hybrid: Logit + DT 평균 확률
        y_proba_hybrid = (y_proba_logit + y_proba_dt) / 2
        y_pred_hybrid = (y_proba_hybrid >= 0.5).astype(int)

        # 7. 성능지표 계산
        def get_metrics(y_true, y_pred, y_proba):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                auc = np.nan
            return acc, prec, rec, f1, auc

        acc_logit, prec_logit, rec_logit, f1_logit, auc_logit = get_metrics(
            y_test, y_pred_logit, y_proba_logit
        )
        acc_dt, prec_dt, rec_dt, f1_dt, auc_dt = get_metrics(
            y_test, y_pred_dt, y_proba_dt
        )
        acc_hyb, prec_hyb, rec_hyb, f1_hyb, auc_hyb = get_metrics(
            y_test, y_pred_hybrid, y_proba_hybrid
        )

        # 8. 📊 성능평가 테이블 (DT / Logit / Hybrid)
        st.subheader("📊 성능평가 (DT / Logit / Hybrid)")

        metrics_df = pd.DataFrame(
            {
                "모델": ["Logit", "DT", "Hybrid"],
                "Accuracy": [acc_logit, acc_dt, acc_hyb],
                "Precision": [prec_logit, prec_dt, prec_hyb],
                "Recall": [rec_logit, rec_dt, rec_hyb],
                "F1-score": [f1_logit, f1_dt, f1_hyb],
                "ROC-AUC": [auc_logit, auc_dt, auc_hyb],
            }
        )

        st.dataframe(metrics_df)

        # 9. ROC 곡선 (DT / Logit / Hybrid)
        st.subheader("📈 ROC 곡선 (DT / Logit / Hybrid)")

        fpr_logit, tpr_logit, _ = roc_curve(y_test, y_proba_logit)
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
        fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_proba_hybrid)

        fig, ax = plt.subplots()
        ax.plot(fpr_logit, tpr_logit, label=f"Logit (AUC={auc_logit:.3f})")
        ax.plot(fpr_dt, tpr_dt, label=f"DT (AUC={auc_dt:.3f})")
        ax.plot(fpr_hyb, tpr_hyb, label=f"Hybrid (AUC={auc_hyb:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC 곡선 - {target}")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # 요약용(원하면 나중에 한 번에 비교 가능)
        summary_rows.append(
            {
                "Target (Y)": target,
                "Logit_Accuracy": round(acc_logit, 4),
                "Logit_ROC_AUC": round(auc_logit, 4) if not np.isnan(auc_logit) else None,
                "DT_Accuracy": round(acc_dt, 4),
                "DT_ROC_AUC": round(auc_dt, 4) if not np.isnan(auc_dt) else None,
                "Hybrid_Accuracy": round(acc_hyb, 4),
                "Hybrid_ROC_AUC": round(auc_hyb, 4) if not np.isnan(auc_hyb) else None,
            }
        )

    # 전체 타깃에 대한 요약표 (옵션)
    if summary_rows:
        st.markdown("---")
        st.subheader("📊 성능 요약 (각 Y별 / 모델별 Accuracy & ROC-AUC)")
        st.dataframe(pd.DataFrame(summary_rows))
    else:
        st.warning("⚠ 성공적으로 훈련된 모델이 없습니다.")

else:
    st.info("👈 왼쪽 설정을 완료한 후 **모든 모델 훈련 시작** 버튼을 눌러주세요.")
