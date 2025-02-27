# inference_dart.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# 학습 시 사용한 전처리 함수들을 정의한다.
age_map_surgery = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38,
    "만38-40세": 39,
    "만40-42세": 41,
    "만43-44세": 43,
    "만45-50세": 47,
    "알 수 없음": np.nan,
    "알_수_없음": np.nan
}

def parse_age_surgery(x):
    return age_map_surgery.get(str(x).strip(), np.nan)

def parse_cycle_count(x):
    if isinstance(x, str):
        if "이상" in x:
            return 6
        elif "회" in x:
            return pd.to_numeric(x.replace("회", ""), errors="coerce")
    return pd.to_numeric(x, errors="coerce")

def parse_bool(x):
    if pd.isna(x):
        return -1
    if isinstance(x, str):
        lx = x.strip().lower()
        if lx in ["true", "1"]:
            return 1
        elif lx in ["false", "0"]:
            return 0
        else:
            return -1
    val = pd.to_numeric(x, errors="coerce")
    if pd.isna(val):
        return -1
    return 1 if val == 1 else 0

def safe_label_transform(series, le: LabelEncoder, unknown_str="unknown"):
    known_labels = set(le.classes_)
    def map_func(v):
        v_str = str(v).strip()
        if v_str not in known_labels:
            return unknown_str
        else:
            return v_str
    tmp = series.astype(str).apply(map_func)
    return le.transform(tmp)

#########################
# 추론 함수를 정의한다.
#########################
def inference(
    test_path="./0_rawdataset/test.csv",
    submission_path="submission.csv",
    model_path="best_lgb_model.pkl",
    scaler_path="scaler.pkl",
    features_path="train_features.pkl",
    labelenc_path="label_encoders.pkl"  # 학습 시 저장한 LabelEncoder 사전 파일
):
    # 테스트 데이터를 로드하고 컬럼명에 공백이 있다면 언더바로 치환한다.
    test_df = pd.read_csv(test_path)
    test_df.columns = [c.replace(" ", "_") for c in test_df.columns]

    # ID 컬럼이 존재하면 추출하고, 없으면 기본 ID를 생성한다.
    if "ID" in test_df.columns:
        test_ids = test_df["ID"].values
    else:
        test_ids = [f"TEST_{i:05d}" for i in range(len(test_df))]

    # 학습 시 저장한 모델, 스케일러, 선택된 피처, LabelEncoder 사전을 불러온다.
    best_lgb_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    train_features = joblib.load(features_path)
    label_encoders = joblib.load(labelenc_path)

    # 시술_당시_나이 컬럼 전처리: 나이 변환 및 결측치 중간값 대체
    if "시술_당시_나이" in test_df.columns:
        test_df["시술_당시_나이"] = test_df["시술_당시_나이"].apply(parse_age_surgery)
        median_val = test_df["시술_당시_나이"].median()
        test_df["시술_당시_나이"].fillna(median_val, inplace=True)

    # 시술, 임신, 출산 횟수 컬럼 전처리: 문자열 형식의 회수 데이터를 숫자로 변환한다.
    cycle_cols = [
        "총_시술_횟수", "클리닉_내_총_시술_횟수", "IVF_시술_횟수", "DI_시술_횟수",
        "총_임신_횟수", "IVF_임신_횟수", "DI_임신_횟수", "총_출산_횟수", "IVF_출산_횟수", "DI_출산_횟수"
    ]
    for c in cycle_cols:
        if c in test_df.columns:
            test_df[c] = test_df[c].apply(parse_cycle_count)

    # 불리언 컬럼 전처리: 문자열 또는 숫자 값을 0 또는 1로 변환한다.
    bool_cols = [
        "단일_배아_이식_여부", "착상_전_유전_검사_사용_여부", "착상_전_유전_진단_사용_여부"
    ]
    for c in bool_cols:
        if c in test_df.columns:
            test_df[c] = test_df[c].apply(parse_bool)

    # 범주형 컬럼 전처리: 결측치는 "unknown"으로 채우고 학습 시 저장한 LabelEncoder를 적용한다.
    cat_cols = [
        "시술_유형", "특정_시술_유형", "배란_유도_유형", "난자_출처", "정자_출처", "배아_생성_주요_이유", "시술_시기_코드"
    ]
    for c in cat_cols:
        if c in test_df.columns:
            test_df[c].fillna("unknown", inplace=True)
            if c in label_encoders:
                le_enc = label_encoders[c]
                test_df[c] = safe_label_transform(test_df[c], le_enc, unknown_str="unknown")

    # 경과일 관련 컬럼 전처리: 숫자형으로 변환하고 결측치는 -1로 채운다.
    day_cols = [
        "난자_채취_경과일", "난자_해동_경과일", "난자_혼합_경과일", "배아_이식_경과일", "배아_해동_경과일"
    ]
    for c in day_cols:
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
            test_df[c].fillna(-1, inplace=True)

    # 기타 결측치는 -1로 처리한다.
    test_df.fillna(-1, inplace=True)

    # 파생변수 생성: 총 생성 배아 수와 이식된 배아 수가 있을 경우 배아 이식 비율을 계산한다.
    if "총_생성_배아_수" in test_df.columns and "이식된_배아_수" in test_df.columns:
        test_df["배아_이식_비율"] = np.where(
            test_df["총_생성_배아_수"] == 0,
            0,
            test_df["이식된_배아_수"] / test_df["총_생성_배아_수"]
        )

    # 학습 시 선택한 피처에 맞추어 컬럼 순서를 재정렬하며, 누락된 컬럼은 -1로 채운다.
    X_test = test_df.reindex(columns=train_features, fill_value=-1).copy()
    X_test = X_test.astype(np.float32)
    X_test_scaled = scaler.transform(X_test)

    # 최종 예측을 수행하고 확률 값을 추출한다.
    preds = best_lgb_model.predict_proba(X_test_scaled)[:, 1]

    # 제출 파일을 생성하여 ID와 예측 확률을 저장한다.
    submission = pd.DataFrame({
        "ID": test_ids,
        "probability": preds
    })
    submission.to_csv(submission_path, index=False)
    print(f"✅ {submission_path} (DART Inference) 생성 완료!")

if __name__ == "__main__":
    inference()
