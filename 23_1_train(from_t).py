# train_lightgbm_optuna.py

import pandas as pd
import numpy as np
import joblib
import optuna
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

###################################################
# 0) 공통 전처리에 쓰일 함수들
###################################################
# 나이 변환 (시술 당시 나이)
age_map_surgery = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38,
    "만38-40세": 39,
    "만40-42세": 41,
    "만43-44세": 43,
    "만45-50세": 47,
    "알_수_없음": np.nan
}

def parse_age_surgery(x):
    return age_map_surgery.get(str(x).strip(), np.nan)

# "0회","1회" ~ "6회 이상" 형태 횟수를 숫자로 변환
def parse_cycle_count(x):
    if isinstance(x, str):
        if "이상" in x:
            return 6
        elif "회" in x:
            return pd.to_numeric(x.replace("회", ""), errors="coerce")
    return pd.to_numeric(x, errors="coerce")

# True/False, 0/1, 'False'/'True' 같은 불리언 컬럼 통일
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

###################################################
# 1) 학습(Training) 실행 부분
###################################################
if __name__ == "__main__":
    # (A) 데이터 로드
    train = pd.read_csv("./0_rawdataset/train.csv")

    # 공백 컬럼명 → 밑줄 치환 (예: "시술 시기 코드" → "시술_시기_코드")
    train.columns = [c.replace(" ", "_") for c in train.columns]

    # ID 제거 (있다면)
    if "ID" in train.columns:
        train.drop(columns=["ID"], inplace=True)

    # 타겟(임신_성공_여부) 지정
    target_col = "임신_성공_여부"
    if train[target_col].dtype == object:
        train[target_col] = train[target_col].apply(lambda v: 1 if str(v).strip() == "1" else 0)

    # 라벨 인코더들을 저장할 dict
    label_encoders = {}

    ############################################
    # (B) 전처리
    ############################################

    # 1) 시술_시기_코드 (범주형)
    code_col = "시술_시기_코드"
    if code_col in train.columns:
        train[code_col].fillna("unknown", inplace=True)
        unique_vals = train[code_col].astype(str).unique().tolist()
        if "unknown" not in unique_vals:
            unique_vals.append("unknown")
        le_code = LabelEncoder()
        le_code.fit(unique_vals)
        train[code_col] = le_code.transform(train[code_col].astype(str))
        label_encoders[code_col] = le_code

    # 2) 시술_당시_나이 (숫자 변환)
    if "시술_당시_나이" in train.columns:
        train["시술_당시_나이"] = train["시술_당시_나이"].apply(parse_age_surgery)

    # 3) 임신_시도_또는_마지막_임신_경과_연수 (숫자 변환)
    gap_col = "임신_시도_또는_마지막_임신_경과_연수"
    if gap_col in train.columns:
        train[gap_col] = pd.to_numeric(train[gap_col], errors="coerce")

    # 4) 범주형 & 불리언
    #    배란_자극_여부, 배란_유도_유형, 단일_배아_이식_여부, 착상_전_유전_검사_사용_여부 등
    cat_cols = []
    bool_cols = []

    # 시술_유형, 특정_시술_유형, 배란_유도_유형 → 범주형
    if "시술_유형" in train.columns:
        cat_cols.append("시술_유형")
    if "특정_시술_유형" in train.columns:
        cat_cols.append("특정_시술_유형")
    if "배란_유도_유형" in train.columns:
        cat_cols.append("배란_유도_유형")

    # 배란_자극_여부 → bool이라고 가정
    if "배란_자극_여부" in train.columns:
        bool_cols.append("배란_자극_여부")

    # 단일_배아_이식_여부, 착상_전_유전_검사_사용_여부, 착상_전_유전_진단_사용_여부
    if "단일_배아_이식_여부" in train.columns:
        bool_cols.append("단일_배아_이식_여부")
    if "착상_전_유전_검사_사용_여부" in train.columns:
        bool_cols.append("착상_전_유전_검사_사용_여부")
    if "착상_전_유전_진단_사용_여부" in train.columns:
        bool_cols.append("착상_전_유전_진단_사용_여부")

    # 기증_배아_사용_여부, 대리모_여부, PGD_시술_여부, PGS_시술_여부
    if "기증_배아_사용_여부" in train.columns:
        bool_cols.append("기증_배아_사용_여부")
    if "대리모_여부" in train.columns:
        bool_cols.append("대리모_여부")
    if "PGD_시술_여부" in train.columns:
        bool_cols.append("PGD_시술_여부")
    if "PGS_시술_여부" in train.columns:
        bool_cols.append("PGS_시술_여부")

    # 5) 불임 원인 관련 변수 (남성, 여성, 부부, 불명확 등)도 bool 처리
    for c in [
        "남성_주_불임_원인","남성_부_불임_원인","여성_주_불임_원인","여성_부_불임_원인",
        "부부_주_불임_원인","부부_부_불임_원인","불명확_불임_원인","불임_원인_-_난관_질환",
        "불임_원인_-_남성_요인","불임_원인_-_배란_장애","불임_원인_-_여성_요인",
        "불임_원인_-_자궁경부_문제","불임_원인_-_자궁내막증","불임_원인_-_정자_농도",
        "불임_원인_-_정자_면역학적_요인","불임_원인_-_정자_운동성","불임_원인_-_정자_형태"
    ]:
        if c in train.columns:
            bool_cols.append(c)

    # 6) 배아_생성_주요_이유 (범주형)
    if "배아_생성_주요_이유" in train.columns:
        cat_cols.append("배아_생성_주요_이유")

    # 7) 시술, 임신, 출산 횟수 변수 (0회~6회 이상)
    cycle_cols = [
        "총_시술_횟수","클리닉_내_총_시술_횟수","IVF_시술_횟수","DI_시술_횟수",
        "총_임신_횟수","IVF_임신_횟수","DI_임신_횟수","총_출산_횟수","IVF_출산_횟수","DI_출산_횟수"
    ]
    for c in cycle_cols:
        if c in train.columns:
            train[c] = train[c].apply(parse_cycle_count)

    # 8) 난자/정자 출처 (범주형)
    if "난자_출처" in train.columns:
        cat_cols.append("난자_출처")
    if "정자_출처" in train.columns:
        cat_cols.append("정자_출처")

    # 9) 난자 기증자 나이, 정자 기증자 나이 (숫자 변환 가능 시)
    #    (사용자가 필요하면 parse 함수 별도 정의)
    #    여기서는 예시로 object -> float 변환만 처리
    for c in ["난자_기증자_나이","정자_기증자_나이"]:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")

    # 10) 난자 채취/해동/혼합 경과일, 배아 이식/해동 경과일 (숫자형)
    day_cols = ["난자_채취_경과일","난자_해동_경과일","난자_혼합_경과일","배아_이식_경과일","배아_해동_경과일"]
    for c in day_cols:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
            train[c].fillna(-1, inplace=True)

    # (B-1) bool_cols → 0/1 변환
    for c in bool_cols:
        if c in train.columns:
            train[c] = train[c].apply(parse_bool)

    # (B-2) cat_cols → Label Encoding (결측 -> 'unknown' 포함)
    #      학습 시점에 'unknown'을 명시적으로 추가해두면 추론 시 unseen label도 처리 가능
    for c in cat_cols:
        if c in train.columns:
            train[c].fillna("unknown", inplace=True)
            unique_vals = train[c].astype(str).unique().tolist()
            if "unknown" not in unique_vals:
                unique_vals.append("unknown")
            le_cat = LabelEncoder()
            le_cat.fit(unique_vals)
            train[c] = le_cat.transform(train[c].astype(str))
            label_encoders[c] = le_cat

    # (B-3) 결측치 처리: 중요한 변수 중간값, 나머지는 -1
    imp_cols = ["시술_당시_나이","난자_기증자_나이","정자_기증자_나이","임신_시도_또는_마지막_임신_경과_연수"]
    for c in imp_cols:
        if c in train.columns:
            median_val = train[c].median()
            train[c].fillna(median_val, inplace=True)

    train.fillna(-1, inplace=True)

    # (B-4) 나이 범위 기반 이상치 제거 (예: 15~60세)
    if "시술_당시_나이" in train.columns:
        train = train[(train["시술_당시_나이"] >= 15) & (train["시술_당시_나이"] <= 60)]

    # (B-5) 배아 관련 파생 변수
    if "총_생성_배아_수" in train.columns and "이식된_배아_수" in train.columns:
        train["배아_이식_비율"] = np.where(
            train["총_생성_배아_수"] == 0,
            0,
            train["이식된_배아_수"] / train["총_생성_배아_수"]
        )

    # 예시로 몇 가지 더:
    if "시술_당시_나이" in train.columns and "이식된_배아_수" in train.columns:
        train["나이x배아수"] = train["시술_당시_나이"] * train["이식된_배아_수"]
    if "시술_당시_나이" in train.columns and "총_생성_배아_수" in train.columns:
        train["나이x총배아수"] = train["시술_당시_나이"] * train["총_생성_배아_수"]

    # (C) X, y 분리
    X = train.drop(columns=[target_col])
    y = train[target_col]

    # (C-1) Feature Selection (LightGBM 기반)
    selector_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        boosting_type="gbdt"
    )
    selector_model.fit(X, y)

    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    selected_features = X.columns[selector.get_support()]
    X_selected = X[selected_features].copy()

    # 무한대/결측 제거
    X_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_selected.dropna(inplace=True)
    y = y.loc[X_selected.index]

    # 타입 변환 (float32 등)
    X_selected = X_selected.astype(np.float32)
    y = y.astype(int)

    # (C-2) SMOTE (오버샘플링)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    # (C-3) StandardScaler
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    ###################################################
    # 2) Optuna로 LightGBM 하이퍼파라미터 탐색
    ###################################################
    # 초기 파라미터 (±20% 탐색)
    initial_params = {
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 20,
        "min_split_gain": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "bagging_freq": 5,
        "max_bin": 255,
        "extra_trees": False,
        "device": "gpu",
        "gpu_use_dp" : True
    }

    # 평가 함수 (10-Fold CV → AUC)
    def evaluate_params(params):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        auc_scores = []
        for train_idx, val_idx in cv.split(X_resampled, y_resampled):
            X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
            y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
            model = lgb.LGBMClassifier(
                boosting_type="gbdt",
                objective="binary",
                metric="auc",
                random_state=42,
                **params
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            preds = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, preds))
        return np.mean(auc_scores)

    # ±20% 범위
    def apply_variation(value, ratio=0.2):
        lower = value * (1 - ratio)
        upper = value * (1 + ratio)
        return lower, upper

    # Optuna objective
    best_auc_global = 0.0

    def objective(trial):
        global best_auc_global
        params = {}

        # 1) learning_rate
        lr_lower, lr_upper = apply_variation(initial_params["learning_rate"])
        params["learning_rate"] = trial.suggest_float("learning_rate", max(0.0001, lr_lower), lr_upper, log=True)

        # 2) n_estimators
        n_est_lower = max(100, int(initial_params["n_estimators"] * 0.8))
        n_est_upper = int(initial_params["n_estimators"] * 1.2)
        params["n_estimators"] = trial.suggest_int("n_estimators", n_est_lower, n_est_upper)

        # 3) num_leaves
        leaves_lower = max(10, int(initial_params["num_leaves"] * 0.8))
        leaves_upper = int(initial_params["num_leaves"] * 1.2)
        params["num_leaves"] = trial.suggest_int("num_leaves", leaves_lower, leaves_upper)

        # 4) max_depth
        depth_lower = max(3, int(initial_params["max_depth"] * 0.8))
        depth_upper = int(initial_params["max_depth"] * 1.2)
        params["max_depth"] = trial.suggest_int("max_depth", depth_lower, depth_upper)

        # 5) min_child_samples
        child_lower = max(5, int(initial_params["min_child_samples"] * 0.8))
        child_upper = int(initial_params["min_child_samples"] * 1.2)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", child_lower, child_upper)

        # 6) min_split_gain
        gain_lower, gain_upper = apply_variation(initial_params["min_split_gain"])
        params["min_split_gain"] = trial.suggest_float("min_split_gain", gain_lower, gain_upper)

        # 7) reg_alpha
        alpha_lower, alpha_upper = apply_variation(initial_params["reg_alpha"])
        params["reg_alpha"] = trial.suggest_float("reg_alpha", alpha_lower, alpha_upper)

        # 8) reg_lambda
        lambda_lower, lambda_upper = apply_variation(initial_params["reg_lambda"])
        params["reg_lambda"] = trial.suggest_float("reg_lambda", lambda_lower, lambda_upper)

        # 9) colsample_bytree
        col_lower, col_upper = apply_variation(initial_params["colsample_bytree"])
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", max(0.1, col_lower), min(1.0, col_upper))

        # 10) subsample
        subs_lower, subs_upper = apply_variation(initial_params["subsample"])
        params["subsample"] = trial.suggest_float("subsample", max(0.1, subs_lower), min(1.0, subs_upper))

        # 11) bagging_freq
        freq_lower = max(1, int(initial_params["bagging_freq"] - 2))
        freq_upper = initial_params["bagging_freq"] + 2
        params["bagging_freq"] = trial.suggest_int("bagging_freq", freq_lower, freq_upper)

        # 12) max_bin
        bin_lower = max(50, int(initial_params["max_bin"] * 0.8))
        bin_upper = int(initial_params["max_bin"] * 1.2)
        params["max_bin"] = trial.suggest_int("max_bin", bin_lower, bin_upper)

        # 13) extra_trees
        params["extra_trees"] = trial.suggest_categorical("extra_trees", [False, True])

        # 평가
        score = evaluate_params(params)

        # 최고 성능 갱신 시 JSON 파일에 저장
        if score > best_auc_global:
            best_auc_global = score
            with open("best_lgb_params.json", "w") as f:
                json.dump(params, f, indent=4)
            print(f"\n🎯 새로운 최고 AUC: {score:.6f}")
            print(f"🔍 최적 파라미터: {params}")

        return score

    # Optuna 실행
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    sampler = TPESampler(multivariate=True)
    pruner = HyperbandPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=1000)

    print("✅ Best trial:", study.best_trial)
    best_params = study.best_trial.params
    best_params.update({
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "random_state": 42
    })
    print("\n✅ 최적 파라미터:")
    print(json.dumps(best_params, indent=4))

    ###################################################
    # 3) 최종 모델 학습 및 저장
    ###################################################
    best_lgb_model = lgb.LGBMClassifier(**best_params)
    best_lgb_model.fit(
        X_resampled, y_resampled,
        callbacks=[lgb.log_evaluation(-1)]
    )

    # 학습 데이터 성능 확인
    train_preds = best_lgb_model.predict_proba(X_resampled)[:, 1]
    train_pred_class = (train_preds >= 0.5).astype(int)
    train_auc = roc_auc_score(y_resampled, train_preds)
    train_acc = accuracy_score(y_resampled, train_pred_class)
    print("\n🔥 Final Model Evaluation (Train Data)")
    print(f"🔥 Train Accuracy: {train_acc:.4f}")
    print(f"🔥 Train ROC-AUC: {train_auc:.4f}")

    # 객체 저장 (모델, 스케일러, 피처 목록, 라벨 인코더)
    joblib.dump(best_lgb_model, "best_lgb_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X_selected.columns), "train_features.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")  # 추론 시 사용
    print("✅ 모델, 스케일러, 피처 목록, 라벨 인코더 사전 저장 완료!")
