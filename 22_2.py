# train_lightgbm_optuna.py

import pandas as pd
import numpy as np
import joblib
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

import warnings
warnings.filterwarnings("ignore")

########################################
# 0) 공통 전처리에 쓰일 함수 및 매핑
########################################
age_map_surgery = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38,
    "만38-40세": 39,
    "만40-42세": 41,
    "만43-44세": 43,
    "만45-50세": 47,
    "알_수_없음": np.nan  # 공백 치환
}
age_map_donor = {
    "만20세_이하": 20,
    "만21-25세": 23,
    "만26-30세": 28,
    "만31-35세": 33,
    "만36-40세": 38,
    "만41-45세": 43,
    "알_수_없음": np.nan
}

def parse_age_surgery(x):
    return age_map_surgery.get(str(x).strip(), np.nan)

def parse_age_donor(x):
    return age_map_donor.get(str(x).strip(), np.nan)

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

################################
# 1) 학습(Training) 실행 부분
################################
if __name__ == "__main__":
    # (A) 데이터 로드
    train = pd.read_csv("./0_rawdataset/train.csv")

    # 공백 컬럼명 → 밑줄 치환
    train.columns = [c.replace(" ", "_") for c in train.columns]

    # ID 제거(있다면)
    if "ID" in train.columns:
        train.drop(columns=["ID"], inplace=True)

    # 타겟 지정
    target_col = "임신_성공_여부"
    if train[target_col].dtype == object:
        train[target_col] = train[target_col].apply(lambda v: 1 if str(v).strip() == "1" else 0)

    # (B) 전처리
    label_encoders = {}

    # 시술 시기 코드
    code_col = "시술_시기_코드"
    if code_col in train.columns:
        train[code_col].fillna("missing", inplace=True)
        le_code = LabelEncoder()
        train[code_col] = le_code.fit_transform(train[code_col].astype(str))
        label_encoders[code_col] = le_code

    # 시술 당시 나이
    if "시술_당시_나이" in train.columns:
        train["시술_당시_나이"] = train["시술_당시_나이"].apply(parse_age_surgery)

    # 사이클(0회~6회이상)
    cycle_cols = [
        "총_시술_횟수","클리닉_내_총_시술_횟수","IVF_시술_횟수","DI_시술_횟수",
        "총_임신_횟수","IVF_임신_횟수","DI_임신_횟수","총_출산_횟수","IVF_출산_횟수","DI_출산_횟수"
    ]
    for c in cycle_cols:
        if c in train.columns:
            train[c] = train[c].apply(parse_cycle_count)

    # bool 컬럼
    bool_cols = [
        "단일_배아_이식_여부","착상_전_유전_검사_사용_여부","착상_전_유전_진단_사용_여부",
        "동결_배아_사용_여부","신선_배아_사용_여부","기증_배아_사용_여부","대리모_여부",
        "PGD_시술_여부","PGS_시술_여부",
        "남성_주_불임_원인","남성_부_불임_원인","여성_주_불임_원인","여성_부_불임_원인","부부_주_불임_원인",
        "부부_부_불임_원인","불명확_불임_원인","불임_원인_-_난관_질환","불임_원인_-_남성_요인",
        "불임_원인_-_배란_장애","불임_원인_-_여성_요인","불임_원인_-_자궁경부_문제","불임_원인_-_자궁내막증",
        "불임_원인_-_정자_농도","불임_원인_-_정자_면역학적_요인","불임_원인_-_정자_운동성","불임_원인_-_정자_형태"
    ]
    for c in bool_cols:
        if c in train.columns:
            train[c] = train[c].apply(parse_bool)

    # 범주형
    cat_cols = [
        "시술_유형","특정_시술_유형","배란_유도_유형","배아_생성_주요_이유","난자_출처","정자_출처"
    ]
    for c in cat_cols:
        if c in train.columns:
            train[c].fillna("missing", inplace=True)
            le_tmp = LabelEncoder()
            train[c] = le_tmp.fit_transform(train[c].astype(str))
            label_encoders[c] = le_tmp

    # 난자 기증자 나이, 정자 기증자 나이
    if "난자_기증자_나이" in train.columns:
        train["난자_기증자_나이"] = train["난자_기증자_나이"].apply(parse_age_donor)
    if "정자_기증자_나이" in train.columns:
        train["정자_기증자_나이"] = train["정자_기증자_나이"].apply(parse_age_donor)

    # ▼▼▼ 경과일 컬럼 추가 보완 ▼▼▼
    day_cols = [
        "난자_채취_경과일","난자_해동_경과일","난자_혼합_경과일","배아_이식_경과일","배아_해동_경과일"
    ]
    for c in day_cols:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
            # 추가로 결측치 처리 or 이상치 제거
            train[c].fillna(-1, inplace=True)
            # 필요하다면 np.clip 등도 가능

    # 결측치 처리
    if "시술_당시_나이" in train.columns:
        median_val = train["시술_당시_나이"].median()
        train["시술_당시_나이"].fillna(median_val, inplace=True)
    train.fillna(-1, inplace=True)

    # 파생 변수
    if "총_생성_배아_수" in train.columns and "이식된_배아_수" in train.columns:
        train["배아_이식_비율"] = np.where(
            train["총_생성_배아_수"] == 0,
            0,
            train["이식된_배아_수"] / train["총_생성_배아_수"]
        )

    # 이상치 제거 (나이 15~60)
    if "시술_당시_나이" in train.columns:
        train = train[(train["시술_당시_나이"]>=15) & (train["시술_당시_나이"]<=60)]

    # (C) X, y 분리 & Feature Selection
    X = train.drop(columns=[target_col])
    y = train[target_col]

    selector_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    selector_model.fit(X, y)

    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    selected_features = X.columns[selector.get_support()]
    X_selected = X[selected_features].copy()

    X_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_selected.dropna(inplace=True)
    y = y.loc[X_selected.index]

    X_selected = X_selected.astype(np.float32)
    y = y.astype(int)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    # 스케일러
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    # (D) Optuna로 LGBM 하이퍼파라미터 탐색
    import optuna
    from sklearn.model_selection import StratifiedKFold
    def objective(trial):
        param = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "max_bin": trial.suggest_int("max_bin", 63, 255),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True])
        }

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        auc_scores = []
        for train_idx, val_idx in cv.split(X_resampled, y_resampled):
            X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
            y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
            model = lgb.LGBMClassifier(**param)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(-1)]
            )
            preds = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, preds))
        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("✅ Best trial:", study.best_trial)
    best_params = study.best_trial.params
    print("✅ 최적 파라미터:", best_params)

    # 최종 모델 학습
    best_params.update({
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "random_state": 42
    })
    best_lgb_model = lgb.LGBMClassifier(**best_params)
    best_lgb_model.fit(X_resampled, y_resampled, callbacks=[lgb.log_evaluation(-1)])

    # 학습 데이터 성능
    from sklearn.metrics import accuracy_score
    train_preds = best_lgb_model.predict_proba(X_resampled)[:,1]
    train_pred_class = (train_preds >= 0.5).astype(int)
    from sklearn.metrics import roc_auc_score
    train_acc = accuracy_score(y_resampled, train_pred_class)
    train_auc = roc_auc_score(y_resampled, train_preds)
    print("🔥 Train Accuracy:", f"{train_acc:.4f}")
    print("🔥 Train ROC-AUC:", f"{train_auc:.4f}")

    # (E) 저장: 모델, 스케일러, 피처 목록, 라벨 인코더 dict
    joblib.dump(best_lgb_model, "22-best_lgb_model.pkl")
    joblib.dump(scaler, "22-scaler.pkl")
    joblib.dump(list(selected_features), "22-train_features.pkl")
    joblib.dump(label_encoders, "22-label_encoders.pkl")
    print("✅ 모든 객체 저장 완료 (경과일 처리 포함)")
