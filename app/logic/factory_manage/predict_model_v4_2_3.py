import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_target_items(df_raw, top_n=5):
    return df_raw["品名"].value_counts().head(top_n).index.tolist()


def generate_reserve_features(df_reserve, top_k_clients=10):
    df_reserve = df_reserve.copy()
    df_reserve["予約日"] = pd.to_datetime(df_reserve["予約日"])
    top_clients = df_reserve["予約得意先名"].value_counts().head(top_k_clients).index
    df_reserve["上位得意先フラグ"] = (
        df_reserve["予約得意先名"].isin(top_clients).astype(int)
    )

    df_feat = df_reserve.groupby("予約日").agg(
        予約件数=("予約得意先名", "count"),
        固定客予約数=("固定客", lambda x: x.sum()),
        非固定客予約数=("固定客", lambda x: (~x).sum()),
        上位得意先予約数=("上位得意先フラグ", "sum"),
    )
    df_feat["固定客比率"] = df_feat["固定客予約数"] / df_feat["予約件数"]
    return df_feat.fillna(0)


def generate_weight_features(past_raw, target_items, holidays):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    for item in target_items:
        if item not in df_pivot.columns:
            df_pivot[item] = 0
    df_pivot = df_pivot.sort_index()
    df_pivot["合計"] = df_pivot[target_items].sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    for item in target_items:
        df_feat[f"{item}_前日値"] = df_pivot[item].shift(1)
        df_feat[f"{item}_前週平均"] = df_pivot[item].shift(1).rolling(7).mean()
    df_feat["合計_前日値"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["合計_前週平均"] = df_pivot["合計"].shift(1).rolling(7).mean()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = (
        daily_avg.shift(1).rolling(60, min_periods=10).median()
    )

    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
    df_feat["祝日前フラグ"] = df_feat.index.map(
        lambda d: (d + pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    df_feat["祝日後フラグ"] = df_feat.index.map(
        lambda d: (d - pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)

    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]
    return df_feat, df_pivot


def train_and_predict_stage1(
    df_feat_today,
    df_past_feat,
    df_past_pivot,
    base_models,
    meta_model_proto,
    feature_list,
    target_items,
    stage1_eval,
    df_pivot,
):
    # print("▶️ train_and_predict_stage1 開始")
    # print("📌 df_feat_today index:", df_feat_today.index)
    # print("📌 学習用特徴量サイズ:", df_past_feat.shape)
    # print("📌 学習用pivotサイズ:", df_past_pivot.shape)

    results = {}
    X_train = df_past_feat[feature_list]
    for item in target_items:
        y_train = df_past_pivot[item]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        selector = VarianceThreshold(1e-4)
        X_train_filtered = selector.fit_transform(X_train_scaled)

        trained_models = []
        for name, model in base_models:
            print(f"🛠 モデル訓練中: {name} for {item}")
            m = clone(model)
            m.fit(X_train_filtered, y_train)
            trained_models.append(m)

        train_meta = np.column_stack(
            [m.predict(X_train_filtered) for m in trained_models]
        )
        meta_model = clone(meta_model_proto)
        meta_model.fit(train_meta, y_train)

        X_target = df_feat_today[feature_list]
        X_target_scaled = scaler.transform(X_target)
        X_target_filtered = selector.transform(X_target_scaled)

        meta_input = np.column_stack(
            [m.predict(X_target_filtered) for m in trained_models]
        )
        pred = meta_model.predict(meta_input)[0]
        results[f"{item}_予測"] = pred

        true_val = df_pivot.loc[df_feat_today.index[0], item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

        print(f"✅ {item} 予測: {pred:.1f}kg / 正解: {true_val:.1f}kg")

        # 特徴量重要度の出力
        # selected_columns = np.array(feature_list)[selector.get_support()]
        # if hasattr(trained_models[0], "coef_"):
        #     elastic_coef = trained_models[0].coef_
        #     print(f"🔍 ElasticNet 係数 ({item}):")
        #     for name, val in zip(selected_columns, elastic_coef, strict=True):
        #         print(f"   {name:<25} : {val:.4f}")

        #     # 可視化（上位15個）
        #     # plot_feature_importances(
        #     #     selected_columns,
        #     #     elastic_coef,
        #     #     title=f"ElasticNet Feature Importance ({item})",
        #     # )

        # if hasattr(trained_models[1], "feature_importances_"):
        #     rf_importances = trained_models[1].feature_importances_
        #     sorted_rf = sorted(
        #         zip(selected_columns, rf_importances, strict=True), key=lambda x: -x[1]
        #     )
        #     print(f"🔍 RandomForest 重要度 ({item}):")
        #     for name, val in sorted_rf[:10]:
        #         print(f"   {name:<25} : {val:.4f}")

        # 可視化（上位15個）
        # plot_feature_importances(
        #     selected_columns,
        #     rf_importances,
        #     title=f"RandomForest Feature Importance ({item})",
        # )

    return results


def train_and_predict_stage2(
    all_stage1_rows, stage1_results, df_feat_today, target_items, show_plot=False
):
    # print("▶️ train_and_predict_stage2 開始")
    df_hist = pd.DataFrame(all_stage1_rows[:-1])
    # print("📌 df_hist サイズ:", df_hist.shape)

    X_train = df_hist.drop(columns=["合計"])
    y_train = df_hist["合計"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    selector = VarianceThreshold(1e-4)
    X_train_filtered = selector.fit_transform(X_train_scaled)

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt.fit(X_train_filtered, y_train)

    X_target = {
        f"{item}_予測": [stage1_results[f"{item}_予測"]] for item in target_items
    }
    for col in df_feat_today.columns:
        if col not in X_target:
            X_target[col] = df_feat_today.iloc[0][col]
    X_target = pd.DataFrame(X_target)

    X_target_scaled = scaler.transform(X_target)
    X_target_filtered = selector.transform(X_target_scaled)

    total_pred = gbdt.predict(X_target_filtered)[0]
    print(f"✅ 合計予測: {total_pred:.1f}kg")

    # GBDT特徴量重要度
    X_cols = X_train.columns[selector.get_support()]
    gbdt_importance = gbdt.feature_importances_
    sorted_gbdt = sorted(zip(X_cols, gbdt_importance, strict=True), key=lambda x: -x[1])
    # print("🔍 GBDT 特徴量重要度 (上位10):")
    # for name, val in sorted_gbdt[:10]:
    #     print(f"   {name:<25} : {val:.4f}")

    # 可視化（上位15個）
    # if show_plot:
    #     plot_feature_importances(
    #         X_cols,
    #         gbdt_importance,
    #         title="Stage2 GBDT Feature Importance",
    #     )

    return total_pred


def evaluate_stage1(stage1_eval, target_items):
    print("\n===== ステージ1評価結果 =====")
    for item in target_items:
        y_true = np.array(stage1_eval[item]["y_true"])
        y_pred = np.array(stage1_eval[item]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{item}: R² = {r2:.3f}, MAE = {mae:,.0f}kg")


def full_walkforward(df_raw, holidays, df_reserve, top_n=5):
    print("▶️ full_walkforward 開始")
    print("📌 入力データ件数:", len(df_raw))

    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)
    df_feat, df_pivot = generate_weight_features(df_raw, target_items, holidays)

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    feature_list = [
        *[f"{item}_前日値" for item in target_items],
        *[f"{item}_前週平均" for item in target_items],
        "合計_前日値",
        "合計_3日平均",
        "合計_3日合計",
        "合計_前週平均",
        "曜日",
        "週番号",
        "1台あたり重量_過去中央値",
        "祝日フラグ",
        "祝日前フラグ",
        "祝日後フラグ",
        "予約件数",
        "固定客予約数",
        "非固定客予約数",
        "固定客比率",
        "上位得意先予約数",
    ]

    all_actual, all_pred, all_stage1_rows = [], [], []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index

    for i, target_date in enumerate(dates):
        if i < 30:
            continue

        df_past_feat = df_feat[df_feat.index < target_date].tail(600)
        df_past_pivot = df_pivot.loc[df_past_feat.index]

        df_reserve_filtered = df_reserve[
            pd.to_datetime(df_reserve["予約日"]) <= target_date
        ]
        df_reserve_feat_all = generate_reserve_features(df_reserve_filtered)

        df_past_feat = df_past_feat.merge(
            df_reserve_feat_all, left_index=True, right_index=True, how="left"
        ).fillna(0)

        df_feat_today = df_feat.loc[[target_date]].copy()
        df_feat_today = df_feat_today.merge(
            df_reserve_feat_all, left_index=True, right_index=True, how="left"
        ).fillna(0)

        print(f"\n=== {target_date.strftime('%Y-%m-%d')} を予測中 ===")
        stage1_result = train_and_predict_stage1(
            df_feat_today,
            df_past_feat,
            df_past_pivot,
            base_models,
            meta_model_proto,
            feature_list,
            target_items,
            stage1_eval,
            df_pivot,  # ✅ 追加
        )

        row = {f"{item}_予測": stage1_result[f"{item}_予測"] for item in target_items}
        for col in df_feat_today.columns:
            if col not in row:
                row[col] = df_feat_today.iloc[0][col]
        row["合計"] = df_pivot.loc[target_date, "合計"]
        all_stage1_rows.append(row)

        if len(all_stage1_rows) > 30:
            total_pred = train_and_predict_stage2(
                all_stage1_rows,
                stage1_result,
                df_feat_today,
                target_items,
                show_plot=(i % 30 == 0),  # 30日に1回だけプロット
            )

    print("\n===== ステージ2評価結果 (合計) =====")
    if all_actual:
        print(
            f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
        )
    else:
        print("評価できるデータが不足しています")

    evaluate_stage1(stage1_eval, target_items)
    return all_actual, all_pred


# 重要度可視化
def plot_feature_importances(names, importances, title="Feature Importance", top_k=15):
    sorted_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), np.array(importances)[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), np.array(names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def save_feature_importance_to_csv(names, importances, filename):
    df = pd.DataFrame(
        {
            "feature": names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    df.to_csv(filename, index=False)


import os


def save_feature_importances_plot(
    names,
    importances,
    title="Feature Importance",
    top_k=15,
    save_path="feature_importance.pdf",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), np.array(importances)[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), np.array(names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"📄 PDF保存完了: {save_path}")


# 相関係数が高いペアを抽出（しきい値 0.9 以上）
def get_high_corr_pairs(corr_matrix, threshold=0.9):
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                pairs.append((cols[i], cols[j], corr_val))
    return sorted(pairs, key=lambda x: -abs(x[2]))
