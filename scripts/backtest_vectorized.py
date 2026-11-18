"""Backtest vetorizado v0.2 — calcula features com window functions do pandas.

Muito mais rápido que o FeatureStore linha-a-linha. Avalia semanas 4-10.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yardline_intelligence.config import CURRENT_SEASON
from yardline_intelligence.data_foundation.dataset import build_player_dataset
from yardline_intelligence.models.direct_model import DirectPointsModel, _spearman
from yardline_intelligence.temporal_feature_store.store import moving_average_baseline


def build_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de feature ao dataset usando groupby + shift (vetorizado).

    Todas as features usam apenas semanas anteriores (shift >= 1), sem data leakage.
    """
    df = df.sort_values(["sleeper_id", "week"]).copy()
    g = df.groupby("sleeper_id")

    # Targets: valor anterior e médias móveis (excluindo a semana atual).
    df["targets_prev"] = g["targets"].shift(1)
    df["targets_mean_3"] = g["targets"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["targets_mean_5"] = g["targets"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["targets_mean_8"] = g["targets"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    df["targets_trend"] = (
        g["targets"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
        - g["targets"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean().shift(2))
    )

    # Snap share.
    df["snap_share_prev"] = g["snap_share"].shift(1)
    df["snap_share_mean_3"] = g["snap_share"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["snap_share_mean_5"] = g["snap_share"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["snap_share_trend"] = (
        g["snap_share"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
        - g["snap_share"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean().shift(2))
    )

    # Eficiência: EPA por alvo e air yards por alvo (usando valores acumulados anteriores).
    df["epa_sum_prev"] = g["epa_sum"].shift(1)
    df["targets_cum_prev"] = g["targets"].transform(lambda s: s.shift(1).cumsum())
    df["epa_sum_cum_prev"] = g["epa_sum"].transform(lambda s: s.shift(1).cumsum())
    df["epa_per_target_mean_3"] = (
        g.apply(lambda d: (d["epa_sum"].shift(1).rolling(3).sum() / d["targets"].shift(1).rolling(3).sum())
                .replace([np.inf, -np.inf], np.nan), include_groups=False)
        .reset_index(level=0, drop=True)
    )
    df["air_yards_per_target_mean_3"] = (
        g.apply(lambda d: (d["air_yards"].shift(1).rolling(3).sum() / d["targets"].shift(1).rolling(3).sum())
                .replace([np.inf, -np.inf], np.nan), include_groups=False)
        .reset_index(level=0, drop=True)
    )

    # Recebendo/corrida.
    df["receiving_yards_mean_3"] = g["receiving_yards"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["rush_yards_mean_3"] = g["rush_yards"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["air_yards_mean_3"] = g["air_yards"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Consistência.
    df["target_consistency"] = g["targets"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())

    # Jogos/disponibilidade.
    df["games_played_prev_n"] = g["games_played"].transform(lambda s: s.shift(1).cumsum())
    df["weeks_active"] = g.cumcount()

    return df


def main():
    target_season = CURRENT_SEASON
    validate_weeks = list(range(4, 11))

    df_target = build_player_dataset(target_season)
    if df_target.empty:
        print("Dataset vazio")
        return

    print("Construindo features vetorizadas...")
    df_feat = build_features_vectorized(df_target)

    # Treina modelos com semanas <= train_through.
    train_through = min(validate_weeks) - 1
    train_df = df_feat[df_feat["week"] <= train_through].copy()
    train_df = train_df[train_df["half_ppr_points"].notna()]

    feature_cols = [
        "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
        "snap_share_trend", "targets_mean_3", "targets_mean_5",
        "targets_prev", "targets_trend", "games_played_prev_n", "weeks_active",
        "target_consistency",
    ]

    models = {
        "baseline_sma_3": None,
        "direct_ridge": None,
        "direct_lightgbm": None,
        "direct_xgboost": None,
    }

    # Prepara X, y de treino.
    def make_xy(d):
        X = d[feature_cols].fillna(0).values
        y = d["half_ppr_points"].values
        return X, y

    Xtr, ytr = make_xy(train_df)

    from sklearn.linear_model import Ridge, ElasticNet
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(Xtr, ytr)
    models["direct_ridge"] = ("ridge", ridge, feature_cols)

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                      random_state=42, verbosity=-1)
        lgb_model.fit(Xtr, ytr)
        models["direct_lightgbm"] = ("lgbm", lgb_model, feature_cols)
        print("LightGBM treinado.")
    except Exception as e:
        print(f"LightGBM falhou: {e}")

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     random_state=42, verbosity=0)
        xgb_model.fit(Xtr, ytr)
        models["direct_xgboost"] = ("xgb", xgb_model, feature_cols)
        print("XGBoost treinado.")
    except Exception as e:
        print(f"XGBoost falhou: {e}")

    # Avalia nas semanas de validação.
    print(f"\n=== RESULTADOS (semanas {validate_weeks}, treino até {train_through}) ===")
    print(f"{'Modelo':<20} {'n':>5} {'MAE':>7} {'RMSE':>7} {'Bias':>7} {'Spear':>7} {'SS%':>6} {'BaseMAE':>8} {'Imp%':>7}")

    all_results = {}
    for name, spec in models.items():
        if spec is None:
            # baseline: calcula por linha.
            preds = []; targets = []; base_preds = []
            for week in validate_weeks:
                sub = df_feat[df_feat["week"] == week]
                for _, r in sub.iterrows():
                    sid = r["sleeper_id"]
                    tgt = r["half_ppr_points"]
                    if pd.isna(tgt):
                        continue
                    # baseline = média móvel dos targets anteriores do jogador.
                    hist = df_feat[(df_feat["sleeper_id"] == sid) & (df_feat["week"] < week)]
                    vals = hist["half_ppr_points"].dropna().tail(3)
                    base = float(vals.mean()) if len(vals) else None
                    if base is None:
                        continue
                    preds.append(base); targets.append(float(tgt)); base_preds.append(base)
            pv = np.array(preds); tv = np.array(targets); bv = np.array(base_preds)
        else:
            kind, model, cols = spec
            preds = []; targets = []; base_preds = []
            for week in validate_weeks:
                sub = df_feat[df_feat["week"] == week]
                for _, r in sub.iterrows():
                    sid = r["sleeper_id"]
                    tgt = r["half_ppr_points"]
                    if pd.isna(tgt):
                        continue
                    x = r[cols].fillna(0).values.reshape(1, -1)
                    pred = float(model.predict(x)[0])
                    # baseline.
                    hist = df_feat[(df_feat["sleeper_id"] == sid) & (df_feat["week"] < week)]
                    vals = hist["half_ppr_points"].dropna().tail(3)
                    base = float(vals.mean()) if len(vals) else None
                    if base is None:
                        continue
                    preds.append(pred); targets.append(float(tgt)); base_preds.append(base)
            pv = np.array(preds); tv = np.array(targets); bv = np.array(base_preds)

        if len(pv) == 0:
            print(f"{name:<20} sem dados")
            continue

        mae = float(np.mean(np.abs(pv - tv)))
        rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        bias = float(np.mean(pv - tv))
        spear = _spearman(pv.tolist(), tv.tolist()) if len(pv) >= 3 else None

        ss_right = ss_total = 0
        for i in range(len(pv)):
            for j in range(i + 1, len(pv)):
                if tv[i] == tv[j] or pv[i] == pv[j]:
                    continue
                ss_total += 1
                if (pv[i] > pv[j]) == (tv[i] > tv[j]):
                    ss_right += 1
        ss = (ss_right / ss_total * 100) if ss_total else 0

        base_mae = float(np.mean(np.abs(bv - tv)))
        imp = (base_mae - mae) / base_mae * 100 if base_mae else 0

        spear_str = f"{spear:>7.3f}" if spear is not None else "  n/a "
        print(f"{name:<20} {len(pv):>5} {mae:>7.3f} {rmse:>7.3f} {bias:>7.3f} "
              f"{spear_str} {ss:>6.1f} "
              f"{base_mae:>8.3f} {imp:>7.1f}")

        all_results[name] = {
            "n": int(len(pv)), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "bias": round(bias, 4), "spearman": round(spear, 4) if spear is not None else None,
            "start_sit_pct": round(ss, 2),
            "baseline_mae": round(base_mae, 4),
            "improvement_pct": round(imp, 2),
        }

    # Salva.
    out = {
        "season": target_season,
        "validate_weeks": validate_weeks,
        "train_through": train_through,
        "results": all_results,
    }
    out_path = Path("data/experiments/backtest_vectorized_2026.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
