"""Backtest otimizado v0.2 — pré-calcula features uma vez, depois avalia modelos.

Mais rápido e cobre semanas 4-10 para estatística robusta.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yardline_intelligence.config import CURRENT_SEASON
from yardline_intelligence.data_foundation.dataset import build_player_dataset
from yardline_intelligence.temporal_feature_store.expanded_store import ExpandedFeatureStore
from yardline_intelligence.models.direct_model import DirectPointsModel, _spearman
from yardline_intelligence.temporal_feature_store.store import moving_average_baseline


def precompute_features(store: ExpandedFeatureStore, weeks: list[int]) -> dict:
    """Pré-calcula features para todos (jogador, semana) e retorna dicionário."""
    cache = {}
    for week in weeks:
        for sid in store.players(week):
            feats = store.features_for(sid, week)
            cache[(sid, week)] = feats
    return cache


def evaluate_with_cache(store, models: dict, cache: dict, validate_weeks: list[int]):
    """Avalia modelos usando features pré-calculadas."""
    results = {}
    for name in models:
        results[name] = {"preds": [], "targets": [], "base_preds": []}

    for week in validate_weeks:
        for sid in store.players(week):
            feats = cache.get((sid, week), {})
            target = feats.get("half_ppr_points")
            if target is None:
                continue
            target = float(target)
            base = moving_average_baseline(store, sid, week, window=3)

            for name, model in models.items():
                if name == "baseline_sma_3":
                    results[name]["preds"].append(base)
                    results[name]["targets"].append(target)
                    results[name]["base_preds"].append(base)
                    continue
                if model is None or not model.fitted:
                    continue
                pred = model.predict(feats)
                if pred is None:
                    continue
                results[name]["preds"].append(pred)
                results[name]["targets"].append(target)
                results[name]["base_preds"].append(base if base is not None else pred)

    return results


def summarize(name, preds, targets, base_preds):
    """Resume métricas para um modelo."""
    valid = [(p, t, b) for p, t, b in zip(preds, targets, base_preds) if p is not None]
    if not valid:
        return None
    pv = np.array([v[0] for v in valid], dtype=float)
    tv = np.array([v[1] for v in valid], dtype=float)
    bv = np.array([v[2] for v in valid], dtype=float)

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

    return {
        "n": len(pv), "mae": round(mae, 4), "rmse": round(rmse, 4),
        "bias": round(bias, 4), "spearman": round(spear, 4) if spear is not None else None,
        "start_sit_pct": round(ss, 2),
        "baseline_mae": round(base_mae, 4),
        "improvement_pct": round(imp, 2),
        "model_mean_pred": round(float(pv.mean()), 3),
        "target_mean": round(float(tv.mean()), 3),
    }


def main():
    target_season = CURRENT_SEASON
    validate_weeks = list(range(4, 11))  # semanas 4-10

    df_target = build_player_dataset(target_season)
    store = ExpandedFeatureStore(target_season, from_df=df_target)

    print("Pré-calculando features...")
    cache = precompute_features(store, validate_weeks)
    print(f"  {(len(cache))} combinações (jogador, semana) calculadas")

    # Treina modelos.
    train_through = min(validate_weeks) - 1
    print(f"Treinando modelos até semana {train_through}...")
    models = {
        "baseline_sma_3": None,
        "direct_ridge": DirectPointsModel(kind="ridge", alpha=1.0),
        "direct_lightgbm": DirectPointsModel(kind="lightgbm", alpha=1.0),
        "direct_xgboost": DirectPointsModel(kind="xgboost", alpha=1.0),
    }
    fitted = {}
    for name, m in models.items():
        if m is None:
            fitted[name] = None
            continue
        try:
            fitted[name] = m.fit(store, train_through, position="WR")
            print(f"  {name}: fitted={fitted[name].fitted}")
        except Exception as e:
            print(f"  {name}: ERRO: {e}")
            fitted[name] = None

    print(f"\n=== RESULTADOS (semanas {validate_weeks}) ===")
    print(f"{'Modelo':<24} {'n':>5} {'MAE':>7} {'RMSE':>7} {'Bias':>7} {'Spear':>7} {'SS%':>6} {'BaseMAE':>8} {'Imp%':>7}")
    all_results = {}
    for name in models:
        r = results = evaluate_with_cache(store, {name: fitted[name]}, cache, validate_weeks)
        summary = summarize(name, r[name]["preds"], r[name]["targets"], r[name]["base_preds"])
        if summary is None:
            print(f"{name:<24} sem dados")
            continue
        all_results[name] = summary
        print(f"{name:<24} {summary['n']:>5} {summary['mae']:>7.3f} {summary['rmse']:>7.3f} "
              f"{summary['bias']:>7.3f} {str(summary['spearman']):>7} {summary['start_sit_pct']:>6.1f} "
              f"{summary['baseline_mae']:>8.3f} {summary['improvement_pct']:>7.1f}")

    # Análise por faixa de volume (LightGBM).
    print("\n=== ERRO POR FAIXA DE VOLUME (LightGBM, alvo = targets_mean_3) ===")
    lgbm = fitted.get("direct_lightgbm")
    if lgbm and lgbm.fitted:
        for vol_label, vol_min, vol_max in [("Baixo (0-5)", 0, 5),
                                             ("Médio (5-10)", 5, 10),
                                             ("Alto (10+)", 10, 999)]:
            errors = []
            base_errors = []
            for week in validate_weeks:
                for sid in store.players(week):
                    feats = cache.get((sid, week), {})
                    target = feats.get("half_ppr_points")
                    prev_targets = feats.get("targets_mean_3")
                    if target is None or prev_targets is None:
                        continue
                    if not (vol_min <= prev_targets < vol_max):
                        continue
                    pred = lgbm.predict(feats)
                    base = moving_average_baseline(store, sid, week, window=3)
                    if pred is None or base is None:
                        continue
                    errors.append(abs(pred - target))
                    base_errors.append(abs(base - target))
            if errors:
                mae = float(np.mean(errors))
                bmae = float(np.mean(base_errors))
                print(f"  {vol_label:<16} n={len(errors):>4} MAE_model={mae:.3f} MAE_base={bmae:.3f} "
                      f"Imp={((bmae - mae) / bmae * 100) if bmae else 0:+.1f}%")

    # Salva.
    out = {
        "season": target_season,
        "validate_weeks": validate_weeks,
        "train_through": train_through,
        "results": all_results,
    }
    out_path = Path("data/experiments/backtest_expanded_2026.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
