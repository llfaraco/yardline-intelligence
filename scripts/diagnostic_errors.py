"""Diagnóstico detalhado dos erros do modelo volume × eficiência.

Isola onde está o problema:
1. VolumeModel: predição de targets vs targets reais.
2. EfficiencyModel: predição de pts/alvo vs pts/alvo real.
3. Composite: pontos preditos vs pontos reais.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yardline_intelligence.config import CURRENT_SEASON
from yardline_intelligence.data_foundation.dataset import build_player_dataset
from yardline_intelligence.temporal_feature_store.expanded_store import ExpandedFeatureStore
from yardline_intelligence.models.wr_models import VolumeModel, EfficiencyModel, CompositeWithBaseline
from yardline_intelligence.temporal_feature_store.store import moving_average_baseline


def main():
    target_season = CURRENT_SEASON
    validate_weeks = [9, 10, 11]
    train_through = min(validate_weeks) - 1

    df_target = build_player_dataset(target_season)
    store = ExpandedFeatureStore(target_season, from_df=df_target)

    # Treina modelos.
    volume = VolumeModel(kind="ridge", alpha=1.0).fit(store, train_through, position="WR")
    efficiency = EfficiencyModel(kind="ridge", alpha=1.0).fit(store, train_through, position="WR")
    composite = CompositeWithBaseline(volume, efficiency)

    print(f"Volume fitted: {volume.fitted}, Efficiency fitted: {efficiency.fitted}")
    print(f"Volume coef: {volume.coef_}")
    print(f"Volume intercept: {volume.intercept_}")
    print(f"Efficiency coef: {efficiency.coef_}")
    print(f"Efficiency intercept: {efficiency.intercept_}")
    print()

    # Diagnóstico 1: VolumeModel - predição de targets.
    print("=== DIAGNÓSTICO 1: VolumeModel (targets) ===")
    vol_errors = []
    vol_preds = []
    vol_targets = []
    for week in validate_weeks:
        for sid in store.players(week):
            feats = store.features_for(sid, week)
            tgt = feats.get("targets")
            if tgt is None:
                continue
            pred = volume.predict(feats)
            if pred is None:
                continue
            vol_errors.append(abs(pred - tgt))
            vol_preds.append(pred)
            vol_targets.append(float(tgt))

    if vol_preds:
        vol_mae = float(np.mean(np.abs(np.array(vol_preds) - np.array(vol_targets))))
        vol_bias = float(np.mean(np.array(vol_preds) - np.array(vol_targets)))
        vol_spear = _spearman(vol_preds, vol_targets)
        print(f"n={len(vol_preds)} MAE_targets={vol_mae:.3f} Bias={vol_bias:.3f} Spearman={vol_spear:.3f}")
        print(f"  Targets reais: média={np.mean(vol_targets):.2f}, desvio={np.std(vol_targets):.2f}")
        print(f"  Targets preditos: média={np.mean(vol_preds):.2f}, desvio={np.std(vol_preds):.2f}")
    else:
        print("Sem predições de volume.")

    print()

    # Diagnóstico 2: EfficiencyModel - predição de pts/alvo.
    print("=== DIAGNÓSTICO 2: EfficiencyModel (pts/alvo) ===")
    eff_errors = []
    eff_preds = []
    eff_targets = []
    for week in validate_weeks:
        for sid in store.players(week):
            feats = store.features_for(sid, week)
            pts = feats.get("half_ppr_points")
            tgt_prev = feats.get("targets_prev")
            if pts is None or tgt_prev is None or tgt_prev <= 0:
                continue
            real_eff = pts / max(tgt_prev, 1.0)
            pred = efficiency.predict(feats)
            if pred is None:
                continue
            eff_errors.append(abs(pred - real_eff))
            eff_preds.append(pred)
            eff_targets.append(real_eff)

    if eff_preds:
        eff_mae = float(np.mean(np.abs(np.array(eff_preds) - np.array(eff_targets))))
        eff_bias = float(np.mean(np.array(eff_preds) - np.array(eff_targets)))
        eff_spear = _spearman(eff_preds, eff_targets)
        print(f"n={len(eff_preds)} MAE_eff={eff_mae:.3f} Bias={eff_bias:.3f} Spearman={eff_spear:.3f}")
        print(f"  Eficiência real: média={np.mean(eff_targets):.3f}, desvio={np.std(eff_targets):.3f}")
        print(f"  Eficiência predita: média={np.mean(eff_preds):.3f}, desvio={np.std(eff_preds):.3f}")
    else:
        print("Sem predições de eficiência.")

    print()

    # Diagnóstico 3: Composite - pontos preditos vs reais.
    print("=== DIAGNÓSTICO 3: Composite (pontos) ===")
    comp_errors = []
    comp_preds = []
    comp_targets = []
    for week in validate_weeks:
        for sid in store.players(week):
            feats = store.features_for(sid, week)
            pts = feats.get("half_ppr_points")
            if pts is None:
                continue
            pred = composite.predict(feats, store, sid, week)
            if pred is None:
                continue
            comp_errors.append(abs(pred - pts))
            comp_preds.append(pred)
            comp_targets.append(float(pts))

    if comp_preds:
        comp_mae = float(np.mean(np.abs(np.array(comp_preds) - np.array(comp_targets))))
        comp_bias = float(np.mean(np.array(comp_preds) - np.array(comp_targets)))
        comp_spear = _spearman(comp_preds, comp_targets)
        print(f"n={len(comp_preds)} MAE_pts={comp_mae:.3f} Bias={comp_bias:.3f} Spearman={comp_spear:.3f}")
        print(f"  Pontos reais: média={np.mean(comp_targets):.2f}, desvio={np.std(comp_targets):.2f}")
        print(f"  Pontos preditos: média={np.mean(comp_preds):.2f}, desvio={np.std(comp_preds):.2f}")

        # Mostra alguns exemplos de erros grandes.
        err_arr = np.array(comp_errors)
        idx_worst = np.argsort(err_arr)[-5:][::-1]
        print("  Piores 5 predições (pred, real, erro):")
        for i in idx_worst:
            print(f"    pred={comp_preds[i]:.2f} real={comp_targets[i]:.2f} erro={comp_preds[i]-comp_targets[i]:.2f}")
    else:
        print("Sem predições do composite.")

    print()

    # Baseline para referência.
    print("=== REFERÊNCIA: Baseline SMA-3 ===")
    base_errors = []
    base_preds = []
    base_targets = []
    for week in validate_weeks:
        for sid in store.players(week):
            feats = store.features_for(sid, week)
            pts = feats.get("half_ppr_points")
            if pts is None:
                continue
            base = moving_average_baseline(store, sid, week, window=3)
            if base is None:
                continue
            base_errors.append(abs(base - pts))
            base_preds.append(base)
            base_targets.append(float(pts))

    if base_preds:
        base_mae = float(np.mean(np.abs(np.array(base_preds) - np.array(base_targets))))
        base_bias = float(np.mean(np.array(base_preds) - np.array(base_targets)))
        base_spear = _spearman(base_preds, base_targets)
        print(f"n={len(base_preds)} MAE={base_mae:.3f} Bias={base_bias:.3f} Spearman={base_spear:.3f}")
        print(f"  Pontos preditos (baseline): média={np.mean(base_preds):.2f}, desvio={np.std(base_preds):.2f}")


def _spearman(x, y):
    if len(x) < 3:
        return None
    xr = _rank(x)
    yr = _rank(y)
    n = len(xr)
    mx = sum(xr) / n
    my = sum(yr) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    den = (sum((a - mx) ** 2 for a in xr) * sum((b - my) ** 2 for b in yr)) ** 0.5
    if den == 0:
        return None
    return num / den


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


if __name__ == "__main__":
    main()
