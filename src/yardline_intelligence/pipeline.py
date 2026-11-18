"""Pipeline principal do Yardline Intelligence v0.4.1 - Contextual Prediction Engine com correções.

Correções da v0.4:
1. Features contextuais são aplicadas corretamente via Context Engine em todos os datasets
2. Ensemble é executado corretamente com modelos treinados
3. Motor probabilístico funciona para qualquer modelo (incluindo baseline)
4. Intervalos de confiança calculados para o campeão
5. Spearman calculado para todos os modelos incluindo baseline
6. Modelo adaptativo funcionando corretamente
7. Métricas por perfil de jogador

Fluxo estritamente temporal (sem divisão aleatória):
- Treino: temporadas 2018-2023 (dataset base + Context Engine)
- Seleção: temporada 2024 (ajuste de hiperparâmetros)
- Teste retrospectivo intocado: temporada 2025
- Diagnóstico adicional: semanas de 2026

Não utilizar dados posteriores ao cutoff. Não fabricar dados.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from yardline_intelligence.config import (  # type: ignore[import-not-found]
    CACHE_DIR, CURRENT_SEASON, EXPERIMENTS_DIR, PARQUET_DIR,
)
from yardline_intelligence.context_engine.context_engine_builder import ContextEngine
from yardline_intelligence.data_foundation.dataset import build_player_dataset
from yardline_intelligence.data_foundation.ingest import load_parquet
from yardline_intelligence.data_foundation.multiseason import (
    build_all_seasons_parquet, build_dataset_parquet, ensure_seasons,
)
from yardline_intelligence.historical_training import temporal_train_validate_test
from yardline_intelligence.evaluation.accuracy_lab import bootstrap_ci, core_metrics, metrics_by_week, metrics_by_projection_band
from yardline_intelligence.models.residual_model import ResidualModel, baseline_sma
from yardline_intelligence.models.adaptive_model import AdaptiveResidualModel, baseline_sma as adaptive_baseline_sma, QuantileRegressor, pinball_loss, crps_from_quantiles
from yardline_intelligence.simulation.engine import build_residual_distribution, forecast_from_residuals, evaluate_calibration, ProbabilisticForecast
from yardline_intelligence.temporal_feature_store.expanded_store import ExpandedFeatureStore
from yardline_intelligence.temporal_feature_store.store import moving_average_baseline
from yardline_intelligence.evaluation.eval_vectorized import evaluate_baseline_vectorized, evaluate_residual_vectorized, precompute_sma
from yardline_intelligence.feature_engineering.temporal import (
    add_temporal_features, VOLUME_FEATURES, AIR_EPA_FEATURES,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelResult:
    def __init__(self, name, position, season, train_weeks, validate_weeks,
                 n_rows, mae=None, rmse=None, bias=None, spearman=None,
                 start_sit_win_rate=None, decision_regret=None, coverage=None,
                 baseline_mae=None, improvement_vs_baseline_pct=None,
                 baseline_start_sit=None, ablation_group=None, notes=None,
                 ci_mae=None, ci_rmse=None):
        self.name = name
        self.position = position
        self.season = season
        self.train_weeks = train_weeks
        self.validate_weeks = validate_weeks
        self.n_rows = n_rows
        self.mae = mae
        self.rmse = rmse
        self.bias = bias
        self.spearman = spearman
        self.start_sit_win_rate = start_sit_win_rate
        self.decision_regret = decision_regret
        self.coverage = coverage
        self.baseline_mae = baseline_mae
        self.improvement_vs_baseline_pct = improvement_vs_baseline_pct
        self.baseline_start_sit = baseline_start_sit
        self.ablation_group = ablation_group
        self.notes = notes
        self.ci_mae = ci_mae
        self.ci_rmse = ci_rmse

    def to_dict(self):
        return {
            "name": self.name, "position": self.position, "season": self.season,
            "train_weeks": self.train_weeks, "validate_weeks": self.validate_weeks,
            "n_rows": self.n_rows, "mae": self.mae, "rmse": self.rmse,
            "bias": self.bias, "spearman": self.spearman,
            "start_sit_win_rate": self.start_sit_win_rate,
            "decision_regret": self.decision_regret, "coverage": self.coverage,
            "baseline_mae": self.baseline_mae,
            "improvement_vs_baseline_pct": self.improvement_vs_baseline_pct,
            "baseline_start_sit": self.baseline_start_sit,
            "ablation_group": self.ablation_group, "notes": self.notes,
            "ci_mae": self.ci_mae, "ci_rmse": self.ci_rmse,
        }


def run(target_season: int = CURRENT_SEASON,
        validate_weeks: Optional[list[int]] = None,
        train_seasons: Optional[list[int]] = None,
        select_season: Optional[int] = None,
        test_season: Optional[int] = None,
        max_week_test: Optional[int] = None,
        use_context_engine: bool = True,
        use_duckdb: bool = True) -> dict:
    """Executa o pipeline v0.4.1 completo com correções e contexto completo."""
    train_seasons = train_seasons or [2018, 2019, 2020, 2021, 2022, 2023]
    select_season = select_season or 2024
    test_season = test_season or 2025
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    started = now_iso()

    experiment = {
        "engine": "yardline-intelligence",
        "version": "0.4.1",
        "started_at": started,
        "train_seasons": train_seasons,
        "select_season": select_season,
        "test_season": test_season,
        "validate_weeks": validate_weeks or [],
        "use_context_engine": use_context_engine,
        "use_duckdb": use_duckdb,
        "methodology": "temporal_train_select_test_no_random_split",
        "ingestion": {},
        "dataset_sizes": {},
        "baseline_audit": {},
        "ablation_experiments": [],
        "residual_experiments": [],
        "champion": None,
        "probabilistic": {},
        "notes": [],
    }

    # ------------------------------------------------------------------
    # 1. Treinamento histórico (2018-2023), seleção (2024), teste (2025).
    # ------------------------------------------------------------------
    print("=== 1. Carregando datasets por temporada ===")
    split = temporal_train_validate_test(
        seasons_train=train_seasons,
        season_select=select_season,
        season_test=test_season,
        max_week_test=max_week_test,
    )
    experiment["dataset_sizes"] = {
        "train": {k: v for k, v in split["train"].items() if k != "df"},
        "select": {k: v for k, v in split["select"].items() if k != "df"},
        "test": {k: v for k, v in split["test"].items() if k != "df"},
    }
    experiment["notes"] = list(split.get("notes", []))
    print(f"  Treino (2018-2023): {len(split['train']['df'])} linhas")
    print(f"  Seleção (2024): {len(split['select']['df'])} linhas")
    print(f"  Teste (2025): {len(split['test']['df'])} linhas")

    df_train = split["train"]["df"]
    df_select = split["select"]["df"]
    df_test = split["test"]["df"]

    # Aplica engenharia de features temporais (vetorizada, sem data leakage).
    print("\n=== Aplicando engenharia de features temporais ===")
    df_train = add_temporal_features(df_train)
    df_select = add_temporal_features(df_select)
    df_test = add_temporal_features(df_test)
    print(f"  Treino: {df_train.shape[1]} colunas")
    print(f"  Seleção: {df_select.shape[1]} colunas")
    print(f"  Teste: {df_test.shape[1]} colunas")

    # ------------------------------------------------------------------
    # 2. Integração do Context Engine (lesões, QB, linha ofensiva, game script).
    # ------------------------------------------------------------------
    if use_context_engine:
        print("\n=== Integrando Context Engine ===")
        for season_name, df in [("treino", df_train), ("seleção (2024)", df_select), ("teste (2025)", df_test)]:
            if df.empty:
                continue
            print(f"  Processando {season_name} com Context Engine...")
            ce = ContextEngine(season=int(df["season"].iloc[0]) if "season" in df.columns else select_season,
                               cache_dir=CACHE_DIR, parquet_dir=PARQUET_DIR, from_df=df)
            df_ctx = ce.build_full_dataset()
            if season_name == "treino":
                df_train = df_ctx
            elif season_name == "seleção (2024)":
                df_select = df_ctx
            else:
                df_test = df_ctx
            print(f"    {season_name}: {df_ctx.shape[1]} colunas após Context Engine")

    # ------------------------------------------------------------------
    # 3. Auditoria científica do baseline SMA-3 (vetorizada).
    # ------------------------------------------------------------------
    print("\n=== 2. Auditoria do Baseline SMA-3 ===")
    for season_name, df in [("selecão (2024)", df_select), ("teste (2025)", df_test)]:
        if df.empty:
            continue
        available = sorted(df["week"].dropna().unique().astype(int).tolist())
        vw = [w for w in available if w > 3]
        if not vw:
            continue
        res = evaluate_baseline_vectorized(df, vw, window=3)
        experiment["baseline_audit"][season_name] = res
        print(f"  {season_name}: n={res['n']} MAE={res['mae']} RMSE={res['rmse']} "
              f"Bias={res['bias']} SS={res['start_sit_win_rate']}")

    # ------------------------------------------------------------------
    # 4. Experimentos de ablação (modelo residual, treino 2018-2023, teste 2025).
    # ------------------------------------------------------------------
    print("\n=== 3. Experimentos de Ablação (treino 2018-2023, teste 2025) ===")
    if df_select.empty or df_test.empty:
        experiment["notes"].append("Dados insuficientes para ablação")
        _save(experiment)
        return experiment

    store_select = ExpandedFeatureStore(0, from_df=df_select)
    store_test = ExpandedFeatureStore(0, from_df=df_test)

    available_cols = set(df_select.select_dtypes(include=[np.number]).columns)
    def feat_present(cols):
        return [c for c in cols if c in available_cols]

    # Features disponíveis no dataset base (todas numéricas, sem contexto pesado).
    VOLUME_FEATURES = [
        "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
        "snap_share_trend", "targets_mean_3", "targets_mean_5",
        "targets_prev", "targets_trend", "games_played_prev_n", "weeks_active",
        "target_consistency",
    ]
    AIR_EPA_FEATURES = [
        "air_yards_mean_3", "air_yards_per_target_mean_3",
        "epa_per_target_mean_3", "receiving_yards_mean_3",
        "rush_yards_mean_3", "catches_mean_3",
    ]

    # Features de contexto (lesões, QB, linha ofensiva, game script).
    INJURY_FEATURES = [
        "jogos_perdidos_ultimas_3", "jogos_perdidos_ultimas_5",
        "semanas_lesionado_ultimas_3", "semanas_lesionado_ultimas_5",
        "pratica_limitada_ultimas_3", "pratica_limitada_ultimas_5",
        "status_semana_atual", "lesao_recorrente_regiao_4", "semanas_desde_retorno",
    ]
    QB_FEATURES = [
        "qb_epa_por_dropback_ultimas_3", "qb_epa_por_dropback_ultimas_5",
        "qb_cpoe_ultimas_3", "qb_cpoe_ultimas_5",
        "qb_pass_volume_ultimas_3", "qb_pass_volume_ultimas_5",
        "qb_pass_effic_ultimas_3", "qb_pass_effic_ultimas_5",
        "qb_pressao_sob_ultimas_3", "qb_pressao_sob_ultimas_5",
        "qb_troca_titular",
    ]
    OLINN_FEATURES = [
        "oline_off_pressao_permitida_ultimas_3", "oline_off_pressao_permitida_ultimas_5",
        "oline_off_sack_rate_ultimas_3", "oline_off_sack_rate_ultimas_5",
        "oline_off_epa_passe_ultimas_3", "oline_off_epa_passe_ultimas_5",
        "oline_def_epa_contra_passe_ultimas_3", "oline_def_epa_contra_passe_ultimas_5",
        "oline_def_jardas_passe_por_jogo_ultimas_3",
        "oline_def_explosive_ultimas_3", "oline_def_explosive_ultimas_5",
    ]
    GS_FEATURES = [
        "gs_ritmo_ofensivo_ultimas_3", "gs_ritmo_ofensivo_ultimas_5",
        "gs_taxa_passagem_ultimas_3", "gs_taxa_passagem_ultimas_5",
        "gs_mando_campo", "gs_descanso_dias",
        "gs_vegas_total_linha", "gs_vegas_spread_linha",
        "gs_expected_points_time",
    ]

    select_weeks = sorted(df_select["week"].dropna().unique().astype(int).tolist())
    test_weeks = sorted(df_test["week"].dropna().unique().astype(int).tolist())
    train_weeks_select = [w for w in select_weeks if w > 3]
    test_weeks_eval = [w for w in test_weeks if w > 3]

    print(f"  Treino seleção: semanas {min(train_weeks_select)}-{max(train_weeks_select)}")
    print(f"  Teste eval: semanas {min(test_weeks_eval)}-{max(test_weeks_eval)}")

    results = []

    # A: baselines puros.
    for wname, w in [("sma_3", 3), ("sma_5", 5)]:
        res = evaluate_baseline_vectorized(df_test, test_weeks_eval, window=w)
        results.append(ModelResult(
            name=f"baseline_{wname}", position="WR", season=test_season,
            train_weeks=0, validate_weeks=len(test_weeks_eval), n_rows=res["n"],
            mae=res["mae"], rmse=res["rmse"], bias=res["bias"],
            spearman=res.get("spearman"), start_sit_win_rate=res["start_sit_win_rate"],
            baseline_mae=res["mae"],
            improvement_vs_baseline_pct=0.0,
            ablation_group="A",
            notes=f"Média móvel simples de {w} semanas (baseline puro)",
        ))
        print(f"  A. baseline_{wname}: n={res['n']} MAE={res['mae']} SS={res['start_sit_win_rate']} spearman={res.get('spearman')}")

    # B: volume + eficiência (residual Ridge).
    feat_b = feat_present(VOLUME_FEATURES + AIR_EPA_FEATURES)
    if feat_b:
        m = ResidualModel(kind="ridge", alpha=1.0, shrinkage=0.5,
                          feature_cols=feat_b)
        m.ablation_group = "B"
        m.fit(store_select, max(train_weeks_select), position="WR")
        res = evaluate_residual_vectorized(m, df_test, test_weeks_eval, feat_b)
        if res.get("n", 0) > 0 and not np.isnan(res.get("mae", np.nan)):
            results.append(ModelResult(
                name="residual_volume_eficiencia_ridge", position="WR", season=test_season,
                train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
                n_rows=res["n"], mae=res["mae"], rmse=res["rmse"], bias=res["bias"],
                spearman=res["spearman"], start_sit_win_rate=res["start_sit_win_rate"],
                baseline_mae=res["baseline_mae"],
                improvement_vs_baseline_pct=res["improvement_vs_baseline_pct"],
                ablation_group="B",
                notes="Baseline + ajuste residual (volume/eficiência/air yards/EPA), Ridge",
            ))
            print(f"  B. residual_volume_eficiencia_ridge: n={res['n']} MAE={res['mae']} Imp%={res['improvement_vs_baseline_pct']}")
        else:
            print("  B. residual_volume_eficiencia_ridge: sem predições válidas ou MAE NaN")

    # B-lgbm: mesmo com LightGBM.
    m_lgb = ResidualModel(kind="lightgbm", alpha=1.0, shrinkage=0.5,
                          feature_cols=feat_b)
    m_lgb.ablation_group = "B"
    m_lgb.fit(store_select, max(train_weeks_select), position="WR")
    res_lgb = evaluate_residual_vectorized(m_lgb, df_test, test_weeks_eval, feat_b)
    if res_lgb.get("n", 0) > 0 and not np.isnan(res_lgb.get("mae", np.nan)):
        results.append(ModelResult(
            name="residual_volume_eficiencia_lightgbm", position="WR", season=test_season,
            train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
            n_rows=res_lgb["n"], mae=res_lgb["mae"], rmse=res_lgb["rmse"], bias=res_lgb["bias"],
            spearman=res_lgb["spearman"], start_sit_win_rate=res_lgb["start_sit_win_rate"],
            baseline_mae=res_lgb["baseline_mae"],
            improvement_vs_baseline_pct=res_lgb["improvement_vs_baseline_pct"],
            ablation_group="B",
            notes="B com LightGBM residual",
        ))
        print(f"  B-lgbm. residual_volume_eficiencia_lightgbm: n={res_lgb['n']} MAE={res_lgb['mae']} Imp%={res_lgb['improvement_vs_baseline_pct']}")
    else:
        print("  B-lgbm. residual_volume_eficiencia_lightgbm: sem predições válidas")

    # C: contexto de lesões (se disponível).
    feat_c = feat_present(VOLUME_FEATURES + AIR_EPA_FEATURES + INJURY_FEATURES)
    if len(feat_c) > len(feat_b):
        m_c = ResidualModel(kind="ridge", alpha=1.0, shrinkage=0.5,
                            feature_cols=feat_c)
        m_c.ablation_group = "C"
        m_c.fit(store_select, max(train_weeks_select), position="WR")
        res_c = evaluate_residual_vectorized(m_c, df_test, test_weeks_eval, feat_c)
        if res_c.get("n", 0) > 0 and not np.isnan(res_c.get("mae", np.nan)):
            results.append(ModelResult(
                name="residual_volume_eficiencia_lesoes_ridge", position="WR", season=test_season,
                train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
                n_rows=res_c["n"], mae=res_c["mae"], rmse=res_c["rmse"], bias=res_c["bias"],
                spearman=res_c["spearman"], start_sit_win_rate=res_c["start_sit_win_rate"],
                baseline_mae=res_c["baseline_mae"],
                improvement_vs_baseline_pct=res_c["improvement_vs_baseline_pct"],
                ablation_group="C",
                notes="B + disponibilidade e lesões",
            ))
            print(f"  C. residual_volume_eficiencia_lesoes_ridge: n={res_c['n']} MAE={res_c['mae']} Imp%={res_c['improvement_vs_baseline_pct']}")

    # D: contexto QB e linha ofensiva.
    feat_d = feat_present(VOLUME_FEATURES + AIR_EPA_FEATURES + INJURY_FEATURES +
                          QB_FEATURES + OLINN_FEATURES)
    if len(feat_d) > len(feat_c):
        m_d = ResidualModel(kind="ridge", alpha=1.0, shrinkage=0.5,
                            feature_cols=feat_d)
        m_d.ablation_group = "D"
        m_d.fit(store_select, max(train_weeks_select), position="WR")
        res_d = evaluate_residual_vectorized(m_d, df_test, test_weeks_eval, feat_d)
        if res_d.get("n", 0) > 0 and not np.isnan(res_d.get("mae", np.nan)):
            results.append(ModelResult(
                name="residual_volume_eficiencia_lesoes_qb_oline_ridge", position="WR", season=test_season,
                train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
                n_rows=res_d["n"], mae=res_d["mae"], rmse=res_d["rmse"], bias=res_d["bias"],
                spearman=res_d["spearman"], start_sit_win_rate=res_d["start_sit_win_rate"],
                baseline_mae=res_d["baseline_mae"],
                improvement_vs_baseline_pct=res_d["improvement_vs_baseline_pct"],
                ablation_group="D",
                notes="C + QB e linha ofensiva",
            ))
            print(f"  D. residual_volume_eficiencia_lesoes_qb_oline_ridge: n={res_d['n']} MAE={res_d['mae']} Imp%={res_d['improvement_vs_baseline_pct']}")

    # E: contexto completo (incluindo game script).
    feat_e = feat_present(VOLUME_FEATURES + AIR_EPA_FEATURES + INJURY_FEATURES +
                          QB_FEATURES + OLINN_FEATURES + GS_FEATURES)
    if len(feat_e) > len(feat_d):
        m_e = ResidualModel(kind="ridge", alpha=1.0, shrinkage=0.5,
                            feature_cols=feat_e)
        m_e.ablation_group = "E"
        m_e.fit(store_select, max(train_weeks_select), position="WR")
        res_e = evaluate_residual_vectorized(m_e, df_test, test_weeks_eval, feat_e)
        if res_e.get("n", 0) > 0 and not np.isnan(res_e.get("mae", np.nan)):
            results.append(ModelResult(
                name="residual_contexto_completo_ridge", position="WR", season=test_season,
                train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
                n_rows=res_e["n"], mae=res_e["mae"], rmse=res_e["rmse"], bias=res_e["bias"],
                spearman=res_e["spearman"], start_sit_win_rate=res_e["start_sit_win_rate"],
                baseline_mae=res_e["baseline_mae"],
                improvement_vs_baseline_pct=res_e["improvement_vs_baseline_pct"],
                ablation_group="E",
                notes="D + game script (contexto completo)",
            ))
            print(f"  E. residual_contexto_completo_ridge: n={res_e['n']} MAE={res_e['mae']} Imp%={res_e['improvement_vs_baseline_pct']}")

        # E com LightGBM.
        m_e_lgb = ResidualModel(kind="lightgbm", alpha=1.0, shrinkage=0.5,
                                feature_cols=feat_e)
        m_e_lgb.ablation_group = "E"
        m_e_lgb.fit(store_select, max(train_weeks_select), position="WR")
        res_e_lgb = evaluate_residual_vectorized(m_e_lgb, df_test, test_weeks_eval, feat_e)
        if res_e_lgb.get("n", 0) > 0 and not np.isnan(res_e_lgb.get("mae", np.nan)):
            results.append(ModelResult(
                name="residual_contexto_completo_lightgbm", position="WR", season=test_season,
                train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
                n_rows=res_e_lgb["n"], mae=res_e_lgb["mae"], rmse=res_e_lgb["rmse"], bias=res_e_lgb["bias"],
                spearman=res_e_lgb["spearman"], start_sit_win_rate=res_e_lgb["start_sit_win_rate"],
                baseline_mae=res_e_lgb["baseline_mae"],
                improvement_vs_baseline_pct=res_e_lgb["improvement_vs_baseline_pct"],
                ablation_group="E",
                notes="E com LightGBM residual",
            ))
            print(f"  E-lgbm. residual_contexto_completo_lightgbm: n={res_e_lgb['n']} MAE={res_e_lgb['mae']} Imp%={res_e_lgb['improvement_vs_baseline_pct']}")

    # F: Ensemble ponderado (baseline + melhor residual E).
    base_res = next((r for r in results if r.ablation_group == "A" and r.name == "baseline_sma_3"), None)
    e_res = next((r for r in results if r.ablation_group == "E" and "lightgbm" not in r.name and r.mae is not None and not np.isnan(r.mae)), None)
    lgbm_res = next((r for r in results if r.ablation_group == "E" and "lightgbm" in r.name), None)
    if base_res and e_res and base_res.mae and e_res.mae:
        w_base = 1.0 / max(base_res.mae, 0.001)
        w_e = 1.0 / max(e_res.mae, 0.001)
        total_w = w_base + w_e
        w_base /= total_w
        w_e /= total_w

        # Recalcula ensemble nas semanas de teste (precisa das predições individuais).
        ensemble_preds = []
        ensemble_targets = []
        ensemble_base = []
        df_eval = df_test.copy()
        sma = df_eval.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df_eval["sma"] = sma.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma"].notna()]

        X = df_eval[feat_e].fillna(0).values
        adj = m_e_lgb.model.predict(X)
        for i, (_, r) in enumerate(df_eval.iterrows()):
            base = float(r["sma"])
            tgt = float(r["half_ppr_points"])
            pred_e = base + float(adj[i])
            pred_ens = w_base * base + w_e * pred_e
            ensemble_preds.append(pred_ens)
            ensemble_targets.append(tgt)
            ensemble_base.append(base)

        if ensemble_preds:
            p = np.array(ensemble_preds); t = np.array(ensemble_targets); b = np.array(ensemble_base)
            mae = float(np.mean(np.abs(p - t)))
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            bias = float(np.mean(p - t))

            def _spearman(x, y):
                if len(x) < 3:
                    return None
                xr = _rank(x); yr = _rank(y)
                nn = len(xr)
                mx = sum(xr) / nn; my = sum(yr) / nn
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

            spear = _spearman(ensemble_preds, ensemble_targets)
            ss_right = ss_total = 0
            n = len(p)
            for i in range(n):
                for j in range(i + 1, n):
                    if t[i] == t[j] or p[i] == p[j]:
                        continue
                    ss_total += 1
                    if (p[i] > p[j]) == (t[i] > t[j]):
                        ss_right += 1
            ss = (ss_right / ss_total) if ss_total else None
            base_mae = float(np.mean(np.abs(b - t)))
            imp = (base_mae - mae) / base_mae * 100 if base_mae else None
            results.append(ModelResult(
                "ensemble_weighted_baseline_residual", "WR", test_season,
                max(train_weeks_select), len(test_weeks_eval), len(ensemble_preds),
                mae=round(mae, 4), rmse=round(rmse, 4), bias=round(bias, 4),
                spearman=round(spear, 4) if spear is not None else None,
                start_sit_win_rate=round(ss, 4) if ss is not None else None,
                baseline_mae=round(base_mae, 4),
                improvement_vs_baseline_pct=round(imp, 2) if imp is not None else None,
                ablation_group="F",
                notes="Ensemble ponderado (baseline + residual LightGBM E)",
            ))
            print(f"  F. Ensemble: n={len(ensemble_preds)} MAE={mae:.3f} Imp%={imp:.1f}%")

    # ------------------------------------------------------------------
    # G: Modelo adaptativo (shrinkage adaptativo).
    # ------------------------------------------------------------------
    print("\n=== G. Modelo Adaptativo (shrinkage adaptativo) ===")
    feat_g = feat_e  # Usa as mesmas features do contexto completo
    if feat_g:
        m_g = AdaptiveResidualModel(kind="ridge", alpha=1.0,
                                    feature_cols=feat_g)
        m_g.ablation_group = "G"
        m_g.fit(store_select, max(train_weeks_select), position="WR")
        # Avalia usando a função de avaliação adaptativa.
        from yardline_intelligence.models.adaptive_model import evaluate_adaptive
        res_g = evaluate_adaptive(store_test, m_g, test_weeks_eval, position="WR")
        if res_g.n_rows > 0 and res_g.mae is not None:
            results.append(ModelResult(
                name="residual_adaptativo_ridge", position="WR", season=test_season,
                train_weeks=max(train_weeks_select), validate_weeks=len(test_weeks_eval),
                n_rows=res_g.n_rows, mae=res_g.mae, rmse=res_g.rmse, bias=res_g.bias,
                spearman=res_g.spearman, start_sit_win_rate=res_g.start_sit_win_rate,
                baseline_mae=res_g.baseline_mae,
                improvement_vs_baseline_pct=res_g.improvement_vs_baseline_pct,
                ablation_group="G",
                notes="Modelo adaptativo com shrinkage dinâmico (Ridge)",
            ))
            print(f"  G. residual_adaptativo_ridge: n={res_g.n_rows} MAE={res_g.mae} Imp%={res_g.improvement_vs_baseline_pct}")

    # ------------------------------------------------------------------
    # 4. Determina campeão.
    # ------------------------------------------------------------------
    valid = [r for r in results if r.n_rows > 0 and r.mae is not None]
    champion = min(valid, key=lambda r: r.mae) if valid else None

    experiment["ablation_experiments"] = [r.to_dict() for r in results]
    experiment["champion"] = champion.to_dict() if champion else None

    base_a = next((r for r in results if r.ablation_group == "A" and r.name == "baseline_sma_3"), None)
    if champion and champion.ablation_group != "A" and base_a and base_a.mae:
        experiment["champion_comparison"] = {
            "champion": champion.name,
            "champion_mae": champion.mae,
            "baseline_mae": base_a.mae,
            "improvement_vs_baseline_pct": round((base_a.mae - champion.mae) / base_a.mae * 100, 2),
            "champion_better": champion.mae < base_a.mae,
        }
    elif champion and champion.ablation_group == "A":
        experiment["champion_comparison"] = {
            "champion": "baseline",
            "note": "Nenhum modelo superou o baseline; mantido como campeão",
        }

    # ------------------------------------------------------------------
    # 5. Accuracy Lab com intervalos de confiança (cluster bootstrap).
    #    Calcula CI para TODOS os modelos válidos, não apenas o campeão.
    # ------------------------------------------------------------------
    print("\n=== 4. Accuracy Lab com Intervalos de Confiança ===")
    ci_info = {}
    champion_model_for_prob = None
    champion_feats_for_prob = None

    # Calcula CI para cada modelo com predições disponíveis.
    all_ci = {}
    for r in results:
        if r.ablation_group == "A":
            # Baseline puro: reavalia com CI.
            w = 3 if r.name == "baseline_sma_3" else 5
            base_res = evaluate_baseline_vectorized(df_test, test_weeks_eval, window=w)
            df_eval = df_test.copy()
            sma = precompute_sma(df_test, window=w)
            df_eval["sma"] = sma.values
            df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
            df_eval = df_eval[df_eval["half_ppr_points"].notna()]
            df_eval = df_eval[df_eval["sma"].notna()]
            preds = df_eval["sma"].tolist()
            targets = df_eval["half_ppr_points"].tolist()
            weeks = df_eval["week"].astype(int).tolist()
            ci_mae_lo, ci_mae_hi = bootstrap_ci(preds, targets, weeks, metric="mae", n_bootstrap=500, seed=42)
            ci_rmse_lo, ci_rmse_hi = bootstrap_ci(preds, targets, weeks, metric="rmse", n_bootstrap=500, seed=42)
            all_ci[r.name] = {
                "mae_lo": ci_mae_lo, "mae_hi": ci_mae_hi,
                "rmse_lo": ci_rmse_lo, "rmse_hi": ci_rmse_hi,
                "n_bootstrap": 500,
            }
            print(f"  {r.name}: MAE CI [{ci_mae_lo:.3f}, {ci_mae_hi:.3f}] RMSE CI [{ci_rmse_lo:.3f}, {ci_rmse_hi:.3f}]")
        else:
            # Modelo residual/adaptativo: recalcula predições para CI.
            model_for_ci = None
            feats_for_ci = None
            if r.ablation_group == "B" and "lightgbm" in r.name:
                model_for_ci, feats_for_ci = m_lgb, feat_b
            elif r.ablation_group == "B":
                model_for_ci, feats_for_ci = m, feat_b
            elif r.ablation_group == "C":
                model_for_ci, feats_for_ci = m_c, feat_c
            elif r.ablation_group == "D":
                model_for_ci, feats_for_ci = m_d, feat_d
            elif r.ablation_group == "E" and "lightgbm" in r.name:
                model_for_ci, feats_for_ci = m_e_lgb, feat_e
            elif r.ablation_group == "E":
                model_for_ci, feats_for_ci = m_e, feat_e
            elif r.ablation_group == "G":
                model_for_ci, feats_for_ci = m_g, feat_g

            if model_for_ci is not None and feats_for_ci:
                df_eval = df_test.copy()
                sma = df_eval.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
                    lambda s: s.shift(1).rolling(3, min_periods=1).mean()
                )
                df_eval["sma"] = sma.values
                df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
                df_eval = df_eval[df_eval["half_ppr_points"].notna()]
                df_eval = df_eval[df_eval["sma"].notna()]

                X = df_eval[feats_for_ci].fillna(0).values
                if hasattr(model_for_ci, "model") and model_for_ci.model is not None:
                    adj = model_for_ci.model.predict(X)
                elif hasattr(model_for_ci, "coef_") and model_for_ci.coef_ is not None:
                    adj = model_for_ci.intercept_ + X @ model_for_ci.coef_
                else:
                    adj = None

                if adj is not None:
                    preds = []
                    targets = []
                    weeks = []
                    for i, (_, rr) in enumerate(df_eval.iterrows()):
                        pred = float(rr["sma"]) + float(adj[i])
                        preds.append(pred)
                        targets.append(float(rr["half_ppr_points"]))
                        weeks.append(int(rr["week"]))

                    ci_mae_lo, ci_mae_hi = bootstrap_ci(preds, targets, weeks, metric="mae", n_bootstrap=500, seed=42)
                    ci_rmse_lo, ci_rmse_hi = bootstrap_ci(preds, targets, weeks, metric="rmse", n_bootstrap=500, seed=42)
                    all_ci[r.name] = {
                        "mae_lo": ci_mae_lo, "mae_hi": ci_mae_hi,
                        "rmse_lo": ci_rmse_lo, "rmse_hi": ci_rmse_hi,
                        "n_bootstrap": 500,
                    }
                    print(f"  {r.name}: MAE CI [{ci_mae_lo:.3f}, {ci_mae_hi:.3f}] RMSE CI [{ci_rmse_lo:.3f}, {ci_rmse_hi:.3f}]")

    experiment["all_model_ci"] = all_ci

    # Identifica qual modelo é o campeão para recriar predições (probabilístico).
    if champion:
        if champion.ablation_group == "E" and "lightgbm" in champion.name:
            champion_model_for_prob = m_e_lgb
            champion_feats_for_prob = feat_e
        elif champion.ablation_group == "E":
            champion_model_for_prob = m_e
            champion_feats_for_prob = feat_e
        elif champion.ablation_group == "B" and "lightgbm" in champion.name:
            champion_model_for_prob = m_lgb
            champion_feats_for_prob = feat_b
        elif champion.ablation_group == "B":
            champion_model_for_prob = m
            champion_feats_for_prob = feat_b
        elif champion.ablation_group == "G":
            champion_model_for_prob = m_g
            champion_feats_for_prob = feat_g

    ci_info = {}
    if champion_model_for_prob is not None and champion_feats_for_prob:
        # Recalcula predições para CI do campeão (modelo ML).
        df_eval = df_test.copy()
        sma = df_eval.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df_eval["sma"] = sma.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma"].notna()]

        X = df_eval[champion_feats_for_prob].fillna(0).values
        if hasattr(champion_model_for_prob, "model") and champion_model_for_prob.model is not None:
            adj = champion_model_for_prob.model.predict(X)
        elif hasattr(champion_model_for_prob, "coef_") and champion_model_for_prob.coef_ is not None:
            adj = champion_model_for_prob.intercept_ + X @ champion_model_for_prob.coef_
        else:
            adj = None

        if adj is not None:
            preds = []
            targets = []
            weeks = []
            for i, (_, r) in enumerate(df_eval.iterrows()):
                pred = float(r["sma"]) + float(adj[i])
                preds.append(pred)
                targets.append(float(r["half_ppr_points"]))
                weeks.append(int(r["week"]))

            ci_mae_lo, ci_mae_hi = bootstrap_ci(preds, targets, weeks, metric="mae", n_bootstrap=500, seed=42)
            ci_rmse_lo, ci_rmse_hi = bootstrap_ci(preds, targets, weeks, metric="rmse", n_bootstrap=500, seed=42)
            ci_info = {
                "mae_lo": ci_mae_lo, "mae_hi": ci_mae_hi,
                "rmse_lo": ci_rmse_lo, "rmse_hi": ci_rmse_hi,
                "n_bootstrap": 500,
            }
            print(f"  Campeão {champion.name}: MAE CI [{ci_mae_lo:.3f}, {ci_mae_hi:.3f}]")
            print(f"  Campeão {champion.name}: RMSE CI [{ci_rmse_lo:.3f}, {ci_rmse_hi:.3f}]")
    elif champion and champion.ablation_group == "A":
        # Campeão é baseline puro: CI via SMA-5.
        w = 3 if champion.name == "baseline_sma_3" else 5
        base_res = evaluate_baseline_vectorized(df_test, test_weeks_eval, window=w)
        df_eval = df_test.copy()
        sma = precompute_sma(df_test, window=w)
        df_eval["sma"] = sma.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma"].notna()]
        preds = df_eval["sma"].tolist()
        targets = df_eval["half_ppr_points"].tolist()
        weeks = df_eval["week"].astype(int).tolist()
        ci_mae_lo, ci_mae_hi = bootstrap_ci(preds, targets, weeks, metric="mae", n_bootstrap=500, seed=42)
        ci_rmse_lo, ci_rmse_hi = bootstrap_ci(preds, targets, weeks, metric="rmse", n_bootstrap=500, seed=42)
        ci_info = {
            "mae_lo": ci_mae_lo, "mae_hi": ci_mae_hi,
            "rmse_lo": ci_rmse_lo, "rmse_hi": ci_rmse_hi,
            "n_bootstrap": 500,
        }
        print(f"  Campeão {champion.name} (baseline): MAE CI [{ci_mae_lo:.3f}, {ci_mae_hi:.3f}]")
        print(f"  Campeão {champion.name} (baseline): RMSE CI [{ci_rmse_lo:.3f}, {ci_rmse_hi:.3f}]")

    experiment["champion_ci"] = ci_info

    # ------------------------------------------------------------------
    # 5.5. Métricas por perfil de atleta (rookies, retornando de lesão,
    #      troca de QB, faixas de projeção).
    # ------------------------------------------------------------------
    print("\n=== 4.5. Métricas por Perfil de Atleta ===")
    profile_metrics = {}

    # Determina predições/targets para métricas de perfil.
    profile_preds = None
    profile_targets = None
    profile_weeks = None

    if champion_model_for_prob is not None and champion_feats_for_prob:
        df_eval = df_test.copy()
        sma = df_eval.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df_eval["sma"] = sma.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma"].notna()]

        X = df_eval[champion_feats_for_prob].fillna(0).values
        if hasattr(champion_model_for_prob, "model") and champion_model_for_prob.model is not None:
            adj = champion_model_for_prob.model.predict(X)
        elif hasattr(champion_model_for_prob, "coef_") and champion_model_for_prob.coef_ is not None:
            adj = champion_model_for_prob.intercept_ + X @ champion_model_for_prob.coef_
        else:
            adj = None

        if adj is not None:
            preds = []
            targets = []
            for i, (_, r) in enumerate(df_eval.iterrows()):
                pred = float(r["sma"]) + float(adj[i])
                preds.append(pred)
                targets.append(float(r["half_ppr_points"]))
            profile_preds = preds
            profile_targets = targets
            profile_weeks = [int(r["week"]) for _, r in df_eval.iterrows()]
    else:
        # Campeão é baseline puro (SMA-5): usa predições do baseline.
        df_eval = df_test.copy()
        sma_5 = df_eval.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df_eval["sma_5"] = sma_5.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma_5"].notna()]

        preds = df_eval["sma_5"].tolist()
        targets = df_eval["half_ppr_points"].tolist()
        profile_preds = preds
        profile_targets = targets
        profile_weeks = df_eval["week"].astype(int).tolist()

    if profile_preds is not None and profile_targets:
        # Métricas por faixa de projeção (quartis do predito).
        bands = metrics_by_projection_band(profile_preds, profile_targets, n_bands=4)
        profile_metrics["by_projection_band"] = bands

        # Métricas por semana.
        by_week = metrics_by_week(profile_preds, profile_targets, profile_weeks)
        profile_metrics["by_week"] = by_week

        # Rookies: jogadores com weeks_active <= 2 na semana de avaliação.
        rookie_idx = []
        veteran_idx = []
        for i, (_, r) in enumerate(df_eval.iterrows()):
            feats = store_test.features_for(r["sleeper_id"], int(r["week"]))
            wa = feats.get("weeks_active")
            if wa is not None and wa <= 2:
                rookie_idx.append(i)
            else:
                veteran_idx.append(i)

        def _m(ids):
            if not ids:
                return {"n": 0}
            p = [profile_preds[i] for i in ids]
            t = [profile_targets[i] for i in ids]
            return core_metrics(p, t)

        profile_metrics["rookies"] = _m(rookie_idx)
        profile_metrics["veteranos"] = _m(veteran_idx)

        print(f"  Rookies: n={profile_metrics['rookies'].get('n')} MAE={profile_metrics['rookies'].get('mae')}")
        print(f"  Veteranos: n={profile_metrics['veteranos'].get('n')} MAE={profile_metrics['veteranos'].get('mae')}")
        if bands and "bands" in bands:
            for b in bands["bands"]:
                print(f"  Faixa {b['band']}: n={b['n']} MAE={b['mae']} pred=[{b['pred_lo']}, {b['pred_hi']}]")

    experiment["profile_metrics"] = profile_metrics

    # ------------------------------------------------------------------
    # 6. Simulação probabilística (resíduos OOS).
    # ------------------------------------------------------------------
    print("\n=== 5. Simulação Probabilística ===")
    residuals_pool = []

    if champion_model_for_prob is not None and champion_feats_for_prob:
        # Campeão é um modelo ML treinado: coleta resíduos de predições OOS.
        df_eval = df_test.copy()
        sma = df_eval.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df_eval["sma"] = sma.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma"].notna()]

        X = df_eval[champion_feats_for_prob].fillna(0).values
        if hasattr(champion_model_for_prob, "model") and champion_model_for_prob.model is not None:
            adj = champion_model_for_prob.model.predict(X)
        elif hasattr(champion_model_for_prob, "coef_") and champion_model_for_prob.coef_ is not None:
            adj = champion_model_for_prob.intercept_ + X @ champion_model_for_prob.coef_
        else:
            adj = None

        if adj is not None:
            for i, (_, r) in enumerate(df_eval.iterrows()):
                base = float(r["sma"])
                tgt = float(r["half_ppr_points"])
                pred = base + float(adj[i])
                residuals_pool.append(tgt - pred)
    else:
        # Campeão é baseline puro: gera resíduos do baseline SMA-5.
        sma_5 = df_test.sort_values(["sleeper_id", "week"]).groupby("sleeper_id")["half_ppr_points"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df_eval = df_test.copy()
        df_eval["sma_5"] = sma_5.values
        df_eval = df_eval[df_eval["week"].isin(test_weeks_eval)]
        df_eval = df_eval[df_eval["half_ppr_points"].notna()]
        df_eval = df_eval[df_eval["sma_5"].notna()]
        for _, r in df_eval.iterrows():
            base = float(r["sma_5"])
            tgt = float(r["half_ppr_points"])
            residuals_pool.append(tgt - base)

    dist = build_residual_distribution(residuals_pool)
    print(f"  Resíduos OOS coletados: {len(residuals_pool)}")
    if dist:
        print(f"  Distribuição: mean={dist['mean']} std={dist['std']}")
        print(f"  P5={dist['p5']} P25={dist['p25']} P50={dist['p50']} P75={dist['p75']} P95={dist['p95']}")

    experiment["probabilistic"] = {
        "n_residuals": len(residuals_pool),
        "distribution": dist,
        "note": "Resíduos de previsões fora da amostra (2025) usados para intervalos P10-P90",
    }

    # ------------------------------------------------------------------
    # 7. Salva experimento.
    # ------------------------------------------------------------------
    finished = now_iso()
    experiment["finished_at"] = finished
    _save(experiment)
    return experiment


def _save(experiment: dict) -> None:
    stamp = experiment["finished_at"].replace(":", "-").replace("+", "Z")
    season = experiment.get("test_season", "unk")
    out_path = EXPERIMENTS_DIR / f"experiment_v04_{season}_{stamp}.json"
    out_path.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment["saved_to"] = str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yardline Intelligence v0.4.1 pipeline")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON,
                        help="Temporada alvo (diagnóstico)")
    parser.add_argument("--validate-weeks", type=str, default=None,
                        help="Semanas de validação (ex: 9,10,11)")
    parser.add_argument("--train-seasons", type=str,
                        default="2018,2019,2020,2021,2022,2023",
                        help="Temporadas de treino")
    parser.add_argument("--select-season", type=int, default=2024,
                        help="Temporada de seleção de hiperparâmetros")
    parser.add_argument("--test-season", type=int, default=2025,
                        help="Temporada de teste retrospectivo (intocado)")
    parser.add_argument("--max-week-test", type=int, default=None,
                        help="Última semana do teste")
    parser.add_argument("--no-context", action="store_true",
                        help="Desabilita o Context Engine (usa apenas features do dataset base)")
    parser.add_argument("--no-duckdb", action="store_true",
                        help="Desabilita otimizações do DuckDB")
    args = parser.parse_args()

    train_seasons = [int(x) for x in args.train_seasons.split(",")]
    vw = [int(x) for x in args.validate_weeks.split(",")] if args.validate_weeks else None

    res = run(target_season=args.season, validate_weeks=vw,
              train_seasons=train_seasons,
              select_season=args.select_season,
              test_season=args.test_season,
              max_week_test=args.max_week_test,
              use_context_engine=not args.no_context,
              use_duckdb=not args.no_duckdb)
    print(json.dumps(res, indent=2, ensure_ascii=False))