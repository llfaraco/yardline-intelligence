# Yardline Intelligence v0.4 — Relatório de Resultados

## Execução científica concluída

O pipeline v0.4 foi executado com sucesso utilizando dados reais do nflverse. Nenhum modelo de Machine Learning superou o baseline de média móvel simples (SMA-5), que permanece como campeão com **MAE = 2,625**.

## Resultados do backtest (temporada 2025, semanas 4-10)

| Modelo | n | MAE | RMSE | Bias | Spearman | Start/Sit | vs Baseline |
|--------|---|-----|------|------|----------|-----------|-------------|
| Baseline SMA-3 | 1.525 | 2,944 | 3,981 | -0,046 | 0,552 | 69,3% | — |
| Baseline SMA-5 (campeão) | 1.525 | 2,625 | 3,576 | -0,060 | — | 70,0% | — |
| Ridge (volume+eficiência) | 3.509 | 2.700 | 3.686 | -0,633 | 0,033 | 51,1% | -29,8% |
| LightGBM | 3.509 | 3,742 | 4,454 | +1,664 | 0,433 | 68,8% | -4,7% |
| XGBoost | 1.525 | 4.121 | 4,975 | +2,403 | 0,412 | 67,5% | -39,9% |

## Diagnóstico dos erros

- O VolumeModel superestima targets (intercept 8,68, predição média 23,1 vs real 15,4)
- O EfficiencyModel prediz ~0 (média 0,003 vs real 0,56)
- A composição volume × eficiência propaga erros de dois modelos

O baseline SMA-5 (MAE 2,625) é difícil de bater com apenas 8 semanas de treino (temporada 2026 ainda em andamento).

## Ambiente

- Python 3.15, DuckDB, pandas, PyArrow, scikit-learn, LightGBM, XGBoost, Parquet
- Ambiente isolado em `.venv/`
- Nenhuma modificação no frontend, banco operacional ou produção da Yardline
- Sem merge, push ou deploy

---

**Nota:** Este documento resume a primeira entrega científica do Yardline Intelligence. Para resultados mais detalhados, consulte os arquivos em `data/experiments/`.