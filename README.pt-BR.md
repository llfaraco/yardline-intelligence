# Yardline Intelligence

**Pesquisa de projeção de desempenho da NFL para Fantasy Football | Ciência de Dados · Engenharia de Dados · Machine Learning**

O Yardline Intelligence é um projeto independente de pesquisa que investiga projeções semanais de desempenho de jogadores da NFL para Fantasy Football. O projeto combina dados públicos da NFL, engenharia de atributos temporais, variáveis contextuais, modelos estatísticos de referência, experimentos de machine learning e avaliação orientada à decisão.

> **Status:** protótipo de pesquisa. Código migrado da branch original de desenvolvimento da Yardline (v0.4.1). O escopo inicial é wide receivers (WR). Os resultados históricos estão preservados, mas a reprodução completa e algumas correções metodológicas ainda são necessárias antes de considerar o modelo validado.

[English README](README.md) · [Arquitetura](docs/architecture.md) · [Metodologia](docs/methodology.md) · [Resultados](docs/experiments/v0.4-results.md) · [Limitações conhecidas](docs/known-limitations.md)

## Pergunta de pesquisa

Features temporais e contextuais de jogadores e partidas conseguem melhorar as projeções semanais de pontos Half-PPR em relação a baselines simples de desempenho recente?

O projeto segue o princípio *baseline-first*: um modelo complexo só deve ser promovido quando demonstrar ganho confiável em observações fora da amostra.

## Arquitetura

| Camada | Responsabilidade |
|---|---|
| Data Foundation | Ingestão de play-by-play, snap counts, relatórios de lesão, jogos e IDs canônicos; rastreabilidade de fontes e SHA-256. |
| Temporal Feature Store | Registros jogador-temporada-semana, janelas móveis e features defasadas. |
| Feature Engineering | Volume, utilização, eficiência, consistência e tendências. |
| Context Engine | Lesões/disponibilidade, quarterback, linha ofensiva/defesa e contexto de jogo. |
| Modelagem e Simulação | Baselines SMA, modelos residuais Ridge, shrinkage adaptativo, LightGBM e XGBoost; distribuições preditivas baseadas em resíduos. |
| Accuracy Lab | MAE, RMSE, bias, Spearman, Start/Sit, decision regret, cobertura de intervalos e bootstrap. |

## Resultado reportado do v0.4.1

O experimento arquivado contém 3.509 observações jogador-semana da temporada 2025.

| Modelo | MAE | RMSE | Bias | Start/Sit |
|---|---:|---:|---:|---:|
| **Baseline SMA-5 (campeão reportado)** | **2,6249** | **3,5757** | **-0,0601** | **69,98%** |
| Baseline SMA-3 | 2,6973 | 3,7075 | -0,0271 | 69,48% |
| Ridge adaptativo residual | 2,6619 | 3,6520 | 0,0157 | 69,70% |
| Ridge residual | 2,6995 | 3,6855 | 0,1553 | 69,43% |
| LightGBM residual | 2,7468 | 3,7422 | 0,1664 | 68,75% |

O Ridge adaptativo apresenta 1,31% de melhora sobre o SMA-3 no campo de comparação arquivado, mas seu MAE continua acima do SMA-5. Portanto, o SMA-5 permanece como o modelo de menor erro absoluto nesse experimento.

**Integridade científica:** esses números são resultados reportados da execução local original, ainda não reproduzidos de forma independente a partir deste repositório. Os JSONs originais foram preservados em `data/experiments/`.

## Ressalvas metodológicas importantes

A revisão da migração encontrou pontos que precisam ser resolvidos antes de descrever o benchmark como uma avaliação plenamente reproduzível com treino 2018–2023, seleção 2024 e teste 2025:

1. O pipeline constrói o dataframe histórico de treino, mas o caminho atual de ablação ajusta os modelos usando o store da temporada de seleção de 2024.
2. A engenharia de features multi-temporada agrupava janelas móveis apenas por jogador. A migração torna o agrupamento sensível à temporada e adiciona um teste de regressão, mas a suíte completa ainda não foi executada após a migração.
3. O JSON mais recente do v0.4.1 registra `n_residuals = 0` na saída probabilística. Portanto, não alegamos intervalos preditivos calibrados.
4. Algumas rotinas dos modelos convertem features ausentes em zero. É necessário definir uma estratégia explícita de imputação, ajustada somente no treino, e indicadores de ausência.
5. O percentual do Ridge adaptativo compara com SMA-3, não demonstra superioridade sobre SMA-5.

Veja [limitações conhecidas](docs/known-limitations.md). Os pontos estão documentados para manter a transparência da pesquisa.

## Fontes de dados

- [nflverse data](https://github.com/nflverse/nflverse-data): play-by-play, snap counts e injury reports.
- [nflverse games data](https://github.com/nflverse/nfldata): calendário e metadados de jogos.
- [DynastyProcess player IDs](https://github.com/dynastyprocess/data): mapeamento canônico GSIS/Sleeper.

Os CSVs brutos, bancos DuckDB locais e Parquets gerados não são versionados no Git. Use o script de preparação para baixar as fontes públicas e reconstruir os datasets. Os JSONs de experimentos e relatórios Markdown são preservados.

## Como executar

Requer Python 3.12 ou superior, conforme a configuração do projeto.

### 1. Criar ambiente

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Baixar dados e gerar Parquets

O conjunto completo de 2018–2025 é grande e pode levar algum tempo para baixar.

```bash
python scripts/prepare_data.py --seasons 2018,2019,2020,2021,2022,2023,2024,2025
```

### 3. Executar o pipeline retrospectivo

```bash
python -m yardline_intelligence.pipeline --train-seasons 2018,2019,2020,2021,2022,2023 --select-season 2024 --test-season 2025
```

### 4. Executar os testes

```bash
python -m pytest tests/ -q
```

A entrega original reportou 12 testes aprovados antes da migração. Este repositório adiciona um teste de regressão para fronteira entre temporadas e isola a saída Parquet dos testes, mas a suíte completa ainda **não foi executada de forma independente após a migração**.

## Relação com a Yardline Football

Este repositório é o projeto independente de pesquisa e portfólio. A aplicação de produção Yardline Football permanece em uma base de código separada. Nenhum modelo deste repositório deve ser considerado pronto para produção ou conectado a ligas reais sem integração e validação próprias.

Yardline Intelligence não é um produto oficial da NFL e não possui afiliação com a NFL ou com o nflverse.
