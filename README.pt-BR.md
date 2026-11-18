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

## Snapshot da validação mais recente do v0.4.1

A branch de origem recebeu uma correção adicional em 18/11/2026: [`24c4f315`](https://github.com/llfaraco/yardline/commit/24c4f315be9852fa1afa584173ee5ded34cbfb0d). O resumo abaixo foi fornecido pelo responsável pelo projeto; o JSON correspondente a essa nova execução ainda não foi versionado na branch de origem.

| Experimento / modelo | MAE | Spearman | Observação |
|---|---:|---:|---|
| **Baseline SMA-5 (campeão)** | **2,6249** | **0,5666** | Continua com o menor MAE reportado |
| B: volume + eficiência Ridge | 2,6995 | — | Baseline residual sem contexto adicional |
| C: contexto de lesões | 2,6967 | — | Features contextuais agora entram no vetor |
| D: QB + linha ofensiva | 2,7167 | — | Mais contexto não reduziu o MAE |
| G: Ridge adaptativo | 2,67 | 0,562 | Não superou o SMA-5 |

Outros diagnósticos reportados:

- 3.509 resíduos out-of-sample coletados para a distribuição probabilística quando o campeão é o SMA-5.
- Intervalo de MAE por cluster bootstrap reportado na validação isolada: [2,577, 2,774].
- Métricas por perfil: grupo de pouco histórico rotulado como “rookies”, MAE 2,1466 (n=307); veteranos, MAE 2,7791 (n=1.486). No código atual, “rookie” é uma aproximação baseada em `weeks_active <= 2`, não uma identificação oficial de novato da NFL.
- As semanas 9–11 de 2025 estão presentes (230, 225 e 247 linhas). O gap reportado de 994 observações foi atribuído a jogadores sem histórico anterior suficiente para um baseline válido, não à ausência de semanas.
- O caminho do ensemble ponderado foi executado; o resumo recebido não informa sua métrica.

Os JSONs originais de v0.3/v0.4 continuam preservados em `data/experiments/`. O artefato JSON da execução pós-correção ainda precisa ser adicionado para permitir rastreabilidade independente dos novos resultados.

## Ressalvas metodológicas importantes

A revisão da migração e do commit mais recente encontrou pontos que precisam ser resolvidos antes de descrever o benchmark como uma avaliação plenamente reproduzível com treino 2018–2023, seleção 2024 e teste 2025:

1. O pipeline constrói o dataframe histórico de treino, mas o caminho atual de ablação ajusta os modelos usando o store da temporada de seleção de 2024.
2. A engenharia de features multi-temporada agrupava janelas móveis apenas por jogador. A migração torna o agrupamento sensível à temporada e adiciona um teste de regressão, mas a suíte completa ainda não foi executada após a migração.
3. A execução mais recente reporta 3.509 resíduos out-of-sample e uma distribuição empírica. Isso ainda não comprova intervalos calibrados por jogador: o pipeline registra a distribuição, mas não executa o avaliador de calibração; a função de forecast associa os percentis 5/95 a campos chamados P10/P90; e o cálculo rotulado CRPS continua sendo uma aproximação.
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

O responsável pelo projeto reporta que os 12 testes originais passam após as correções do v0.4.1. Este repositório standalone contém 16 testes: os 12 originais e quatro testes adicionais de migração/regressão. A suíte completa ainda **precisa passar por uma execução limpa no CI**.

## Relação com a Yardline Football

Este repositório é o projeto independente de pesquisa e portfólio. A aplicação de produção Yardline Football permanece em uma base de código separada. Nenhum modelo deste repositório deve ser considerado pronto para produção ou conectado a ligas reais sem integração e validação próprias.

Yardline Intelligence não é um produto oficial da NFL e não possui afiliação com a NFL ou com o nflverse.
