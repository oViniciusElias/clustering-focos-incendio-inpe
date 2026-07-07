# Identification of Spatial Patterns in Wildfire Hotspots in Brazil (SDG 15)

Unsupervised spatial analysis comparing **K-Means**, **GMM**, and **DBSCAN**
on fire hotspots from INPE (BDQueimadas, 2025), aligned with Sustainable
Development Goal 15 (Life on Land).

## Overview

The feature vector combines **Latitude, Longitude, and FRP** (Fire Radiative
Power). The three algorithms are compared across **30 independent runs**
using the **Silhouette Coefficient** and execution time, with:

- **Kruskal-Wallis** hypothesis test over the 30 runs;
- **Parameter variation study** (`k` for K-Means, `eps` for DBSCAN);
- **Segmented analysis by biome** (Cerrado × Amazon) with `eps` recalibration;
- Exploratory figures (EDA) and results plots.

## Structure

```
codigo.V4.py                                  # main script
bdqueimadas_2025-01-01_2025-12-31.csv         # dataset (INPE)
project_report.pdf                            # paper (IEEE template, double column)
requirements.txt                              # dependencies
```

## How to Reproduce

1. Create and activate a virtual environment (optional, but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the CSV `bdqueimadas_2025-01-01_2025-12-31.csv` is in the same folder.
4. Run:
   ```bash
   python codigo.V4.py
   ```

## Reproducibility

- Random seeds are explicitly declared (`semente = i * 42`), ensuring the 30
  runs are deterministic.
- Algorithm fitting occurs on 80% of the dense base (preserving the density
  structure required by DBSCAN); silhouette computation uses internal subsampling
  of 5,000 points (`sample_size`) to manage the O(n²) cost.
- Hyperparameters: K-Means `k=8`; GMM `n_components=8`, covariance `full`;
  DBSCAN `eps=0.30`, `MinPts=15`.

## Generated Outputs (PNG)

| File | Content |
|---|---|
| `eda_frp_hist.png` | FRP histogram (log scale) |
| `eda_focos_bioma.png` | Hotspot map by biome |
| `param_kmeans_k.png` | Silhouette × `k` (K-Means) |
| `param_dbscan_eps.png` | Silhouette and noise × `eps` (DBSCAN) |
| `boxplot_silhueta.png` | Silhouette distribution across 30 runs |
| `scatter_dbscan.png` | DBSCAN spatial clusters |

## Data Source

INPE — Queimadas Program / TerraBrasilis: https://terrabrasilis.dpi.inpe.br/

---

---


# Identificação de Padrões Espaciais em Focos de Incêndio no Brasil (ODS 15)

Análise espacial não supervisionada comparando **K-Means**, **GMM** e **DBSCAN**
sobre focos de queimadas do INPE (BDQueimadas, 2025), alinhada ao Objetivo de
Desenvolvimento Sustentável 15 (Vida Terrestre).

## Visão geral

O vetor de atributos combina **Latitude, Longitude e FRP** (Potência Radiativa do
Fogo). Os três algoritmos são comparados ao longo de **30 execuções independentes**
pelo **Coeficiente de Silhueta** e pelo tempo de execução, com:

- Teste de hipótese **Kruskal-Wallis** sobre as 30 execuções;
- Estudo de **variação de parâmetros** (`k` no K-Means, `eps` no DBSCAN);
- Análise **segmentada por bioma** (Cerrado × Amazônia) com recalibração de `eps`;
- Figuras exploratórias (EDA) e gráficos de resultados.

## Estrutura

```
analise_clusters.py                           # script principal
bdqueimadas_2025-01-01_2025-12-31.csv         # base de dados (INPE)
relatorio.pdf                                 # artigo (template IEEE, dupla coluna)
requirements.txt                              # dependências
```

## Como reproduzir

1. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Garanta que o CSV `bdqueimadas_2025-01-01_2025-12-31.csv` esteja na mesma pasta.
4. Execute:
   ```bash
   python codigo.V4.py
   ```

## Reprodutibilidade

- As sementes aleatórias (`random seeds`) são declaradas explicitamente
  (`semente = i * 42`), garantindo que as 30 execuções sejam determinísticas.
- O `fit` dos algoritmos ocorre sobre 80% da base densa (preservando a densidade
  exigida pelo DBSCAN); o cálculo da silhueta usa subamostragem interna de 5.000
  pontos (`sample_size`) para viabilizar o custo O(n²).
- Hiperparâmetros: K-Means `k=8`; GMM `n_components=8`, covariância `full`;
  DBSCAN `eps=0.30`, `MinPts=15`.

## Saídas geradas (PNG)

| Arquivo | Conteúdo |
|---|---|
| `eda_frp_hist.png` | Histograma da FRP (escala log) |
| `eda_focos_bioma.png` | Mapa dos focos por bioma |
| `param_kmeans_k.png` | Silhueta × `k` (K-Means) |
| `param_dbscan_eps.png` | Silhueta e ruído × `eps` (DBSCAN) |
| `boxplot_silhueta.png` | Distribuição da silhueta nas 30 execuções |
| `scatter_dbscan.png` | Clusters espaciais do DBSCAN |

## Fonte dos dados

INPE — Programa Queimadas / TerraBrasilis: https://terrabrasilis.dpi.inpe.br/
