# -*- coding: utf-8 -*-
# Projeto de IA - focos de incendio no Brasil (ODS 15)
# Universidade Federal de São Paulo - ICT
# Compara K-Means, GMM e DBSCAN usando Latitude, Longitude e FRP.
#
# O script faz a EDA, roda o experimento das 30 execucoes, aplica o
# Kruskal-Wallis, varia os parametros (k e eps), separa por bioma e no
# fim salva os graficos (eda_frp_hist, eda_focos_bioma, param_kmeans_k,
# param_dbscan_eps, boxplot_silhueta, scatter_dbscan).

import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # sem janela, so salva os png
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal

warnings.filterwarnings('ignore')

# parametros
ATRIBUTOS = ['Latitude', 'Longitude', 'FRP']
N_EXECUCOES = 30
FRACAO_AMOSTRA = 0.8       # 80% da base por rodada
SAMPLE_SILHUETA = 5000     # silhueta e O(n^2), entao subamostro
EPS_DBSCAN = 0.30          # ajustado pro espaco 3D (Lat/Long/FRP)
MINPTS_DBSCAN = 15
K_CLUSTERS = 8


# ---- 1. carrega e limpa os dados ----
print("Carregando base de dados e limpando valores ausentes...")
df = pd.read_csv('bdqueimadas_2025-01-01_2025-12-31.csv')

# o csv vem com -999 onde falta valor
df = df.replace([-999, -999.0], np.nan)

df = df.dropna(subset=ATRIBUTOS)
X = df[ATRIBUTOS].values

print(f"Base carregada! Total de instancias validas (Lat/Long/FRP): {len(X)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---- 2. EDA ----
print("\nGerando figuras exploratorias (EDA)...")

# histograma do FRP - uso log no y porque a distribuicao e bem assimetrica
plt.figure(figsize=(8, 6))
sns.histplot(df['FRP'], bins=60, color='firebrick')
plt.yscale('log')
plt.title('Distribuicao da Potencia Radiativa do Fogo (FRP)')
plt.xlabel('FRP (MW)')
plt.ylabel('Frequencia (escala log)')
plt.tight_layout()
plt.savefig('eda_frp_hist.png', dpi=300)
plt.close()

# mapa dos focos por bioma - pego uma amostra so pra nao pesar o plot
np.random.seed(0)
n_eda = min(15000, len(df))
df_eda = df.sample(n=n_eda, random_state=0)
plt.figure(figsize=(8, 7))
sns.scatterplot(data=df_eda, x='Longitude', y='Latitude', hue='Bioma',
                s=8, alpha=0.5, palette='tab10')
plt.title('Distribuicao Geografica dos Focos de Incendio por Bioma (2025)')
plt.legend(markerscale=2, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('eda_focos_bioma.png', dpi=300)
plt.close()
print("Figuras 'eda_frp_hist.png' e 'eda_focos_bioma.png' salvas.")


# ---- 3. funcoes auxiliares ----
def silhueta_subamostrada(X, labels, semente):
    """Calcula a silhueta subamostrando os pontos (senao fica O(n^2) caro).

    Tiro os pontos de ruido (-1) do DBSCAN antes de calcular, porque o ruido
    nao e um cluster de verdade e baguncaria a metrica. K-Means e GMM nao tem
    -1, entao pra eles nao muda nada.

    Faco a subamostragem na mao em vez de usar o sample_size do sklearn pra
    garantir que nenhum cluster valido fique de fora da amostra (ja aconteceu
    do DBSCAN ter um cluster gigante e a amostra pegar so um rotulo).
    """
    labels = np.asarray(labels)
    mask = labels != -1
    X_val = X[mask]
    labels_val = labels[mask]

    # silhueta precisa de pelo menos 2 clusters (fora o ruido)
    clusters = np.unique(labels_val)
    if len(clusters) <= 1:
        return 0.0

    if len(X_val) > SAMPLE_SILHUETA:
        rng = np.random.RandomState(semente)
        idx = rng.choice(len(X_val), size=SAMPLE_SILHUETA, replace=False)
        # garante que todo cluster valido apareca ao menos uma vez
        presentes = set(labels_val[idx])
        faltantes = [c for c in clusters if c not in presentes]
        if faltantes:
            extra = [rng.choice(np.where(labels_val == c)[0]) for c in faltantes]
            idx = np.concatenate([idx, np.array(extra, dtype=int)])
        X_val = X_val[idx]
        labels_val = labels_val[idx]

    return silhouette_score(X_val, labels_val)


def dbscan_seguro(X, eps, min_samples):
    """DBSCAN protegido de MemoryError (eps grande estoura a memoria)."""
    try:
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    except MemoryError:
        return None


def roda_algoritmos(X_fit, semente):
    """Roda os 3 algoritmos em X_fit e devolve labels, silhueta e tempo."""
    saida = {}

    t0 = time.time()
    kmeans = KMeans(n_clusters=K_CLUSTERS, init='k-means++',
                    random_state=semente, n_init=10)
    labels_km = kmeans.fit_predict(X_fit)
    saida['K-Means'] = (labels_km, silhueta_subamostrada(X_fit, labels_km, semente),
                        time.time() - t0)

    t0 = time.time()
    gmm = GaussianMixture(n_components=K_CLUSTERS, covariance_type='full',
                          random_state=semente)
    labels_gmm = gmm.fit_predict(X_fit)
    saida['GMM'] = (labels_gmm, silhueta_subamostrada(X_fit, labels_gmm, semente),
                    time.time() - t0)

    t0 = time.time()
    dbscan = DBSCAN(eps=EPS_DBSCAN, min_samples=MINPTS_DBSCAN)
    labels_db = dbscan.fit_predict(X_fit)
    saida['DBSCAN'] = (labels_db, silhueta_subamostrada(X_fit, labels_db, semente),
                       time.time() - t0)

    return saida


# ---- 4. experimento principal: 30 execucoes ----
resultados = {'Algoritmo': [], 'Silhueta': [], 'Tempo': []}
X_amostra_final = None
labels_db_final = None

print(f"\nIniciando as {N_EXECUCOES} execucoes com reamostragem...")
for i in range(N_EXECUCOES):
    semente = i * 42
    np.random.seed(semente)

    # reamostra 80% da base inteira (mantem a densidade que o DBSCAN precisa)
    n_amostra = int(FRACAO_AMOSTRA * len(X_scaled))
    indices_amostra = np.random.choice(len(X_scaled), size=n_amostra, replace=False)
    X_amostra = X_scaled[indices_amostra]

    saida = roda_algoritmos(X_amostra, semente)
    for alg, (labels, sil, tempo) in saida.items():
        resultados['Algoritmo'].append(alg)
        resultados['Silhueta'].append(sil)
        resultados['Tempo'].append(tempo)

    X_amostra_final = X_amostra
    labels_db_final = saida['DBSCAN'][0]

res_df = pd.DataFrame(resultados)

# resumo do DBSCAN da ultima rodada
n_clusters_db = len(set(labels_db_final) - {-1})
perc_ruido_db = float(np.mean(labels_db_final == -1) * 100)


# ---- 5. estatisticas finais ----
print("\n=== ESTATISTICAS FINAIS (30 EXECUCOES) ===")
for alg in ['K-Means', 'GMM', 'DBSCAN']:
    subset = res_df[res_df['Algoritmo'] == alg]
    print(f"\n--- {alg} ---")
    print(f"Silhueta -> Media {subset['Silhueta'].mean():.4f} | "
          f"Mediana {subset['Silhueta'].median():.4f} | "
          f"DP {subset['Silhueta'].std():.4f}")
    print(f"Tempo(s) -> Media {subset['Tempo'].mean():.4f} | "
          f"Mediana {subset['Tempo'].median():.4f} | "
          f"DP {subset['Tempo'].std():.4f}")

print(f"\nDBSCAN (ultima rodada): {n_clusters_db} clusters | "
      f"{perc_ruido_db:.1f}% de ruido")


# ---- 6. teste de hipotese (Kruskal-Wallis na silhueta) ----
sil_km = res_df.loc[res_df['Algoritmo'] == 'K-Means', 'Silhueta']
sil_gmm = res_df.loc[res_df['Algoritmo'] == 'GMM', 'Silhueta']
sil_db = res_df.loc[res_df['Algoritmo'] == 'DBSCAN', 'Silhueta']

stat, p_val = kruskal(sil_km, sil_gmm, sil_db)
print("\n=== TESTE DE HIPOTESE (Kruskal-Wallis) ===")
print(f"Estatistica H: {stat:.4f}")
print(f"p-value: {p_val:.4e}")
if p_val < 0.05:
    print("Diferenca estatisticamente significativa entre os algoritmos (p < 0.05).")
else:
    print("Sem diferenca estatisticamente significativa (p >= 0.05).")


# ---- 7. variacao de parametros ----
print("\n=== ESTUDO DE VARIACAO DE PARAMETROS ===")

# amostra fixa pros testes (80%, semente 42)
np.random.seed(42)
idx_par = np.random.choice(len(X_scaled), size=int(FRACAO_AMOSTRA * len(X_scaled)),
                           replace=False)
X_par = X_scaled[idx_par]

# 7.1 K-Means variando k
print("\nK-Means: silhueta em funcao de k")
ks = list(range(2, 13))
sil_por_k = []
for k in ks:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    lab = km.fit_predict(X_par)
    s = silhueta_subamostrada(X_par, lab, 42)
    sil_por_k.append(s)
    print(f"  k={k:2d} -> silhueta {s:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(ks, sil_por_k, marker='o', color='steelblue')
plt.title('K-Means: Coeficiente de Silhueta em funcao de k')
plt.xlabel('Numero de clusters (k)')
plt.ylabel('Coeficiente de Silhueta')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('param_kmeans_k.png', dpi=300)
plt.close()

# 7.2 DBSCAN variando eps
# fica limitado a 0.30: acima disso os clusters viram um grupo so e o
# grafo de vizinhanca tende a estourar a memoria com ~109k pontos.
print("\nDBSCAN: silhueta, n_clusters e ruido em funcao de eps")
eps_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
sil_por_eps, ruido_por_eps, nclus_por_eps, eps_validos = [], [], [], []
for eps in eps_grid:
    lab = dbscan_seguro(X_par, eps, MINPTS_DBSCAN)
    if lab is None:
        print(f"  eps={eps:.2f} -> ignorado (MemoryError)")
        continue
    s = silhueta_subamostrada(X_par, lab, 42)
    ruido = float(np.mean(lab == -1) * 100)
    nclus = len(set(lab) - {-1})
    eps_validos.append(eps)
    sil_por_eps.append(s)
    ruido_por_eps.append(ruido)
    nclus_por_eps.append(nclus)
    print(f"  eps={eps:.2f} -> silhueta {s:.4f} | clusters {nclus:3d} | ruido {ruido:4.1f}%")

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(eps_validos, sil_por_eps, marker='o', color='seagreen', label='Silhueta')
ax1.set_xlabel('Raio de vizinhanca (eps)')
ax1.set_ylabel('Coeficiente de Silhueta', color='seagreen')
ax1.tick_params(axis='y', labelcolor='seagreen')
ax2 = ax1.twinx()
ax2.plot(eps_validos, ruido_por_eps, marker='s', linestyle='--', color='darkorange',
         label='Ruido (%)')
ax2.set_ylabel('Pontos classificados como ruido (%)', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
plt.title('DBSCAN: influencia de eps na silhueta e no ruido')
fig.tight_layout()
plt.savefig('param_dbscan_eps.png', dpi=300)
plt.close()
print("Figuras 'param_kmeans_k.png' e 'param_dbscan_eps.png' salvas.")


# ---- 8. analise por bioma (Cerrado x Amazonia) ----
print("\n=== ANALISE SEGMENTADA POR BIOMA ===")
linhas_bioma = []
recalibracao = []
for bioma in ['Cerrado', 'Amazônia']:
    df_bioma = df[df['Bioma'] == bioma]
    if len(df_bioma) < MINPTS_DBSCAN:
        print(f"Bioma {bioma}: instancias insuficientes.")
        continue

    X_b = StandardScaler().fit_transform(df_bioma[ATRIBUTOS].values)

    # (a) com a config global (eps = 0.30)
    saida_bioma = roda_algoritmos(X_b, semente=42)
    for alg, (_, sil, tempo) in saida_bioma.items():
        linhas_bioma.append({'Bioma': bioma, 'Algoritmo': alg,
                             'Silhueta': sil, 'Tempo': tempo,
                             'Instancias': len(df_bioma)})

    # (b) procura o melhor eps pra cada bioma
    melhor_eps, melhor_sil = None, -2.0
    for eps in [0.15, 0.20, 0.25, 0.30, 0.40]:
        lab = dbscan_seguro(X_b, eps, MINPTS_DBSCAN)
        if lab is None:
            continue
        nclus = len(set(lab) - {-1})
        ruido = float(np.mean(lab == -1) * 100)
        if nclus > 1 and ruido < 50:
            s = silhueta_subamostrada(X_b, lab, 42)
            if s > melhor_sil:
                melhor_sil, melhor_eps = s, eps
    recalibracao.append({'Bioma': bioma, 'eps_otimo': melhor_eps,
                         'Silhueta_DBSCAN': melhor_sil})

bioma_df = pd.DataFrame(linhas_bioma)
if not bioma_df.empty:
    print("\nConfiguracao global (eps = 0.30):")
    print(bioma_df.to_string(index=False,
                             formatters={'Silhueta': '{:.4f}'.format,
                                         'Tempo': '{:.4f}'.format}))

recal_df = pd.DataFrame(recalibracao)
if not recal_df.empty:
    print("\nRecalibracao de eps por bioma (DBSCAN):")
    print(recal_df.to_string(index=False,
                             formatters={'Silhueta_DBSCAN': '{:.4f}'.format}))


# ---- 9. graficos finais (boxplot e scatter) ----
# boxplot da silhueta das 30 execucoes
plt.figure(figsize=(8, 6))
sns.boxplot(x='Algoritmo', y='Silhueta', data=res_df, palette='Set2')
plt.title('Comparacao do Coeficiente de Silhueta (30 Execucoes)')
plt.ylabel('Coeficiente de Silhueta')
plt.tight_layout()
plt.savefig('boxplot_silhueta.png', dpi=300)
plt.close()

# scatter dos clusters do DBSCAN (ultima rodada)
np.random.seed(0)
n_plot = min(8000, len(X_amostra_final))
idx_plot = np.random.choice(len(X_amostra_final), size=n_plot, replace=False)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_amostra_final[idx_plot, 1], y=X_amostra_final[idx_plot, 0],
                hue=labels_db_final[idx_plot], palette='viridis', legend=False, s=10)
plt.title('Clusters Espaciais do DBSCAN (Longitude vs Latitude)')
plt.xlabel('Longitude Padronizada')
plt.ylabel('Latitude Padronizada')
plt.tight_layout()
plt.savefig('scatter_dbscan.png', dpi=300)
plt.close()

print("\nTodos os graficos foram salvos com sucesso!")
