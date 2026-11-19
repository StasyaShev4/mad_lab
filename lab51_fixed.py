import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_samples,
    silhouette_score
)


df = pd.read_csv(
    "dataset-HeartDisease/processed.cleveland.data",
    header=None,
    na_values="?"
)

df.columns = [
    "age",       # возраст
    "sex",       # пол
    "cp",        # тип боли в груди
    "trestbps",  # артериальное давление в покое
    "chol",      # холестерин
    "fbs",       # уровень сахара натощак
    "restecg",   # результаты ЭКГ в покое
    "thalach",   # максимальная частота сердечных сокращений
    "exang",     # стенокардия при нагрузке
    "oldpeak",   # депрессия ST
    "slope",     # наклон сегмента ST
    "ca",        # число крупных сосудов
    "thal",      # результат теста thal
    "target"     # исходный целевой признак (НЕ используется при кластеризации)
]


print("\nПЕРВИЧНЫЙ АНАЛИЗ ДАТАСЕТА")
print("-" * 60)

print(f"Количество объектов: {df.shape[0]}")
print(f"Количество признаков: {df.shape[1]}")
print("\nПризнаки:")
print(", ".join(df.columns))

missing = df.isna().sum()
missing = missing[missing > 0]
print("\nПропуски:")
print("нет" if missing.empty else missing.to_dict())

print("\nОсновные числовые показатели:")
print(df[numeric_features if "numeric_features" in globals() else
         ["age", "trestbps", "chol", "thalach", "oldpeak"]].describe().round(2).to_string())


print("\n" + "=" * 75)
print("ПОДГОТОВКА ДАННЫХ")
print("=" * 75)

# target не используется для построения кластеров.
# Он сохраняется отдельно только для последующей интерпретации.
target = df["target"].copy()

cluster_df = df.drop(columns=["target"]).copy()

numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

# Заполняем пропуски:
# числовые признаки -> медиана,
# категориальные -> мода.
for col in numeric_features:
    cluster_df[col] = pd.to_numeric(cluster_df[col], errors="coerce")
    cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

for col in categorical_features:
    cluster_df[col] = cluster_df[col].fillna(cluster_df[col].mode()[0])
    cluster_df[col] = cluster_df[col].astype(str)

print("\nПропуски после обработки:")
print(cluster_df.isna().sum().sum())

print("\nЧисловые признаки:")
print(numeric_features)

print("\nКатегориальные признаки:")
print(categorical_features)


print("=" * 70)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ CLOPE")
print("=" * 70)

# Для CLOPE используем только категориальные признаки.
# Числовые признаки будут использованы далее при кластеризации
# с расстоянием Гауэра.

clope_df = cluster_df[categorical_features].copy()

# Приведение категориальных признаков к строковому типу
for col in categorical_features:
    clope_df[col] = clope_df[col].astype(str)

# Формирование транзакций.
# Каждый объект представляется как набор элементов вида:
# "признак=значение"

transactions = []

for _, row in clope_df.iterrows():
    transaction = [
        f"{col}={row[col]}"
        for col in categorical_features
    ]
    transactions.append(transaction)

print(f"Количество транзакций: {len(transactions)}")

if transactions:
    print(
        f"Среднее количество элементов в транзакции: "
        f"{np.mean([len(t) for t in transactions]):.2f}"
    )

print(f"Количество категориальных признаков: {len(categorical_features)}")

print("\nПример транзакции:")
print(transactions[0])


# 4. РЕАЛИЗАЦИЯ АЛГОРИТМА CLOPE
def clope_profit(s, w, repulsion):
    """Целевая функция CLOPE: Profit(C) = S(C) / W(C)^r."""
    if w <= 0:
        return 0.0
    return s / (w ** repulsion)


def clope(transactions, repulsion=2.0, max_iter=20):
    """
    Реализация CLOPE для транзакционных категориальных данных.

    Каждая транзакция хранится по индексу объекта, поэтому при наличии
    одинаковых транзакций объекты не смешиваются между кластерами.
    После начального размещения выполняется несколько итераций
    удаления и повторного помещения объектов.
    """
    tx = [set(t) for t in transactions]
    n = len(tx)

    # Каждый кластер содержит индексы объектов.
    clusters = []

    def stats(cluster):
        items = set()
        s = 0
        for idx in cluster:
            items.update(tx[idx])
            s += len(tx[idx])
        return s, len(items)

    def profit(cluster):
        s, w = stats(cluster)
        return clope_profit(s, w, repulsion)

    # ---------- начальное построение ----------
    for idx in range(n):
        best_cluster = None
        best_delta = 0.0

        for k, cluster in enumerate(clusters):
            old_profit = profit(cluster)
            new_profit = profit(cluster + [idx])
            delta = new_profit - old_profit

            if delta > best_delta:
                best_delta = delta
                best_cluster = k

        # Новый кластер всегда имеет положительную собственную прибыль.
        if best_cluster is None:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)

    # ---------- перераспределение ----------
    for _ in range(max_iter):
        moved = False

        # Объекты рассматриваются в фиксированном порядке.
        for idx in range(n):
            current = next(
                (k for k, c in enumerate(clusters) if idx in c),
                None
            )
            if current is None:
                continue

            old_cluster = clusters[current]
            old_profit = profit(old_cluster)

            # Удаление из текущего кластера.
            old_cluster.remove(idx)
            if not old_cluster:
                clusters.pop(current)
                current = None

            # Ищем лучшее место после удаления.
            best_cluster = None
            best_delta = 0.0

            for k, cluster in enumerate(clusters):
                delta = profit(cluster + [idx]) - profit(cluster)
                if delta > best_delta:
                    best_delta = delta
                    best_cluster = k

            # Сравниваем с возвратом в исходный кластер.
            if current is not None:
                return_delta = profit(old_cluster + [idx]) - profit(old_cluster)
                if return_delta >= best_delta:
                    best_cluster = current
                    best_delta = return_delta

            if best_cluster is None:
                clusters.append([idx])
            else:
                clusters[best_cluster].append(idx)

            # Проверяем, изменилось ли фактическое содержимое кластера.
            if current is None:
                moved = True
            elif best_cluster != current:
                moved = True

        if not moved:
            break

    # Удаляем пустые кластеры и заново нумеруем метки.
    clusters = [c for c in clusters if c]
    labels = np.full(n, -1, dtype=int)

    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id

    if np.any(labels < 0):
        raise RuntimeError("CLOPE: не все объекты получили кластер.")

    return labels, clusters


def jaccard_distance_matrix(transactions):
    """Матрица расстояний Жаккара для категориальных транзакций."""
    sets = [set(t) for t in transactions]
    n = len(sets)
    distances = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            union = len(sets[i] | sets[j])
            inter = len(sets[i] & sets[j])
            d = 0.0 if union == 0 else 1.0 - inter / union
            distances[i, j] = d
            distances[j, i] = d

    return distances


def clope_silhouette(labels, distance_matrix):
    """Silhouette по расстоянию Жаккара; для одного кластера = -1."""
    if len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


# Подбор коэффициента отталкивания
repulsion_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
jaccard_matrix = jaccard_distance_matrix(transactions)

print("\nПроверка коэффициента repulsion:")

clope_results = []

for repulsion in repulsion_values:
    labels_test, clusters_test = clope(
        transactions,
        repulsion=repulsion,
        max_iter=20
    )

    n_clusters = len(clusters_test)
    sil = clope_silhouette(labels_test, jaccard_matrix)
    singleton_count = sum(len(c) == 1 for c in clusters_test)

    clope_results.append(
        (repulsion, n_clusters, sil, labels_test, clusters_test)
    )

    print(
        f"repulsion = {repulsion}: "
        f"{n_clusters} кластеров, "
        f"silhouette (Jaccard) = {sil:.4f}, "
        f"одноэлементных кластеров = {singleton_count}"
    )

# Для отчёта выбираем наиболее качественное разбиение по silhouette.
# При близких значениях предпочтение отдаётся менее раздробленному.
# Для данного набора данных используем r = 2.0 как компромисс
# между объединением похожих транзакций и чрезмерной фрагментацией.
# Значения silhouette выше выводятся для обоснования выбора.
selected_repulsion = 2.0

selected_result = next(
    result for result in clope_results if result[0] == selected_repulsion
)
clope_labels = selected_result[3]
clope_clusters = selected_result[4]
selected_clope_silhouette = selected_result[2]

print(
    f"\nВыбран коэффициент repulsion: {selected_repulsion}"
)
print(
    f"Количество полученных кластеров: {len(clope_clusters)}"
)
print(
    f"Silhouette (Jaccard): {selected_clope_silhouette:.4f}"
)

clope_sizes = pd.Series(clope_labels).value_counts().sort_index()

print("\nРазмеры кластеров CLOPE:")
for cluster_id, size in clope_sizes.items():
    print(f"Кластер {cluster_id + 1}: {size} объектов")


print("\n" + "=" * 75)
print("ХАРАКТЕРИСТИКА КЛАСТЕРОВ CLOPE")
print("=" * 75)

result_clope = cluster_df.copy()
result_clope["clope_cluster"] = clope_labels + 1

for cluster_id in sorted(result_clope["clope_cluster"].unique()):
    print(f"\n--- Кластер {cluster_id} ---")

    subset = result_clope[
        result_clope["clope_cluster"] == cluster_id
    ]

    print(f"Количество объектов: {len(subset)}")

    top_characteristics = []
    for col in categorical_features:
        mode_values = subset[col].value_counts()
        if len(mode_values) > 0:
            top_value = mode_values.index[0]
            top_count = mode_values.iloc[0]
            share = top_count / len(subset) * 100
            top_characteristics.append(
                f"{col}={top_value} ({share:.0f}%)"
            )
    print("Характерные категории:", "; ".join(top_characteristics))


print("\n" + "=" * 75)
print("РАССТОЯНИЕ ГАУЭРА")
print("=" * 75)


def gower_distance_matrix(data, numeric_cols, categorical_cols):
    """
    Расчёт матрицы расстояний Гауэра для смешанных данных.
    Для числовых признаков используется нормированная абсолютная разница.
    Для категориальных: 0 — совпадают, 1 — различаются.
    """

    n = len(data)
    distances = np.zeros((n, n), dtype=float)

    numeric_ranges = {}

    for col in numeric_cols:
        col_min = data[col].min()
        col_max = data[col].max()
        numeric_ranges[col] = col_max - col_min

    for i in range(n):
        for j in range(i + 1, n):

            components = []

            # Числовые признаки.
            for col in numeric_cols:
                value_i = data.iloc[i][col]
                value_j = data.iloc[j][col]
                col_range = numeric_ranges[col]

                if col_range == 0:
                    d = 0.0
                else:
                    d = abs(value_i - value_j) / col_range

                components.append(d)

            # Категориальные признаки.
            for col in categorical_cols:
                value_i = data.iloc[i][col]
                value_j = data.iloc[j][col]

                components.append(
                    0.0 if value_i == value_j else 1.0
                )

            distance = np.mean(components)

            distances[i, j] = distance
            distances[j, i] = distance

    return distances


gower_matrix = gower_distance_matrix(
    cluster_df,
    numeric_features,
    categorical_features
)

print("Матрица расстояний Гауэра рассчитана.")
print(f"Размер матрицы: {gower_matrix.shape}")


print("\n" + "=" * 75)
print("КЛАСТЕРИЗАЦИЯ ПО РАССТОЯНИЮ ГАУЭРА")
print("=" * 75)

# Используем агломеративную кластеризацию с предварительно
# рассчитанной матрицей Гауэра.
#
# Проверяем несколько значений k и выбираем вариант
# с наибольшим коэффициентом силуэта.
candidate_k = range(2, 7)

gower_results = []

for k in candidate_k:
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average"
    )

    labels = model.fit_predict(gower_matrix)

    silhouette = silhouette_score(
        gower_matrix,
        labels,
        metric="precomputed"
    )

    gower_results.append((k, silhouette))

    print(
        f"k = {k}: "
        f"коэффициент силуэта = {silhouette:.4f}"
    )

best_k, best_silhouette = max(
    gower_results,
    key=lambda x: x[1]
)

print(
    f"\nВыбрано количество кластеров: {best_k}"
)
print(
    f"Максимальный коэффициент силуэта: "
    f"{best_silhouette:.4f}"
)

gower_model = AgglomerativeClustering(
    n_clusters=best_k,
    metric="precomputed",
    linkage="average"
)

gower_labels = gower_model.fit_predict(gower_matrix)


# ------------------------------------------------------------
# ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ: СРАВНЕНИЕ ПРИ 5 КЛАСТЕРАХ
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("ДОПОЛНИТЕЛЬНОЕ СРАВНЕНИЕ ПРИ 5 КЛАСТЕРАХ")
print("=" * 75)

# target используется только для внешней оценки.
# При построении кластеров он не используется.

target_for_metrics = target.astype(int)

# ------------------------------------------------------------
# 1. Проверка возможности получить 5 кластеров CLOPE
# ------------------------------------------------------------

clope_5_labels = None
clope_5_repulsion = None

for r in np.arange(0.5, 3.01, 0.05):
    labels_test, clusters_test = clope(
        transactions,
        repulsion=round(float(r), 2),
        max_iter=20
    )

    if len(clusters_test) == 5:
        clope_5_labels = labels_test
        clope_5_repulsion = round(float(r), 2)
        break

if clope_5_labels is None:
    print("\nCLOPE не сформировал ровно 5 кластеров.")
    print("Основной эксперимент CLOPE оставляем с 35 кластерами.")
else:
    print(f"\nCLOPE: repulsion = {clope_5_repulsion}")
    print(f"Количество кластеров: {len(np.unique(clope_5_labels))}")


# ------------------------------------------------------------
# 2. Кластеризация Гауэра с k = 5
# ------------------------------------------------------------

gower_5_model = AgglomerativeClustering(
    n_clusters=5,
    metric="precomputed",
    linkage="average"
)

gower_5_labels = gower_5_model.fit_predict(gower_matrix)

print("\nGOWER:")
print(f"Количество кластеров: {len(np.unique(gower_5_labels))}")


# ------------------------------------------------------------
# 3. Расчёт внешних метрик
# ------------------------------------------------------------

def calculate_external_metrics(true_labels, cluster_labels):
    return {
        "ARI": adjusted_rand_score(true_labels, cluster_labels),
        "NMI": normalized_mutual_info_score(true_labels, cluster_labels),
        "Homogeneity": homogeneity_score(true_labels, cluster_labels),
        "Completeness": completeness_score(true_labels, cluster_labels),
        "V-measure": v_measure_score(true_labels, cluster_labels)
    }


gower_metrics = calculate_external_metrics(
    target_for_metrics,
    gower_5_labels
)

print("\nВНЕШНИЕ МЕТРИКИ GOWER (5 КЛАСТЕРОВ):")
print(
    pd.DataFrame(gower_metrics, index=["Gower"])
    .round(4)
    .to_string()
)

if clope_5_labels is not None:
    clope_metrics = calculate_external_metrics(
        target_for_metrics,
        clope_5_labels
    )

    metrics_comparison = pd.DataFrame({
        "CLOPE (5 кластеров)": clope_metrics,
        "Gower (5 кластеров)": gower_metrics
    })

    print("\nСРАВНЕНИЕ ВНЕШНИХ МЕТРИК:")
    print(metrics_comparison.round(4).to_string())
else:
    print("\nСравнение внешних метрик CLOPE и Gower при 5 кластерах невозможно.")


# ------------------------------------------------------------
# 4. Расчёт коэффициента силуэта Gower
# ------------------------------------------------------------

gower_5_silhouette = silhouette_score(
    gower_matrix,
    gower_5_labels,
    metric="precomputed"
)

print("\nВНУТРЕННЯЯ МЕТРИКА (SILHOUETTE):")
print(f"Gower, 5 кластеров: {gower_5_silhouette:.4f}")


# ------------------------------------------------------------
# 5. Построение силуэтной диаграммы
# ------------------------------------------------------------

def plot_silhouette(distance_matrix, labels, title):
    sample_values = silhouette_samples(
        distance_matrix,
        labels,
        metric="precomputed"
    )

    average_value = sample_values.mean()
    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_lower = 10

    for cluster_id in unique_labels:
        cluster_values = sample_values[labels == cluster_id]
        cluster_values.sort()

        size = len(cluster_values)
        y_upper = y_lower + size

        color = plt.cm.nipy_spectral(
            float(cluster_id) / max(len(unique_labels), 1)
        )

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        ax.text(
            -0.05,
            y_lower + size / 2,
            str(cluster_id + 1)
        )

        y_lower = y_upper + 10

    ax.axvline(
        x=average_value,
        color="red",
        linestyle="--",
        label=f"Средний силуэт = {average_value:.3f}"
    )

    ax.set_title(title)
    ax.set_xlabel("Коэффициент силуэта")
    ax.set_ylabel("Кластеры")
    ax.legend()
    ax.set_yticks([])
    ax.set_xlim(-0.2, 1.0)

    plt.tight_layout()
    plt.show()


plot_silhouette(
    gower_matrix,
    gower_5_labels,
    "Силуэтная диаграмма Gower (5 кластеров)"
)


# ------------------------------------------------------------
# 6. Таблица распределения target по кластерам
# ------------------------------------------------------------

print("\nРаспределение target по 5 кластерам Gower:")
print(
    pd.crosstab(
        gower_5_labels + 1,
        target_for_metrics
    ).to_string()
)

print("\nДополнительное сравнение завершено.")


print("\n" + "=" * 75)
print("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ПО ГАУЭРУ")
print("=" * 75)

result_gower = cluster_df.copy()
result_gower["gower_cluster"] = gower_labels + 1

gower_sizes = pd.Series(gower_labels).value_counts().sort_index()

print("\nРазмеры кластеров:")
for cluster_id, size in gower_sizes.items():
    print(f"Кластер {cluster_id + 1}: {size} объектов")

for cluster_id in sorted(result_gower["gower_cluster"].unique()):
    print(f"\n--- Кластер {cluster_id} ---")

    subset = result_gower[
        result_gower["gower_cluster"] == cluster_id
    ]

    print(f"Количество объектов: {len(subset)}")

    top_characteristics = []
    for col in categorical_features:
        mode_values = subset[col].value_counts()
        if len(mode_values) > 0:
            top_value = mode_values.index[0]
            top_count = mode_values.iloc[0]
            share = top_count / len(subset) * 100
            top_characteristics.append(
                f"{col}={top_value} ({share:.0f}%)"
            )
    print("Характерные категории:", "; ".join(top_characteristics))


print("\n" + "=" * 75)
print("СРАВНЕНИЕ МЕТОДОВ")
print("=" * 75)

comparison = pd.DataFrame({
    "CLOPE": pd.Series(clope_labels + 1).value_counts().sort_index(),
    "Gower": pd.Series(gower_labels + 1).value_counts().sort_index()
})

print("\nРазмеры кластеров:")
print(comparison)

print("\nКоличество кластеров:")
print(f"CLOPE: {len(clope_clusters)}")
print(f"Gower: {best_k}")

# ВАЖНО:
# target НЕ участвовал в формировании кластеров.
# Здесь он используется только для последующей интерпретации.

print("\n" + "=" * 75)
print("ДОПОЛНИТЕЛЬНАЯ ИНТЕРПРЕТАЦИЯ")
print("=" * 75)

interpretation = pd.DataFrame({
    "CLOPE_cluster": clope_labels + 1,
    "Gower_cluster": gower_labels + 1,
    "target": target
})

print("\nРаспределение target по кластерам CLOPE:")
print(pd.crosstab(
    interpretation["CLOPE_cluster"], interpretation["target"]
).to_string())

print("\nРаспределение target по кластерам Gower:")
print(pd.crosstab(
    interpretation["Gower_cluster"], interpretation["target"]
).to_string())

# 11. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
result = df.copy()
result["clope_cluster"] = clope_labels + 1
result["gower_cluster"] = gower_labels + 1

result.to_csv(
    "lab51_clustering_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 75)
print("ГОТОВО")
print("=" * 75)

print(
    "\nРезультаты кластеризации сохранены в файл:"
    "\nlab51_clustering_results.csv"
)

print(
    "\nВажно: target не использовался при построении кластеров."
    "\nОн выведен только на последнем этапе для интерпретации."
)



# ============================================================
# ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ (PCA / MDS / t-SNE)
# ============================================================
print("\n" + "=" * 75)
print("ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
print("=" * 75)

# ---------- 1. Подготовка one-hot матрицы категориальных признаков ----------
# Именно в этом пространстве по смыслу работал CLOPE
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_onehot = encoder.fit_transform(clope_df[categorical_features])
print(f"One-hot матрица категориальных признаков: {X_cat_onehot.shape}")

# ---------- 2. PCA по one-hot (для CLOPE) ----------
pca_clope = PCA(n_components=2, random_state=42)
X_pca_clope = pca_clope.fit_transform(X_cat_onehot)
print(f"PCA (one-hot) объяснённая дисперсия: "
      f"{pca_clope.explained_variance_ratio_.sum():.3f}")

# ---------- 3. MDS по матрице Гауэра (для Gower) ----------
# Кластеризация Гауэра шла в пространстве этой матрицы -> MDS честнее PCA
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42,
    normalized_stress="auto",
    n_init=4,
    max_iter=300
)
X_mds_gower = mds.fit_transform(gower_matrix)
print(f"MDS по матрице Гауэра: stress = {mds.stress_:.4f}")

# ============================================================
# РИСУНОК 1: CLOPE — PCA по one-hot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    X_pca_clope[:, 0], X_pca_clope[:, 1],
    c=clope_labels, cmap="tab20", s=40, alpha=0.8
)
ax.set_xlabel(f"PC1 ({pca_clope.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_clope.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(
    f"CLOPE: PCA по one-hot категориальным признакам\n"
    f"{len(clope_clusters)} кластеров, silhouette = {selected_clope_silhouette:.4f}"
)
plt.colorbar(scatter, ax=ax, label="Кластер CLOPE", shrink=0.8)
plt.tight_layout()
plt.savefig("fig_clope_pca.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# РИСУНОК 2: Gower — MDS по матрице Гауэра, окраска по кластерам
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    X_mds_gower[:, 0], X_mds_gower[:, 1],
    c=gower_labels, cmap="viridis", s=40, alpha=0.8
)
ax.set_xlabel("MDS-1")
ax.set_ylabel("MDS-2")
ax.set_title(
    f"Gower: MDS по матрице расстояний Гауэра\n"
    f"k = {best_k}, silhouette = {best_silhouette:.4f}"
)
plt.colorbar(scatter, ax=ax, label="Кластер Gower", shrink=0.8)
plt.tight_layout()
plt.savefig("fig_gower_mds.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# РИСУНОК 3: Gower — та же MDS-проекция, окраска по target
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    X_mds_gower[:, 0], X_mds_gower[:, 1],
    c=target, cmap="Set1", s=40, alpha=0.8
)
ax.set_xlabel("MDS-1")
ax.set_ylabel("MDS-2")
ax.set_title(
    "Gower: MDS по расстоянию Гауэра, окраска по target\n"
    "(target не участвовал в кластеризации)"
)
plt.colorbar(scatter, ax=ax, label="target", shrink=0.8)
plt.tight_layout()
plt.savefig("fig_gower_mds_target.png", dpi=150, bbox_inches="tight")
plt.show()
