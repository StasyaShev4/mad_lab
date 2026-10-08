import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №5. КЛАСТЕРИЗАЦИЯ
# Датасет: Heart Disease (processed.cleveland.data)
# Метод CLOPE + кластеризация по расстоянию Гауэра
# ============================================================

# ------------------------------------------------------------
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ------------------------------------------------------------

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

print("=" * 75)
print("ЛАБОРАТОРНАЯ РАБОТА №5. КЛАСТЕРИЗАЦИЯ")
print("=" * 75)

print("\n1. ПЕРВИЧНЫЙ АНАЛИЗ ДАТАСЕТА")
print("-" * 75)

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

# ------------------------------------------------------------
# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("2. ПОДГОТОВКА ДАННЫХ")
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

# ============================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ CLOPE
# ============================================================

print("=" * 70)
print("3. ПОДГОТОВКА ДАННЫХ ДЛЯ CLOPE")
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

# ------------------------------------------------------------
# 4. РЕАЛИЗАЦИЯ АЛГОРИТМА CLOPE
# ------------------------------------------------------------

def clope_profit(cluster, repulsion):
    """Profit(C) = S(C) / W(C)^r."""
    if not cluster:
        return 0.0
    s = sum(len(t) for t in cluster)
    w = len(set().union(*cluster))
    return s / (w ** repulsion) if w else 0.0


def clope(transactions, repulsion=2.0, max_iter=20):
    """CLOPE с жадным размещением и последующей перестановкой транзакций."""
    clusters = []

    # Начальное размещение.
    for transaction in transactions:
        best_idx = -1
        best_delta = 0.0
        for i, cluster in enumerate(clusters):
            delta = (clope_profit(cluster + [transaction], repulsion)
                     - clope_profit(cluster, repulsion))
            if delta > best_delta + 1e-12:
                best_delta = delta
                best_idx = i
        if best_idx >= 0:
            clusters[best_idx].append(transaction)
        else:
            clusters.append([transaction])

    # Итеративное улучшение.
    for _ in range(max_iter):
        moved = 0
        for transaction in transactions:
            current_idx = next(
                (i for i, c in enumerate(clusters) if transaction in c),
                None
            )
            if current_idx is None:
                continue

            clusters[current_idx].remove(transaction)
            if not clusters[current_idx]:
                clusters.pop(current_idx)

            best_idx = -1
            best_delta = 0.0
            for i, cluster in enumerate(clusters):
                delta = (clope_profit(cluster + [transaction], repulsion)
                         - clope_profit(cluster, repulsion))
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_idx = i

            if best_idx >= 0:
                clusters[best_idx].append(transaction)
            else:
                clusters.append([transaction])
            moved += 1

        if moved == 0:
            break

    # Восстановление меток. Используем очереди индексов для одинаковых
    # транзакций, чтобы не терять объекты с одинаковыми категориями.
    positions = {}
    for i, t in enumerate(transactions):
        positions.setdefault(frozenset(t), []).append(i)

    labels = np.empty(len(transactions), dtype=int)
    for cluster_id, cluster in enumerate(clusters, start=1):
        for t in cluster:
            key = frozenset(t)
            idx = positions[key].pop(0)
            labels[idx] = cluster_id

    return labels, clusters


def jaccard_distance_matrix(transactions):
    n = len(transactions)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a = set(transactions[i])
            b = set(transactions[j])

            union = len(a | b)
            inter = len(a & b)
            value = 1.0 - inter / union if union else 0.0
            d[i, j] = value
            d[j, i] = value
    return d


def transaction_silhouette(labels, distances):
    """Silhouette для заранее рассчитанной матрицы расстояний."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0

    scores = []
    for i in range(len(labels)):
        own = labels[i]
        same = np.where(labels == own)[0]
        same = same[same != i]
        if len(same) == 0:
            scores.append(0.0)
            continue

        a = distances[i, same].mean()
        b = min(
            distances[i, np.where(labels == other)[0]].mean()
            for other in unique if other != own
        )
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)

    return float(np.mean(scores))


jaccard_dist = jaccard_distance_matrix(transactions)

# Подбор коэффициента repulsion по silhouette на категориальных
# транзакциях. Это лучше, чем заранее фиксировать r=2.0.
repulsion_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
clope_results = []

print("\nПроверка коэффициента repulsion:")
for r in repulsion_values:
    labels_r, clusters_r = clope(transactions, repulsion=r, max_iter=20)
    sil_r = transaction_silhouette(labels_r, jaccard_dist)
    n_clusters = len(np.unique(labels_r))
    clope_results.append((r, sil_r, n_clusters, labels_r))
    print(f"repulsion = {r:.1f}: {n_clusters} кластеров, "
          f"silhouette (Jaccard) = {sil_r:.4f}")

# Максимальный silhouette — основной критерий; при равенстве выбирается
# вариант с меньшим числом кластеров.
best_r, best_sil, best_n, clope_labels = max(
    clope_results,
    key=lambda x: (x[1], -x[2])
)

selected_repulsion = best_r

print(f"\nВыбран коэффициент repulsion: {selected_repulsion}")
print(f"Количество полученных кластеров: {best_n}")
print(f"Silhouette (Jaccard): {best_sil:.4f}")

clope_sizes = pd.Series(clope_labels).value_counts().sort_index()
print("\nРазмеры кластеров CLOPE:")
for cluster_id, size in clope_sizes.items():
    print(f"  Кластер {cluster_id}: {size} объектов")

if len(clope_sizes) > 20:
    print(f"... ещё {len(clope_sizes) - 20} кластеров")

# ------------------------------------------------------------
# 5. ХАРАКТЕРИСТИКА КЛАСТЕРОВ CLOPE
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("5. ХАРАКТЕРИСТИКА КЛАСТЕРОВ CLOPE")
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

# ------------------------------------------------------------
# 6. РАССТОЯНИЕ ГАУЭРА
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("6. РАССТОЯНИЕ ГАУЭРА")
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

# ------------------------------------------------------------
# 7. ВЫБОР КОЛИЧЕСТВА КЛАСТЕРОВ ДЛЯ ВТОРОГО МЕТОДА
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("7. КЛАСТЕРИЗАЦИЯ ПО РАССТОЯНИЮ ГАУЭРА")
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
# 8. РАЗМЕРЫ И ХАРАКТЕРИСТИКА КЛАСТЕРОВ ГАУЭРА
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("8. РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ПО ГАУЭРУ")
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

# ------------------------------------------------------------
# 9. СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("9. СРАВНЕНИЕ МЕТОДОВ")
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

# ------------------------------------------------------------
# 10. СВЯЗЬ КЛАСТЕРОВ С TARGET
# ------------------------------------------------------------
# ВАЖНО:
# target НЕ участвовал в формировании кластеров.
# Здесь он используется только для последующей интерпретации.

print("\n" + "=" * 75)
print("10. ДОПОЛНИТЕЛЬНАЯ ИНТЕРПРЕТАЦИЯ")
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

# ------------------------------------------------------------
# 11. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ------------------------------------------------------------

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
