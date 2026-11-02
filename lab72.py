from pathlib import Path
import warnings
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
FILE_NAME = BASE / "SeoulBikeData.csv"
PLOTS = BASE / "plots_lab72"
PLOTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
RANDOM_STATE = 42

def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)

print("=" * 88)
print("ЛАБОРАТОРНАЯ РАБОТА №7. ВИЗУАЛИЗАЦИЯ ДАННЫХ — ЧАСТЬ 2")
print("=" * 88)

if not FILE_NAME.exists():
    raise SystemExit(f"ОШИБКА: положи SeoulBikeData.csv рядом с lab72.py\nПуть: {FILE_NAME}")

df = None
for enc in ("utf-8", "cp1252", "ISO-8859-1"):
    try:
        df = pd.read_csv(FILE_NAME, encoding=enc)
        break
    except UnicodeDecodeError:
        pass
if df is None:
    raise SystemExit("Не удалось прочитать CSV-файл.")

df = df.rename(columns={
    "Rented Bike Count": "Rented_Bike_Count",
    "Temperature(°C)": "Temperature",
    "Humidity(%)": "Humidity",
    "Wind speed (m/s)": "Wind_Speed",
    "Visibility (10m)": "Visibility",
    "Dew point temperature(°C)": "Dew_Point_Temperature",
    "Solar Radiation (MJ/m2)": "Solar_Radiation",
    "Rainfall(mm)": "Rainfall",
    "Snowfall (cm)": "Snowfall",
    "Functioning Day": "Functioning_Day",
})
needed = ["Rented_Bike_Count", "Hour", "Temperature", "Humidity", "Wind_Speed",
          "Visibility", "Dew_Point_Temperature", "Solar_Radiation",
          "Rainfall", "Snowfall", "Seasons", "Holiday", "Functioning_Day"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"В датасете отсутствуют столбцы: {missing}")
for c in needed[:10]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=needed).copy()
df = df[df["Rented_Bike_Count"] >= 0].copy()
print(f"Датасет подготовлен: {len(df)} строк.")

# ------------------------------------------------------------
# 1. КЛАССИФИКАЦИЯ
# Цель: определить категорию спроса (низкий/средний/высокий)
# по погодным условиям, часу, сезону и календарным признакам.
# ------------------------------------------------------------
print("\n1. КЛАССИФИКАЦИЯ УРОВНЯ СПРОСА")
q1, q2 = df["Rented_Bike_Count"].quantile([1/3, 2/3])
df["Demand_Level"] = pd.cut(
    df["Rented_Bike_Count"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Низкий", "Средний", "Высокий"],
    include_lowest=True
)
print(f"Границы классов по терцилям: низкий ≤ {q1:.1f}; средний ≤ {q2:.1f}; высокий > {q2:.1f}")
print("Распределение классов:")
print(df["Demand_Level"].value_counts().reindex(["Низкий", "Средний", "Высокий"]))

features_num = ["Hour", "Temperature", "Humidity", "Wind_Speed", "Visibility",
                "Dew_Point_Temperature", "Solar_Radiation", "Rainfall", "Snowfall"]
X = df[features_num + ["Seasons", "Holiday", "Functioning_Day"]]
X = pd.get_dummies(X, columns=["Seasons", "Holiday", "Functioning_Day"], drop_first=False)
y = df["Demand_Level"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
clf = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced",
    min_samples_leaf=2, n_jobs=-1
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(f"Accuracy = {accuracy_score(y_test, pred):.4f}")
print("\nОтчёт классификации:")
print(classification_report(y_test, pred, digits=3, zero_division=0))
labels = ["Низкий", "Средний", "Высокий"]
cm = confusion_matrix(y_test, pred, labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d"
)
ax.set_title("Матрица ошибок классификации уровня спроса")
save_plot(fig, "01_classification_confusion_matrix.png")

importance = pd.DataFrame({"Признак": X.columns, "Важность": clf.feature_importances_})
importance = importance.sort_values("Важность", ascending=False).head(12)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=importance, x="Важность", y="Признак", color="steelblue", ax=ax)
ax.set_title("12 наиболее важных признаков классификации")
save_plot(fig, "02_classification_feature_importance.png")

# ------------------------------------------------------------
# 2. КЛАСТЕРИЗАЦИЯ K-MEANS
# Кластеры строятся по погоде и времени, а спрос используется
# после обучения для описания полученных кластеров.
# ------------------------------------------------------------
print("\n2. КЛАСТЕРИЗАЦИЯ K-MEANS")
cluster_features = ["Hour", "Temperature", "Humidity", "Wind_Speed",
                    "Visibility", "Solar_Radiation", "Rainfall", "Snowfall"]
cluster_data = df[cluster_features].copy()
scaler = StandardScaler()
Z = scaler.fit_transform(cluster_data)

# Выбираем число кластеров из диапазона 2..6 по силуэту.
silhouette_values = {}
sample_n = min(2500, len(Z))
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(Z), size=sample_n, replace=False)
for k in range(2, 7):
    model_tmp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_tmp = model_tmp.fit_predict(Z)
    silhouette_values[k] = silhouette_score(Z[sample_idx], labels_tmp[sample_idx])
best_k = max(silhouette_values, key=silhouette_values.get)
print("Коэффициенты силуэта:", {k: round(v, 4) for k, v in silhouette_values.items()})
print(f"Выбрано число кластеров: {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans.fit_predict(Z)
print("Количество наблюдений по кластерам:")
print(df["Cluster"].value_counts().sort_index())
cluster_profile = df.groupby("Cluster")[cluster_features + ["Rented_Bike_Count"]].mean()
print("\nСредние характеристики кластеров:")
print(cluster_profile.round(2).to_string())

pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(Z)
plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Кластер": df["Cluster"].astype(str).to_numpy()})
if len(plot_df) > 4000:
    plot_df = plot_df.sample(4000, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Кластер", palette="tab10",
                alpha=.65, s=22, ax=ax)
ax.set_title("Кластеры погодных условий и времени (проекция PCA)")
save_plot(fig, "03_clustering_pca_scatter.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=cluster_profile.index.astype(str), y=cluster_profile["Rented_Bike_Count"].values, color="steelblue", ax=ax)
ax.set(title="Среднее количество аренд в каждом кластере", xlabel="Кластер", ylabel="Среднее количество аренд")
save_plot(fig, "04_clustering_demand_by_cluster.png")

# Тепловая карта стандартизованных средних профилей кластеров.
profile_scaled = (cluster_profile - cluster_profile.mean()) / cluster_profile.std(ddof=0).replace(0, 1)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(profile_scaled, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Профили кластеров (стандартизованные средние)")
save_plot(fig, "05_clustering_profile_heatmap.png")

# ------------------------------------------------------------
# 3. АССОЦИАТИВНЫЕ ПРАВИЛА — собственная реализация Apriori.
# Числовые признаки дискретизируются в понятные категории.
# ------------------------------------------------------------
print("\n3. ПОИСК АССОЦИАТИВНЫХ ПРАВИЛ (APRIORI)")
transactions = []
for _, row in df.iterrows():
    temp = "Холодно" if row["Temperature"] < 10 else ("Умеренно" if row["Temperature"] < 20 else "Тепло")
    humidity = "Низкая влажность" if row["Humidity"] < 40 else ("Средняя влажность" if row["Humidity"] < 70 else "Высокая влажность")
    rain = "Есть дождь" if row["Rainfall"] > 0 else "Нет дождя"
    snow = "Есть снег" if row["Snowfall"] > 0 else "Нет снега"
    wind = "Сильный ветер" if row["Wind_Speed"] >= 4 else "Слабый ветер"
    demand = "Спрос=Низкий" if row["Rented_Bike_Count"] <= q1 else (
        "Спрос=Средний" if row["Rented_Bike_Count"] <= q2 else "Спрос=Высокий"
    )
    hour_group = "Ночь" if row["Hour"] < 6 else ("Утро" if row["Hour"] < 12 else ("День" if row["Hour"] < 18 else "Вечер"))
    transactions.append({
        f"Температура={temp}", f"Влажность={humidity}", f"Осадки={rain}",
        f"Снег={snow}", f"Ветер={wind}", f"Сезон={row['Seasons']}",
        f"Период={hour_group}", demand
    })

n_transactions = len(transactions)
min_support = 0.05
min_confidence = 0.35
min_count = int(np.ceil(min_support * n_transactions))
print(f"Транзакций: {n_transactions}; минимальная поддержка: {min_support:.2f}; минимальная достоверность: {min_confidence:.2f}")

# Частые наборы размера 1.
item_counts = Counter(item for trans in transactions for item in trans)
frequent = {frozenset([item]): count for item, count in item_counts.items() if count >= min_count}
all_frequent = dict(frequent)
k = 2
while frequent and k <= 3:
    prev_sets = list(frequent.keys())
    candidates = set()
    for a, b in combinations(prev_sets, 2):
        union = a | b
        if len(union) == k:
            candidates.add(union)
    # Apriori pruning: все подмножества размера k-1 должны быть частыми.
    candidates = {
        c for c in candidates
        if all(frozenset(sub) in frequent for sub in combinations(c, k-1))
    }
    counts = Counter()
    for trans in transactions:
        for cand in candidates:
            if cand.issubset(trans):
                counts[cand] += 1
    frequent = {itemset: count for itemset, count in counts.items() if count >= min_count}
    all_frequent.update(frequent)
    k += 1

rules = []
for itemset, item_count in all_frequent.items():
    if len(itemset) < 2:
        continue
    # Для этой работы нас интересуют правила, где следствие — категория спроса.
    demand_items = [x for x in itemset if x.startswith("Спрос=")]
    if not demand_items:
        continue
    consequent = frozenset([demand_items[0]])
    antecedent = itemset - consequent
    if not antecedent:
        continue
    antecedent_count = sum(antecedent.issubset(t) for t in transactions)
    if antecedent_count == 0:
        continue
    support = item_count / n_transactions
    confidence = item_count / antecedent_count
    demand_count = item_counts[demand_items[0]]
    consequent_support = demand_count / n_transactions
    lift = confidence / consequent_support if consequent_support else np.nan
    if confidence >= min_confidence:
        rules.append({
            "Условие": " + ".join(sorted(antecedent)),
            "Следствие": demand_items[0].replace("Спрос=", ""),
            "Support": support, "Confidence": confidence, "Lift": lift,
            "Размер набора": len(itemset)
        })

rules_df = pd.DataFrame(rules)
if not rules_df.empty:
    rules_df = rules_df.sort_values(["Lift", "Confidence", "Support"], ascending=False)
    print(f"Найдено правил с заданными порогами: {len(rules_df)}")
    print("\nТоп-15 правил:")
    print(rules_df.head(15).to_string(index=False, formatters={
        "Support": "{:.3f}".format, "Confidence": "{:.3f}".format, "Lift": "{:.3f}".format
    }))
    top = rules_df.head(12).copy()
    top["Правило"] = top["Условие"] + " → " + top["Следствие"]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=top, x="Lift", y="Правило", color="steelblue", ax=ax)
    ax.axvline(1, color="red", linestyle="--", label="Lift = 1")
    ax.set_title("Ассоциативные правила: топ по Lift")
    ax.legend()
    save_plot(fig, "06_association_rules_top_lift.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = 40 + rules_df["Support"] * 500
    points = ax.scatter(rules_df["Support"], rules_df["Confidence"],
                         c=rules_df["Lift"], s=sizes, cmap="viridis", alpha=.7)
    ax.set(title="Ассоциативные правила: Support, Confidence и Lift",
           xlabel="Support", ylabel="Confidence")
    fig.colorbar(points, ax=ax, label="Lift")
    save_plot(fig, "07_association_rules_support_confidence.png")
else:
    print("При заданных порогах правила не найдены. Попробуй снизить min_support или min_confidence.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(.5, .5, "Правила не найдены при текущих порогах", ha="center", va="center")
    ax.axis("off")
    save_plot(fig, "06_association_rules_no_results.png")

print("\nИнтерпретация метрик:")
print("Support — доля наблюдений, в которых встречается всё правило.")
print("Confidence — доля наблюдений с условием, в которых выполняется следствие.")
print("Lift > 1 означает положительную ассоциацию относительно базовой частоты следствия.")
print("\nГОТОВО. CSV-файлы не создаются.")
print(f"Графики сохранены в: {PLOTS}")
for path in sorted(PLOTS.glob("*.png")):
    print(" -", path.name)
