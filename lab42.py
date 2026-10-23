import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №2. КЛАССИФИКАЦИЯ
# 3. Провести отбор наиболее информативных признаков. 
# 4. Стандартизировать переменные (при необходимости).
# 5. Построить категоризованные диаграммы рассеивания, оценить 
# расположение классов. 
# 6. Разделить данные на обучающее и тестовое подмножества. 
# ============================================================

df = pd.read_csv(
    "dataset-HeartDisease/processed.cleveland.data",
    header=None,
    na_values="?"
)

df.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

print("=" * 60)
print("ОТБОР ИНФОРМАТИВНЫХ ПРИЗНАКОВ")
print("=" * 60)

print(f"\nКоличество объектов: {df.shape[0]}")
print(f"Количество признаков: {df.shape[1] - 1}")

# ОБРАБОТКА ПРОПУСКОВ
print("\nПропущенные значения до обработки:")

print(df[["ca", "thal"]].isna().sum())

# Для анализа заполняем пропуски медианными значениями
# соответствующих признаков
df_analysis = df.copy()

for column in ["ca", "thal"]:
    df_analysis[column] = df_analysis[column].fillna(
        df_analysis[column].median()
    )

print("\nПропущенные значения после обработки:")

print(df_analysis.isna().sum().sum())

# ПРИЗНАКИ И ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

X = df_analysis[features]
y = df_analysis["target"]

# 1. КОРРЕЛЯЦИЯ ПРИЗНАКОВ С TARGET
print("\n" + "=" * 60)
print("КОРРЕЛЯЦИЯ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("=" * 60)

correlations = X.corrwith(y)

correlation_table = pd.DataFrame({
    "feature": correlations.index,
    "correlation": correlations.values,
    "abs_correlation": correlations.abs().values
})

correlation_table = correlation_table.sort_values(
    by="abs_correlation",
    ascending=False
)

print("\nПризнаки в порядке убывания абсолютной корреляции:")
print(correlation_table.to_string(index=False))


print("\n" + "=" * 60)
print("ВАЖНОСТЬ ПРИЗНАКОВ ПО ДЕРЕВУ РЕШЕНИЙ")
print("=" * 60)

tree = DecisionTreeClassifier(
    random_state=42
)

tree.fit(X, y)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": tree.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nПризнаки в порядке убывания важности:")
print(feature_importance.to_string(index=False))


print("\n" + "=" * 60)
print("СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ")
print("=" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nПервые 5 строк после стандартизации:")
print(X_scaled.head())

print("\nСредние значения после стандартизации:")
print(X_scaled.mean())

print("\nСтандартные отклонения после стандартизации:")
print(X_scaled.std())


print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 60)

print("""
Для дальнейшего построения моделей будут использоваться
исходные признаки для дерева решений и стандартизированные
признаки для метода опорных векторов (SVM).
""")

# ============================================================
# 5. КАТЕГОРИЗОВАННЫЕ ДИАГРАММЫ РАССЕИВАНИЯ
# ============================================================

# Создаём папку для графиков
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

pairs = [
    ("ca", "thal"),
    ("oldpeak", "thalach"),
    ("cp", "thalach"),
    ("age", "thalach"),
    ("oldpeak", "ca")
]

for x_feature, y_feature in pairs:
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df_analysis,
        x=x_feature,
        y=y_feature,
        hue="target",
        palette="tab10",
        s=70
    )

    plt.title(f"Расположение классов: {x_feature} и {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title="Класс")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Имя файла
    filename = f"scatter_{x_feature}_{y_feature}.png"
    filepath = os.path.join(plots_dir, filename)

    # Сохраняем график
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"График сохранён: {filepath}")


# ============================================================
# 6. РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ============================================================

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("6. РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 60)

print(f"Размер исходной выборки: {len(X)}")
print(f"Обучающая выборка: {len(X_train)} объектов")
print(f"Тестовая выборка: {len(X_test)} объектов")

print("\nРаспределение классов в обучающей выборке:")
print(y_train.value_counts().sort_index())

print("\nРаспределение классов в тестовой выборке:")
print(y_test.value_counts().sort_index())
