import pandas as pd
from scipy.stats import shapiro

# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №2. КЛАССИФИКАЦИЯ
# Лабораторная 4. Первичный анализ датасета Heart Disease
# ============================================================
# 2. Провести дескриптивный анализ; определить, согласуются ли 
# выборки с нормальным распределением. Оценить распределение 
# объектов по классам. Аргументировать выбор методов решения 
# задачи классификации по результатам описания датасета. 

df = pd.read_csv(
    "dataset-HeartDisease/processed.cleveland.data",
    header=None,
    na_values="?"
)

print("=" * 60)
print("ПЕРВИЧНЫЙ АНАЛИЗ ДАТАСЕТА HEART DISEASE")
print("=" * 60)

print("\nРазмер датасета:")
print(f"Количество объектов: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

print("\nПервые 10 строк:")
print(df.head(10))

print("\nИнформация о данных:")
df.info()


print("\nКоличество пропусков:")
print(df.isna().sum())

print("\nРаспределение объектов по классам:")
print(df.iloc[:, -1].value_counts().sort_index())


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
    "target"     # целевая переменная
]

print("\n" + "=" * 60)
print("НАЗВАНИЯ ПРИЗНАКОВ")
print("=" * 60)

print(df.columns.tolist())

print("\nПервые 10 строк с названиями признаков:")
print(df.head(10))

print("\n" + "=" * 60)
print("ДЕСКРИПТИВНЫЙ АНАЛИЗ")
print("=" * 60)

print("\nОписательная статистика:")
print(df.describe())

print("\nМедианные значения:")
print(df.median(numeric_only=True))

print("\nРаспределение объектов по классам:")

class_counts = df["target"].value_counts().sort_index()

for class_value, count in class_counts.items():
    percentage = count / len(df) * 100
    print(f"Класс {int(class_value)}: {count} объектов ({percentage:.2f}%)")


print("\nПропущенные значения:")

missing = df.isna().sum()

for column, count in missing.items():
    if count > 0:
        print(f"{column}: {count}")


print("\n" + "=" * 60)
print("ПРОВЕРКА НА НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ")
print("=" * 60)

# Проверяем только количественные непрерывные признаки
continuous_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

print("\nКритерий Шапиро–Уилка:")
print("H0: данные имеют нормальное распределение")
print("H1: данные не имеют нормального распределения")
print("Уровень значимости α = 0.05")

for feature in continuous_features:
    data = df[feature].dropna()

    statistic, p_value = shapiro(data)

    print(f"\nПризнак: {feature}")
    print(f"Статистика W = {statistic:.4f}")
    print(f"p-value = {p_value:.6f}")

    if p_value < 0.05:
        print("Вывод: H0 отвергается — распределение отличается от нормального.")
    else:
        print("Вывод: нет оснований отвергать H0 — распределение можно считать близким к нормальному.")


print("\n" + "=" * 60)
print("РАСПРЕДЕЛЕНИЕ ОБЪЕКТОВ ПО КЛАССАМ")
print("=" * 60)

class_counts = df["target"].value_counts().sort_index()

print("\nКоличество объектов и доля каждого класса:")

for class_value, count in class_counts.items():
    percentage = count / len(df) * 100
    print(
        f"Класс {int(class_value)}: "
        f"{count} объектов ({percentage:.2f}%)"
    )

# Определяем наиболее и наименее представленные классы
max_class = class_counts.idxmax()
min_class = class_counts.idxmin()

print("\nНаиболее представленный класс:")
print(
    f"Класс {int(max_class)} — "
    f"{class_counts[max_class]} объектов "
    f"({class_counts[max_class] / len(df) * 100:.2f}%)"
)

print("\nНаименее представленный класс:")
print(
    f"Класс {int(min_class)} — "
    f"{class_counts[min_class]} объектов "
    f"({class_counts[min_class] / len(df) * 100:.2f}%)"
)

print("\nОценка сбалансированности:")
imbalance_ratio = class_counts.max() / class_counts.min()

print(f"Отношение размера крупнейшего класса к наименьшему: "
      f"{imbalance_ratio:.2f}")

if imbalance_ratio > 2:
    print("Классы распределены неравномерно.")
else:
    print("Классы распределены относительно равномерно.")
