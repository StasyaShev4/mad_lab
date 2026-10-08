import pandas as pd

# 1. Загрузка датасета
df = pd.read_csv("forestfires.csv")

print("=" * 60)
print("1. ПЕРВЫЕ 10 СТРОК ДАТАСЕТА")
print("=" * 60)
print(df.head(10))

# 2. Общая информация о датасете
print("\n" + "=" * 60)
print("2. ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 60)
print(df.info())

# 3. Размер датасета
print("\n" + "=" * 60)
print("3. РАЗМЕР ДАТАСЕТА")
print("=" * 60)
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

# 4. Проверка пропущенных значений
print("\n" + "=" * 60)
print("4. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
print("=" * 60)
print(df.isnull().sum())

# 5. Описательная статистика числовых признаков
print("\n" + "=" * 60)
print("5. ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 60)
print(df.describe())

# 6. Категориальный признак month
print("\n" + "=" * 60)
print("6. РАСПРЕДЕЛЕНИЕ ПО МЕСЯЦАМ")
print("=" * 60)
print(df["month"].value_counts())

# 7. Категориальный признак day
print("\n" + "=" * 60)
print("7. РАСПРЕДЕЛЕНИЕ ПО ДНЯМ НЕДЕЛИ")
print("=" * 60)
print(df["day"].value_counts())

# 8. Выбранные признаки для дальнейшего анализа
selected_columns = ["temp", "RH", "wind", "area", "month"]

print("\n" + "=" * 60)
print("8. ВЫБРАННЫЕ ПРИЗНАКИ ДЛЯ ЛАБОРАТОРНОЙ")
print("=" * 60)
print(df[selected_columns].head(10))

print("\nГотово! Датасет успешно загружен.")
