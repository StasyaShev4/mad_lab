import pandas as pd

# 4. В зависимости от типа переменных выбрать линейную модель
# регрессии или пуассоновскую регрессию.

FILE_NAME = "SeoulBikeData.csv"

try:
    df = pd.read_csv(FILE_NAME, encoding="ISO-8859-1")
except FileNotFoundError:
    print(f"Ошибка: файл {FILE_NAME} не найден.")
    print("Положите SeoulBikeData.csv в одну папку с этим скриптом.")
    raise SystemExit
except UnicodeDecodeError:
    df = pd.read_csv(FILE_NAME, encoding="cp1252")


print("=" * 70)
print("Исходные данные и распределение категориальной переменной")
print("=" * 70)

print("\nРазмер датасета:")
print(f"Строк: {df.shape[0]}")
print(f"Столбцов: {df.shape[1]}")

print("\nНазвания столбцов:")
for column in df.columns:
    print("-", column)

print("\nПервые 5 строк:")
print(df.head())

print("\nТипы данных:")
print(df.dtypes)


# Дескриптивная статистика, проверка нормальности,
# формула Стерджесса и гистограммы уже рассчитываются
# в lab20-11.py.
categorical_variable = "Seasons"

print("\n" + "=" * 70)
print(f"РАСПРЕДЕЛЕНИЕ КАТЕГОРИАЛЬНОЙ ПЕРЕМЕННОЙ: {categorical_variable}")
print("=" * 70)

season_counts = df[categorical_variable].value_counts()

print("\nКоличество наблюдений по категориям:")
print(season_counts)

print("\nДоли категорий (%):")
print(
    (df[categorical_variable].value_counts(normalize=True) * 100)
    .round(2)
)
