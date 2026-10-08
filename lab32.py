import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, f
from scipy.stats import kruskal


# 6. Для независимой переменной (фактора) определить градации,
# разбить значения зависимой переменной в соответствии с
# градациями фактора. Если фактор – не категориальная переменная,
# а числовая, аргументировать разбиение ее значений на интервалы.
# 7. Провести однофакторный дисперсионный анализ (статистика
# Фишера).
# 8. Подтвердить или опровергнуть выдвинутую гипотезу.
# Интерпретировать результаты.
# 9. Применить критерий Краскела-Уоллиса. Подтвердить или
# опровергнуть выдвинутую гипотезу. Интерпретировать результаты.


BASE = Path(__file__).parent
FILE_NAME = BASE / "SeoulBikeData.csv"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

if not FILE_NAME.exists():
    print("ОШИБКА: SeoulBikeData.csv должен находиться рядом с lab31.py")
    raise SystemExit

df = None
for enc in ["utf-8", "cp1252", "ISO-8859-1"]:
    try:
        df = pd.read_csv(FILE_NAME, encoding=enc)
        break
    except UnicodeDecodeError:
        pass

if df is None:
    print("Не удалось прочитать CSV.")
    raise SystemExit

df = df.rename(columns={
    "Rented Bike Count": "Rented_Bike_Count",
    "Temperature(°C)": "Temperature",
    "Humidity(%)": "Humidity"
})


print("\n" + "=" * 90)
print("6. ГРАДАЦИИ ФАКТОРА И РАЗБИЕНИЕ ЗАВИСИМОЙ ПЕРЕМЕННОЙ")
print("=" * 90)

# Независимая переменная (фактор) — Seasons.
# Фактор является категориальной переменной, поэтому
# дополнительное разбиение на интервалы не требуется.

factor_levels = df["Seasons"].dropna().unique()

print("\nНезависимая переменная (фактор): Seasons")
print("Количество градаций:", len(factor_levels))
print("Градации фактора:")

for level in sorted(factor_levels):
    print(f"- {level}")

# Разбиение зависимой переменной Rented Bike Count
# в соответствии с градациями фактора.
season_groups = {
    level: df.loc[
        df["Seasons"] == level,
        "Rented_Bike_Count"
    ].dropna()
    for level in sorted(factor_levels)
}

print("\nРаспределение зависимой переменной по градациям фактора:")

for level, values in season_groups.items():
    print(f"\n{level}:")
    print(f"Количество наблюдений: {len(values)}")
    print(f"Среднее значение: {values.mean():.4f}")
    print(f"Минимальное значение: {values.min():.4f}")
    print(f"Максимальное значение: {values.max():.4f}")

print("\nДля дальнейшего дисперсионного анализа сформированы 4 группы:")
for level, values in season_groups.items():
    print(f"{level}: n = {len(values)}")


print("\n" + "=" * 90)
print("7. ОДНОФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ (ANOVA)")
print("=" * 90)


# Формирование групп по градациям фактора Seasons
factor_levels = sorted(df["Seasons"].dropna().unique())

season_groups = {
    level: df.loc[
        df["Seasons"] == level,
        "Rented_Bike_Count"
    ].dropna()
    for level in factor_levels
}


print("\nГруппы для дисперсионного анализа:")

for level, values in season_groups.items():
    print(
        f"{level}: "
        f"n = {len(values)}, "
        f"среднее = {values.mean():.4f}"
    )


# Расчёт однофакторного ANOVA
f_statistic, p_value = f_oneway(
    *season_groups.values()
)


# Количество групп и общее количество наблюдений
k = len(season_groups)
N = sum(len(values) for values in season_groups.values())


# Степени свободы
df_between = k - 1
df_within = N - k


# Критическое значение F-критерия при уровне значимости 0.05
alpha = 0.05

f_critical = f.ppf(
    1 - alpha,
    df_between,
    df_within
)


print("\n" + "-" * 90)
print("РЕЗУЛЬТАТЫ ОДНОФАКТОРНОГО ANOVA")
print("-" * 90)

print(f"Количество групп: {k}")
print(f"Общее количество наблюдений: {N}")

print(f"\nСтепени свободы между группами: {df_between}")
print(f"Степени свободы внутри групп: {df_within}")

print(f"\nF-критерий: {f_statistic:.6f}")
print(f"p-value: {p_value:.10f}")

print(f"\nУровень значимости α: {alpha}")
print(f"Критическое значение F: {f_critical:.6f}")


print("\n" + "=" * 90)
print("8. ПРОВЕРКА ГИПОТЕЗЫ И ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ ANOVA")
print("=" * 90)

# Формирование групп по градациям фактора Seasons
factor_levels = sorted(df["Seasons"].dropna().unique())

season_groups = {
    level: df.loc[
        df["Seasons"] == level,
        "Rented_Bike_Count"
    ].dropna()
    for level in factor_levels
}


# Однофакторный ANOVA
f_statistic, p_value = f_oneway(
    *season_groups.values()
)

alpha = 0.05

print("\nПроверяемые гипотезы:")
print("H0: среднее количество арендованных велосипедов")
print("    одинаково для всех сезонов.")
print("\nH1: среднее количество арендованных велосипедов")
print("    различается хотя бы для двух сезонов.")


print("\n" + "-" * 90)
print("РЕЗУЛЬТАТ ПРОВЕРКИ")
print("-" * 90)

print(f"F-критерий: {f_statistic:.6f}")
print(f"p-value: {p_value:.10f}")
print(f"Уровень значимости α: {alpha:.2f}")


if p_value < alpha:
    print("\np-value < α")
    print("Нулевая гипотеза H0 отвергается.")
    print("Принимается альтернативная гипотеза H1.")
else:
    print("\np-value >= α")
    print("Нет оснований отвергать нулевую гипотезу H0.")


print("\n" + "-" * 90)
print("СРЕДНИЕ ЗНАЧЕНИЯ ПО СЕЗОНАМ")
print("-" * 90)

for level, values in season_groups.items():
    print(f"{level}: {values.mean():.4f}")


print("\n" + "=" * 90)
print("9. КРИТЕРИЙ КРАСКЕЛА–УОЛЛИСА")
print("=" * 90)

# Формирование групп по градациям фактора Seasons
factor_levels = sorted(df["Seasons"].dropna().unique())

season_groups = {
    level: df.loc[
        df["Seasons"] == level,
        "Rented_Bike_Count"
    ].dropna()
    for level in factor_levels
}


print("\nГруппы для критерия Краскела–Уоллиса:")

for level, values in season_groups.items():
    print(
        f"{level}: "
        f"n = {len(values)}, "
        f"медиана = {values.median():.4f}"
    )


# Критерий Краскела–Уоллиса
h_statistic, p_value = kruskal(
    *season_groups.values()
)

alpha = 0.05

print("\n" + "-" * 90)
print("РЕЗУЛЬТАТ КРИТЕРИЯ КРАСКЕЛА–УОЛЛИСА")
print("-" * 90)

print(f"H-критерий: {h_statistic:.6f}")
print(f"p-value: {p_value:.10f}")
print(f"Уровень значимости α: {alpha:.2f}")
