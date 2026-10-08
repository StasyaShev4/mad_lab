# 10. ДВУХФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ (ANOVA)

import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols


BASE = Path(__file__).parent
FILE_NAME = BASE / "SeoulBikeData.csv"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)


if not FILE_NAME.exists():
    raise FileNotFoundError(f"Файл не найден: {FILE_NAME}")


encodings = ["utf-8", "cp1252", "ISO-8859-1"]

for encoding in encodings:
    try:
        df = pd.read_csv(FILE_NAME, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    raise UnicodeDecodeError(
        "Не удалось определить кодировку файла",
        b"",
        0,
        1,
        "Проверьте кодировку SeoulBikeData.csv"
    )


df = df.rename(columns={
    "Rented Bike Count": "Rented_Bike_Count",
    "Temperature(°C)": "Temperature",
    "Humidity(%)": "Humidity"
})


print("\n" + "=" * 90)
print("10. ДВУХФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ (ANOVA)")
print("=" * 90)


# Зависимая переменная:
# Rented_Bike_Count
#
# Первый фактор:
# Seasons
#
# Второй фактор:
# Hour
#
# Взаимодействие факторов:
# Seasons * Hour


analysis_df = df[
    [
        "Rented_Bike_Count",
        "Seasons",
        "Hour"
    ]
].dropna()


print("\nЗависимая переменная: Rented_Bike_Count")
print("Первый фактор: Seasons")
print("Второй фактор: Hour")

print("\nКоличество наблюдений:", len(analysis_df))
print("Количество уровней Seasons:", analysis_df["Seasons"].nunique())
print("Количество уровней Hour:", analysis_df["Hour"].nunique())


print("\nГрадации факторов:")

print("\nSeasons:")
for level in sorted(analysis_df["Seasons"].unique()):
    print(f"- {level}")

print("\nHour:")
print(
    ", ".join(
        str(int(level))
        for level in sorted(analysis_df["Hour"].unique())
    )
)


# Построение модели двухфакторного ANOVA
# C() указывает, что Seasons и Hour являются категориальными факторами.
# * включает основные эффекты обоих факторов и их взаимодействие.

model = ols(
    "Rented_Bike_Count ~ C(Seasons) * C(Hour)",
    data=analysis_df
).fit()


anova_table = sm.stats.anova_lm(
    model,
    typ=2
)


print("\n" + "-" * 90)
print("РЕЗУЛЬТАТЫ ДВУХФАКТОРНОГО ANOVA")
print("-" * 90)

print(
    anova_table.round(6).to_string()
)


alpha = 0.05


print("\n" + "-" * 90)
print("ПРОВЕРКА СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ")
print("-" * 90)

print(f"Уровень значимости α: {alpha}")


effects = {
    "Seasons": "C(Seasons)",
    "Hour": "C(Hour)",
    "Seasons × Hour": "C(Seasons):C(Hour)"
}


for name, effect in effects.items():
    p_value = anova_table.loc[effect, "PR(>F)"]
    f_value = anova_table.loc[effect, "F"]

    print(f"\n{name}:")
    print(f"F-критерий: {f_value:.6f}")
    print(f"p-value: {p_value:.10f}")

    if p_value < alpha:
        print("Результат статистически значим.")
    else:
        print("Результат статистически незначим.")


print("\n" + "-" * 90)
print("ИНТЕРПРЕТАЦИЯ")
print("-" * 90)

print(
    "Двухфакторный ANOVA позволяет отдельно оценить влияние "
    "сезона, часа и их взаимодействия на количество арендованных велосипедов."
)

print(
    "Взаимодействие Seasons × Hour показывает, "
    "изменяется ли влияние часа суток в зависимости от сезона."
)


print("\n" + "=" * 90)
print("Двухфакторный ANOVA завершён.")
print("=" * 90)