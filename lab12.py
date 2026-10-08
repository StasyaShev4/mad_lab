import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("forestfires.csv")

# Выбранные числовые признаки
features = ["temp", "RH", "wind", "area"]

print("=" * 80)
print("ДЕСКРИПТИВНЫЙ АНАЛИЗ")
print("=" * 80)

# 1. Основные статистические характеристики
for column in features:
    data = df[column]

    print(f"\n{'-' * 80}")
    print(f"ПРИЗНАК: {column}")
    print(f"{'-' * 80}")

    print(f"Количество наблюдений:       {len(data)}")
    print(f"Среднее:                     {data.mean():.4f}")
    print(f"Медиана:                     {data.median():.4f}")

    # Мода
    modes = data.mode()
    print(f"Мода:                        {', '.join(map(str, modes.tolist()[:10]))}")

    print(f"Минимум:                     {data.min():.4f}")
    print(f"Максимум:                    {data.max():.4f}")
    print(f"Размах:                      {data.max() - data.min():.4f}")
    print(f"Дисперсия:                   {data.var():.4f}")
    print(f"Стандартное отклонение:      {data.std():.4f}")

    # Квартили
    print(f"25-й процентиль (Q1):        {data.quantile(0.25):.4f}")
    print(f"50-й процентиль (Q2):        {data.quantile(0.50):.4f}")
    print(f"75-й процентиль (Q3):        {data.quantile(0.75):.4f}")

    # Асимметрия и эксцесс
    print(f"Асимметрия:                  {data.skew():.4f}")
    print(f"Эксцесс (Fisher):             {data.kurtosis():.4f}")

# 2. Проверка согласованности с нормальным распределением: критерий Шапиро–Уилка
print("\n" + "=" * 80)
print("ПРОВЕРКА НОРМАЛЬНОСТИ (КРИТЕРИЙ ШАПИРО–УИЛКА)")
print("=" * 80)

alpha = 0.05

for column in features:
    data = df[column].dropna()

    # Shapiro-Wilk рассчитан для выборки до 5000 наблюдений,
    # поэтому наши 517 наблюдений подходят.
    statistic, p_value = stats.shapiro(data)

    print(f"\n{column}:")
    print(f"  W = {statistic:.6f}")
    print(f"  p-value = {p_value:.10f}")

    if p_value > alpha:
        print("  Вывод: нет оснований отвергать гипотезу о нормальном распределении.")
    else:
        print("  Вывод: гипотеза о нормальном распределении отвергается.")

# 3. Краткий автоматический вывод
print("\n" + "=" * 80)
print("КРАТКИЙ ВЫВОД")
print("=" * 80)

print("""
Уровень значимости alpha = 0.05.

Если p-value > 0.05, выборка не имеет статистически значимых
отличий от нормального распределения.

Если p-value <= 0.05, гипотеза о нормальном распределении
отвергается.

Для дальнейшего корреляционного анализа выбор метода будет
определяться результатами проверки нормальности.
""")

print("Анализ завершён.")
