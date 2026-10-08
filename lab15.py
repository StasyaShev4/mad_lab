import pandas as pd
from scipy.stats import spearmanr

# Корреляционный анализ: коэффициент Спирмена
df = pd.read_csv("forestfires.csv")
variables = ["temp", "RH", "wind", "area"]

print("=" * 70)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ — КОЭФФИЦИЕНТ СПИРМЕНА")
print("=" * 70)

print("\nПоскольку исследуемые переменные не имеют нормального распределения,")
print("используем ранговый коэффициент корреляции Спирмена.")

# 1. Матрица коэффициентов Спирмена
corr_matrix = df[variables].corr(method="spearman")

print("\n" + "=" * 70)
print("1. МАТРИЦА КОРРЕЛЯЦИИ СПИРМЕНА")
print("=" * 70)
print(corr_matrix.round(4))

# 2. Подробный расчёт каждой пары
print("\n" + "=" * 70)
print("2. КОРРЕЛЯЦИЯ МЕЖДУ ПАРАМИ ПЕРЕМЕННЫХ")
print("=" * 70)

for i in range(len(variables)):
    for j in range(i + 1, len(variables)):
        var1 = variables[i]
        var2 = variables[j]

        pair = df[[var1, var2]].dropna()
        rho, p_value = spearmanr(pair[var1], pair[var2])

        print(f"\n{var1} <-> {var2}")
        print(f"Коэффициент Спирмена ρ = {rho:.4f}")
        print(f"p-value = {p_value:.6f}")

        if rho > 0:
            direction = "положительная"
        elif rho < 0:
            direction = "отрицательная"
        else:
            direction = "отсутствует"

        print(f"Направление связи: {direction}")

print("\n" + "=" * 70)
print("Готово.")
print("=" * 70)
