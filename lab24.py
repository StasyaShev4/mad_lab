import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path


# 11. Построить модель пуассоновской регрессии. Оценить качество
# модели (повторить пункты 6-10).

try:
    df = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")
except UnicodeDecodeError:
    df = pd.read_csv("SeoulBikeData.csv", encoding="cp1252")


formula = (
    'Q("Rented Bike Count") ~ '
    'Q("Hour") + Q("Temperature(°C)") + '
    'Q("Humidity(%)") + C(Seasons)'
)

print("=" * 80)
print("POISSON-РЕГРЕССИЯ")
print("=" * 80)

print("\nМодель:")
print("Rented Bike Count ~ Hour + Temperature + Humidity + C(Seasons)")
print("\nФункция связи: log(E[Y]) = линейный предиктор")

# 3. Построение модели
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson()
).fit()

# 4. Сводка
print("\n" + "=" * 80)
print("СВОДКА POISSON-РЕГРЕССИИ")
print("=" * 80)
print(model.summary())

# 5. Коэффициенты
print("\n" + "=" * 80)
print("КОЭФФИЦИЕНТЫ")
print("=" * 80)

coefficients = pd.DataFrame({
    "Коэффициент": model.params,
    "Стандартная ошибка": model.bse,
    "z-статистика": model.tvalues,
    "p-value": model.pvalues
})

coefficients["Значимость (p < 0.05)"] = np.where(
    coefficients["p-value"] < 0.05,
    "значим",
    "не значим"
)

print(coefficients.round(6).to_string())


# 6. Экспоненцированные коэффициенты
print("\n" + "=" * 80)
print("ЭКСПОНЕНЦИРОВАННЫЕ КОЭФФИЦИЕНТЫ exp(β)")
print("=" * 80)

for variable, value in np.exp(model.params).items():
    print(f"{variable}: {value:.6f}")


# 7. Уравнение
print("\n" + "=" * 80)
print("УРАВНЕНИЕ POISSON-РЕГРЕССИИ")
print("=" * 80)

equation = f"log(E[Y]) = {model.params.iloc[0]:.6f}"

for variable in model.params.index[1:]:
    coefficient = model.params[variable]
    sign = "+" if coefficient >= 0 else "-"
    equation += f" {sign} {abs(coefficient):.6f}*({variable})"

print(equation)
print("\nE[Y] = exp(линейный предиктор)")


# 8. Качество
print("\n" + "=" * 80)
print("ОЦЕНКА КАЧЕСТВА POISSON-МОДЕЛИ")
print("=" * 80)

dispersion = model.pearson_chi2 / model.df_resid

print(f"Deviance = {model.deviance:.6f}")
print(f"Pearson chi2 = {model.pearson_chi2:.6f}")
print(f"Log-Likelihood = {model.llf:.6f}")
print(f"AIC = {model.aic:.6f}")
print(f"Pearson chi2 / df_resid = {dispersion:.6f}")

if dispersion > 1:
    print("Наблюдается признак сверхдисперсии (overdispersion).")
elif dispersion < 1:
    print("Наблюдается признак недодисперсии.")
else:
    print("Отношение дисперсии к степени свободы близко к 1.")


# 9. Остатки
fitted_values = model.fittedvalues
residuals = df["Rented Bike Count"] - fitted_values
deviance_residuals = model.resid_deviance

print("\n" + "=" * 80)
print("АНАЛИЗ ОСТАТКОВ")
print("=" * 80)

print(f"Среднее обычных остатков: {residuals.mean():.6f}")
print(f"Стандартное отклонение: {residuals.std():.6f}")
print(f"Минимальный остаток: {residuals.min():.6f}")
print(f"Максимальный остаток: {residuals.max():.6f}")


# 10. Графики
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, deviance_residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Предсказанные значения Rented Bike Count")
plt.ylabel("Deviance residuals")
plt.title("График остатков Poisson-регрессии")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "residuals_poisson.png", dpi=300)
plt.show()

standardized_residuals = (
    deviance_residuals /
    np.sqrt(np.mean(deviance_residuals ** 2))
)

plt.figure(figsize=(10, 6))
plt.hist(standardized_residuals, bins=15, edgecolor="black")
plt.xlabel("Стандартизированные deviance residuals")
plt.ylabel("Частота")
plt.title("Гистограмма стандартизированных остатков Poisson-регрессии")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "standardized_residuals_poisson.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df["Rented Bike Count"], fitted_values, alpha=0.35)

min_value = min(df["Rented Bike Count"].min(), fitted_values.min())
max_value = max(df["Rented Bike Count"].max(), fitted_values.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Наблюдаемые значения")
plt.ylabel("Предсказанные значения")
plt.title("Наблюдаемые и предсказанные значения Poisson-модели")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "observed_vs_predicted_poisson.png", dpi=300)
plt.show()

print("\n" + "=" * 80)
print("POISSON-РЕГРЕССИЯ ПОСТРОЕНА")
print("=" * 80)
print("Графики сохранены в папке plots:")
print("  residuals_poisson.png")
print("  standardized_residuals_poisson.png")
print("  observed_vs_predicted_poisson.png")
