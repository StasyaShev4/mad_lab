import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path


# 12. Построить нелинейную регрессионную модель с фиктивными
# переменными (повторить пункты 6-10). Можно использовать
# библиотечные функции для линейной свертки нелинейных элементов,
# например, y=aox3+a1x2*logx

try:
    df = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")
except UnicodeDecodeError:
    df = pd.read_csv("SeoulBikeData.csv", encoding="cp1252")


# ============================================================
# 2. Нелинейная модель
# ============================================================
# В отличие от lab22 здесь добавляем квадратичные и кубические
# члены для количественных факторов.
#
# Модель остаётся линейной по коэффициентам, но является
# нелинейной по исходным факторам.
#
# C(Seasons) создаёт dummy-переменные.
#
# Используем:
# Hour, Hour², Hour³
# Temperature, Temperature²
# Humidity, Humidity²
# Seasons (dummy)

formula = (
    'Q("Rented Bike Count") ~ '
    'Q("Hour") + I(Q("Hour")**2) + I(Q("Hour")**3) + '
    'Q("Temperature(°C)") + I(Q("Temperature(°C)")**2) + '
    'Q("Humidity(%)") + I(Q("Humidity(%)")**2) + '
    'C(Seasons)'
)


# 3. Построение модели
print("=" * 80)
print("НЕЛИНЕЙНАЯ РЕГРЕССИЯ С DUMMY-ПЕРЕМЕННЫМИ")
print("=" * 80)

print("\nМодель содержит:")
print("- Hour, Hour², Hour³")
print("- Temperature, Temperature²")
print("- Humidity, Humidity²")
print("- dummy-переменные Seasons")

model = smf.ols(
    formula=formula,
    data=df
).fit()


# 4. Сводка модели
print("\n" + "=" * 80)
print("СВОДКА НЕЛИНЕЙНОЙ РЕГРЕССИИ")
print("=" * 80)

print(model.summary())


# 5. Коэффициенты и их значимость
print("\n" + "=" * 80)
print("КОЭФФИЦИЕНТЫ НЕЛИНЕЙНОЙ МОДЕЛИ")
print("=" * 80)

coefficients = pd.DataFrame({
    "Коэффициент": model.params,
    "Стандартная ошибка": model.bse,
    "t-статистика": model.tvalues,
    "p-value": model.pvalues
})

coefficients["Значимость (p < 0.05)"] = np.where(
    coefficients["p-value"] < 0.05,
    "значим",
    "не значим"
)

print(coefficients.round(6).to_string())


# 6. Уравнение регрессии
print("\n" + "=" * 80)
print("УРАВНЕНИЕ НЕЛИНЕЙНОЙ РЕГРЕССИИ")
print("=" * 80)

params = model.params

equation = f"Y = {params.iloc[0]:.6f}"

for variable in params.index[1:]:
    coefficient = params[variable]

    if coefficient >= 0:
        equation += f" + {coefficient:.6f}*({variable})"
    else:
        equation += f" - {abs(coefficient):.6f}*({variable})"

print(equation)


# 7. Оценка качества модели
print("\n" + "=" * 80)
print("ОЦЕНКА КАЧЕСТВА НЕЛИНЕЙНОЙ МОДЕЛИ")
print("=" * 80)

print(f"R² = {model.rsquared:.6f}")
print(f"Скорректированный R² = {model.rsquared_adj:.6f}")
print(f"F-статистика = {model.fvalue:.6f}")
print(f"p-value F-теста = {model.f_pvalue:.6g}")
print(f"AIC = {model.aic:.6f}")
print(f"BIC = {model.bic:.6f}")

if model.f_pvalue < 0.05:
    print("Вывод: модель статистически значима.")
else:
    print("Вывод: модель статистически не значима.")


# 8. Значимость коэффициентов
print("\n" + "=" * 80)
print("ЗНАЧИМОСТЬ КОЭФФИЦИЕНТОВ")
print("=" * 80)

for variable, p_value in model.pvalues.items():

    if p_value < 0.05:
        result = "значим"
    else:
        result = "не значим"

    print(
        f"{variable}: p-value = {p_value:.6g} -> {result}"
    )


# 9. Остатки
fitted_values = model.fittedvalues
residuals = model.resid

# Внутренние стандартизированные остатки
standardized_residuals = model.get_influence().resid_studentized_internal

print("\n" + "=" * 80)
print("АНАЛИЗ ОСТАТКОВ")
print("=" * 80)

print(f"Среднее остатков: {residuals.mean():.6f}")
print(f"Стандартное отклонение: {residuals.std():.6f}")
print(f"Минимальный остаток: {residuals.min():.6f}")
print(f"Максимальный остаток: {residuals.max():.6f}")


# 10. Папка для графиков
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)


# 11. График остатков
plt.figure(figsize=(10, 6))

plt.scatter(
    fitted_values,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Предсказанные значения Rented Bike Count")
plt.ylabel("Остатки")
plt.title("График остатков нелинейной регрессии")
plt.grid(alpha=0.3)

residual_plot = output_dir / "residuals_nonlinear.png"

plt.tight_layout()
plt.savefig(residual_plot, dpi=300)
plt.show()

print(f"\nГрафик остатков сохранён: {residual_plot}")


# 12. Гистограмма стандартизированных остатков
plt.figure(figsize=(10, 6))

plt.hist(
    standardized_residuals,
    bins=15,
    edgecolor="black"
)

plt.xlabel("Стандартизированные остатки")
plt.ylabel("Частота")
plt.title("Гистограмма стандартизированных остатков нелинейной регрессии")
plt.grid(axis="y", alpha=0.3)

residual_hist = output_dir / "standardized_residuals_nonlinear.png"

plt.tight_layout()
plt.savefig(residual_hist, dpi=300)
plt.show()

print(f"Гистограмма остатков сохранена: {residual_hist}")


# 13. Наблюдаемые и предсказанные значения
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Rented Bike Count"],
    fitted_values,
    alpha=0.35
)

min_value = min(
    df["Rented Bike Count"].min(),
    fitted_values.min()
)

max_value = max(
    df["Rented Bike Count"].max(),
    fitted_values.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Наблюдаемые значения")
plt.ylabel("Предсказанные значения")
plt.title("Наблюдаемые и предсказанные значения нелинейной модели")
plt.grid(alpha=0.3)

prediction_plot = output_dir / "observed_vs_predicted_nonlinear.png"

plt.tight_layout()
plt.savefig(prediction_plot, dpi=300)
plt.show()

print(f"График наблюдаемых и предсказанных значений сохранён: {prediction_plot}")


# 14. Итог
print("\n" + "=" * 80)
print("НЕЛИНЕЙНАЯ РЕГРЕССИЯ ПОСТРОЕНА")
print("=" * 80)

print("Зависимая переменная: Rented Bike Count")
print("Факторы: Hour, Temperature, Humidity, Seasons")
print("Добавлены квадратичные и кубические члены.")

print("\nГрафики сохранены в папке plots:")
print("  residuals_nonlinear.png")
print("  standardized_residuals_nonlinear.png")
print("  observed_vs_predicted_nonlinear.png")
