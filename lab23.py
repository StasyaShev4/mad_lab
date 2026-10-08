import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path


# 5. Построить линейную регрессионную модель с фиктивными
# переменными для выбранных признаков.
# 6. Вывести график остатков. Оценить постоянство среднего и
# дисперсии.
# 7. Построить гистограмму стандартизированных остатков.
# 8. Записать уравнение регрессии.

try:
    df = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")
except UnicodeDecodeError:
    df = pd.read_csv("SeoulBikeData.csv", encoding="cp1252")

# Зависимая переменная
y = "Rented Bike Count"

# Числовые независимые переменные
numeric_features = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)"
]

# Категориальная переменная
categorical_feature = "Seasons"


# 3. Линейная регрессия

# C(Seasons) автоматически создаёт dummy-переменные.
# Один сезон используется как базовая категория,
# а остальные коэффициенты показывают отличие
# соответствующего сезона от базового.

formula = (
    'Q("Rented Bike Count") ~ '
    'Q("Hour") + Q("Temperature(°C)") + '
    'Q("Humidity(%)") + C(Seasons)'
)

model = smf.ols(formula=formula, data=df).fit()


# 4. Результаты регрессионной модели
print("\n" + "=" * 80)
print("ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("=" * 80)

print("\nСводка модели:")
print(model.summary())


# 5. Коэффициенты регрессии
print("\n" + "=" * 80)
print("КОЭФФИЦИЕНТЫ РЕГРЕССИИ")
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
print("УРАВНЕНИЕ РЕГРЕССИИ")
print("=" * 80)

equation = f"Rented Bike Count = {model.params.iloc[0]:.4f}"

for variable in model.params.index[1:]:
    coefficient = model.params[variable]

    if coefficient >= 0:
        equation += f" + {coefficient:.4f}*({variable})"
    else:
        equation += f" - {abs(coefficient):.4f}*({variable})"

print(equation)


# 7. Качество модели
print("\n" + "=" * 80)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
print("=" * 80)

print(f"R² = {model.rsquared:.6f}")
print(f"Скорректированный R² = {model.rsquared_adj:.6f}")
print(f"F-статистика = {model.fvalue:.6f}")
print(f"p-value F-теста = {model.f_pvalue:.6f}")


# 8. Значимость уравнения
alpha = 0.05

print("\n" + "=" * 80)
print("ЗНАЧИМОСТЬ УРАВНЕНИЯ")
print("=" * 80)

print(f"Уровень значимости alpha = {alpha}")
print(f"p-value F-теста = {model.f_pvalue:.6f}")

if model.f_pvalue < alpha:
    print("Вывод: регрессионная модель статистически значима.")
else:
    print("Вывод: нет оснований считать регрессионную модель статистически значимой.")


# 9. Значимость коэффициентов
print("\n" + "=" * 80)
print("ЗНАЧИМОСТЬ КОЭФФИЦИЕНТОВ")
print("=" * 80)

for variable, p_value in model.pvalues.items():
    if p_value < alpha:
        result = "значим"
    else:
        result = "не значим"

    print(
        f"{variable}: p-value = {p_value:.6f} "
        f"-> коэффициент {result}"
    )


# 10. Остатки
residuals = model.resid
standardized_residuals = model.get_influence().resid_studentized_internal
fitted_values = model.fittedvalues


# 11. График остатков
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 6))

plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Предсказанные значения Rented Bike Count")
plt.ylabel("Остатки")
plt.title("График остатков линейной регрессии")

plt.tight_layout()
plt.savefig(output_dir / "residuals_linear.png", dpi=300)
plt.show()


# 12. Гистограмма стандартизированных остатков
plt.figure(figsize=(10, 6))

plt.hist(
    standardized_residuals,
    bins=15,
    edgecolor="black"
)

plt.xlabel("Стандартизированные остатки")
plt.ylabel("Частота")
plt.title("Гистограмма стандартизированных остатков")

plt.tight_layout()
plt.savefig(output_dir / "standardized_residuals_linear.png", dpi=300)
plt.show()


# 13. Краткая информация об остатках
print("\n" + "=" * 80)
print("АНАЛИЗ ОСТАТКОВ")
print("=" * 80)

print(f"Среднее остатков: {residuals.mean():.6f}")
print(f"Стандартное отклонение остатков: {residuals.std():.6f}")
print(f"Минимальный остаток: {residuals.min():.6f}")
print(f"Максимальный остаток: {residuals.max():.6f}")

print("\nЛинейная регрессия построена.")
print("Графики сохранены в папке plots:")
print("  residuals_linear.png")
print("  standardized_residuals_linear.png")
