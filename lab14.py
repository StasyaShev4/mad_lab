import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("forestfires.csv")

# Используем два категориальных признака:
# month — месяц
# day   — день недели
table = pd.crosstab(
    df["month"],
    df["day"],
    margins=True,
    margins_name="Итого"
)

print("=" * 60)
print("ТАБЛИЦА СОПРЯЖЁННОСТИ: MONTH × DAY")
print("=" * 60)
print(table)

# Таблица в процентах от общего количества наблюдений
percent_table = pd.crosstab(
    df["month"],
    df["day"],
    normalize=True
) * 100

print("\n" + "=" * 60)
print("ТАБЛИЦА СОПРЯЖЁННОСТИ В ПРОЦЕНТАХ")
print("=" * 60)
print(percent_table.round(2))

# Самые частые сочетания месяц + день
counts = (
    df.groupby(["month", "day"])
      .size()
      .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("САМЫЕ ЧАСТЫЕ СОЧЕТАНИЯ MONTH + DAY")
print("=" * 60)
print(counts.head(10))


# КРИТЕРИЙ ХИ-КВАДРАТ ПИРСОНА
# Проверка независимости month и day

# Строим таблицу без строки и столбца "Итого"
chi_table = pd.crosstab(
    df["month"],
    df["day"]
)

# Расчёт критерия хи-квадрат
chi2, p_value, degrees_of_freedom, expected = chi2_contingency(chi_table)

print("\n" + "=" * 60)
print("КРИТЕРИЙ ХИ-КВАДРАТ ПИРСОНА")
print("=" * 60)

print(f"Хи-квадрат (χ²): {chi2:.4f}")
print(f"Степени свободы: {degrees_of_freedom}")
print(f"p-value: {p_value:.6f}")

# Ожидаемые частоты
expected_table = pd.DataFrame(
    expected,
    index=chi_table.index,
    columns=chi_table.columns
)

print("\nОЖИДАЕМЫЕ ЧАСТОТЫ ПРИ НЕЗАВИСИМОСТИ:")
print(expected_table.round(2))

# Проверка гипотезы
alpha = 0.05

print("\nГИПОТЕЗЫ:")
print("H0: месяц и день недели независимы.")
print("H1: между месяцем и днём недели существует зависимость.")

print("\nРЕЗУЛЬТАТ:")

if p_value < alpha:
    print(f"p-value = {p_value:.6f} < {alpha}")
    print("H0 отвергается.")
    print("Обнаружена статистически значимая зависимость между месяцем и днём недели.")
else:
    print(f"p-value = {p_value:.6f} >= {alpha}")
    print("Нет оснований отвергать H0.")
    print("Статистически значимая зависимость между месяцем и днём недели не подтверждена.")
