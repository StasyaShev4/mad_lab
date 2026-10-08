import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, spearmanr, chi2_contingency
from pathlib import Path

# 2. Провести дескриптивный анализ, оценить согласованность выборок с
# нормальным распределением.
# 3. Построить гистограммы выбранных атрибутов. Длину интервала
# рассчитать с помощью формулы Стерджесса

BASE = Path(__file__).parent
FILE_NAME = BASE / "SeoulBikeData.csv"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

if not FILE_NAME.exists():
    print("ОШИБКА: SeoulBikeData.csv должен находиться рядом с lab26.py")
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

# Наши 4 количественные переменные
numeric_cols = [
    "Rented_Bike_Count",
    "Hour",
    "Temperature",
    "Humidity"
]

labels = {
    "Rented_Bike_Count": "Rented Bike Count",
    "Hour": "Hour",
    "Temperature": "Temperature (°C)",
    "Humidity": "Humidity (%)"
}

data = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()

print("=" * 90)
print("АНАЛИЗ ДАННЫХ Seoul Bike Sharing Demand")
print("=" * 90)
print(f"Количество наблюдений: {len(data)}")
print(f"Количество выбранных количественных признаков: {len(numeric_cols)}")

# 2. ДЕСКРИПТИВНЫЙ АНАЛИЗ
print("\n" + "=" * 90)
print("2. ДЕСКРИПТИВНЫЙ АНАЛИЗ И ПРОВЕРКА НОРМАЛЬНОСТИ")
print("=" * 90)

desc_rows = []

for col in numeric_cols:
    x = data[col]

    # Шапиро-Уилк: scipy имеет ограничение по размеру выборки,
    # поэтому для n > 5000 используем случайную подвыборку 5000.
    # Для нашей выборки n = 8760.
    if len(x) > 5000:
        x_test = x.sample(5000, random_state=42)
        shapiro_note = "подвыборка 5000"
    else:
        x_test = x
        shapiro_note = "вся выборка"

    W, p = shapiro(x_test)

    desc_rows.append({
        "Переменная": labels[col],
        "N": len(x),
        "Среднее": x.mean(),
        "Медиана": x.median(),
        "Std": x.std(),
        "Min": x.min(),
        "Max": x.max(),
        "Асимметрия": x.skew(),
        "Эксцесс": x.kurt(),
        "Shapiro W": W,
        "Shapiro p": p,
        "Нормальность (alpha=0.05)": "не подтверждена" if p < 0.05 else "не отвергается"
    })

desc = pd.DataFrame(desc_rows)

print("\nОписательные статистики:")
print(desc[[
    "Переменная", "N", "Среднее", "Медиана", "Std",
    "Min", "Max", "Асимметрия", "Эксцесс"
]].round(4).to_string(index=False))

print("\nКритерий Шапиро–Уилка:")
print(desc[[
    "Переменная", "Shapiro W", "Shapiro p",
    "Нормальность (alpha=0.05)"
]].round(6).to_string(index=False))

print("\nПримечание:", shapiro_note)


# 3. ГИСТОГРАММЫ + СТЕРДЖЕСС
print("\n" + "=" * 90)
print("3. ГИСТОГРАММЫ И ФОРМУЛА СТЕРДЖЕССА")
print("=" * 90)

n = len(data)
k_sturges = int(np.ceil(1 + 3.322 * np.log10(n)))

print(f"n = {n}")
print(f"k = ceil(1 + 3.322 * log10(n)) = {k_sturges}")

for col in numeric_cols:
    x = data[col]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=k_sturges, edgecolor="black")
    ax.set_title(f"Гистограмма: {labels[col]}")
    ax.set_xlabel(labels[col])
    ax.set_ylabel("Частота")
    ax.grid(axis="y", alpha=0.25)

    filename = f"hist_{col}.png"
    fig.tight_layout()
    fig.savefig(PLOTS / filename, dpi=150)
    plt.close(fig)

    print(f"Сохранено: plots/{filename}")


# 4. ТАБЛИЦА СОПРЯЖЕННОСТИ
print("\n" + "=" * 90)
print("4. ТАБЛИЦА СОПРЯЖЕННОСТИ")
print("=" * 90)

# Используем две исходные категориальные переменные:
# Seasons и Functioning Day
if "Seasons" not in df.columns or "Functioning Day" not in df.columns:
    print("В файле нет нужных категориальных переменных.")
else:
    contingency = pd.crosstab(
        df["Seasons"],
        df["Functioning Day"],
        margins=True
    )

    print("\nТаблица сопряженности Seasons × Functioning Day:")
    print(contingency.to_string())

    # Для критерия chi-square берём таблицу без итогов
    observed = pd.crosstab(
        df["Seasons"],
        df["Functioning Day"]
    )

    chi2, p_chi, dof, expected = chi2_contingency(observed)

    expected_df = pd.DataFrame(
        expected,
        index=observed.index,
        columns=observed.columns
    )

    print("\nОжидаемые частоты:")
    print(expected_df.round(2).to_string())

    print("\nКритерий χ² Пирсона:")
    print(f"χ² = {chi2:.6f}")
    print(f"df = {dof}")
    print(f"p-value = {p_chi:.6g}")

    if p_chi < 0.05:
        print("Вывод: H0 об отсутствии связи отвергается.")
        print("Между Seasons и Functioning Day есть статистически значимая связь.")
    else:
        print("Вывод: H0 об отсутствии связи не отвергается.")

    # График таблицы сопряженности
    observed.plot(kind="bar", figsize=(8, 5))
    plt.title("Seasons × Functioning Day")
    plt.xlabel("Seasons")
    plt.ylabel("Количество наблюдений")
    plt.xticks(rotation=0)
    plt.legend(title="Functioning Day")
    plt.tight_layout()
    plt.savefig(PLOTS / "contingency_Seasons_FunctioningDay.png", dpi=150)
    plt.close()


# 5. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
print("\n" + "=" * 90)
print("5. РАНГОВЫЙ КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 90)

normality_failed = (desc["Shapiro p"] < 0.05).any()

if normality_failed:
    method = "Спирмена"
    print("Так как нормальность выборок не подтверждена, используется коэффициент Спирмена.")
else:
    method = "Пирсона"
    print("Нормальность не отвергается, используется коэффициент Пирсона.")

corr = data.corr(method="spearman" if method == "Спирмена" else "pearson")

print(f"\nМатрица корреляций ({method}):")
print(corr.round(4).to_string())

# p-values
p_matrix = pd.DataFrame(
    np.zeros((len(numeric_cols), len(numeric_cols))),
    index=numeric_cols,
    columns=numeric_cols
)

for i, col1 in enumerate(numeric_cols):
    for j, col2 in enumerate(numeric_cols):
        if i == j:
            p_matrix.loc[col1, col2] = 0
        else:
            if method == "Спирмена":
                _, p = spearmanr(data[col1], data[col2])
            else:
                from scipy.stats import pearsonr
                _, p = pearsonr(data[col1], data[col2])

            p_matrix.loc[col1, col2] = p

print("\nP-value:")
print(p_matrix.map(lambda x: f"{x:.6g}").to_string())



# 6. ЗНАЧИМОСТЬ КОЭФФИЦИЕНТОВ
print("\n" + "=" * 90)
print("6. ОЦЕНКА ЗНАЧИМОСТИ КОЭФФИЦИЕНТОВ")
print("=" * 90)

pairs = []

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1 = numeric_cols[i]
        col2 = numeric_cols[j]

        coefficient = corr.loc[col1, col2]
        p = p_matrix.loc[col1, col2]

        pairs.append({
            "Признак 1": labels[col1],
            "Признак 2": labels[col2],
            "Коэффициент": coefficient,
            "p-value": p,
            "Значимость": "значим" if p < 0.05 else "не значим"
        })

pairs_df = pd.DataFrame(pairs)

print(pairs_df.round(6).to_string(index=False))


# 7. СИЛЬНЫЕ И СЛАБЫЕ КОРРЕЛЯЦИИ
print("\n" + "=" * 90)
print("7. СИЛЬНО- И СЛАБОКОРРЕЛИРОВАННЫЕ ПРИЗНАКИ")
print("=" * 90)

print("""
Для интерпретации используется абсолютное значение коэффициента:
|r| < 0.3       — слабая связь
0.3 <= |r| < 0.7 — умеренная связь
|r| >= 0.7      — сильная связь
""")

for _, row in pairs_df.iterrows():
    r = row["Коэффициент"]
    ar = abs(r)

    if ar < 0.3:
        strength = "слабая"
    elif ar < 0.7:
        strength = "умеренная"
    else:
        strength = "сильная"

    direction = "положительная" if r > 0 else "отрицательная"

    print(
        f"{row['Признак 1']} ↔ {row['Признак 2']}: "
        f"{r:.4f} — {strength}, {direction}."
    )

# Отдельно связь с зависимой переменной
target = "Rented_Bike_Count"

print("\nСвязи с Rented Bike Count:")
target_rows = []

for col in numeric_cols:
    if col == target:
        continue

    r = corr.loc[target, col]
    p = p_matrix.loc[target, col]

    target_rows.append({
        "Фактор": labels[col],
        "Коэффициент": r,
        "p-value": p
    })

target_df = pd.DataFrame(target_rows)
target_df["abs_r"] = target_df["Коэффициент"].abs()

print(
    target_df.drop(columns="abs_r")
    .sort_values("Коэффициент", key=lambda x: x.abs(), ascending=False)
    .round(6)
    .to_string(index=False)
)



# 8. ИНТЕРПРЕТАЦИЯ
print("\n" + "=" * 90)
print("8. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 90)

print("\nПо результатам анализа:")

if normality_failed:
    print("1. Распределения выбранных количественных признаков")
    print("   не соответствуют нормальному распределению по критерию Шапиро–Уилка.")
    print("2. Поэтому для корреляционного анализа выбран коэффициент Спирмена.")
else:
    print("1. Нормальность распределений не отвергается.")
    print("2. Для корреляционного анализа выбран коэффициент Пирсона.")

# Автоматически находим наиболее выраженную связь с Y
target_df_sorted = target_df.sort_values("abs_r", ascending=False)
strongest = target_df_sorted.iloc[0]

print(
    f"3. Наиболее выраженная связь с Rented Bike Count: "
    f"{strongest['Фактор']} (r = {strongest['Коэффициент']:.4f})."
)

print("4. Знак коэффициента показывает направление связи:")
print("   положительный — при увеличении одного признака второй имеет тенденцию увеличиваться;")
print("   отрицательный — при увеличении одного признака второй имеет тенденцию уменьшаться.")

print("5. Статистическая значимость оценивается по p-value при α = 0.05.")
print("6. Корреляция показывает статистическую взаимосвязь, но сама по себе не доказывает причинно-следственную связь.")
