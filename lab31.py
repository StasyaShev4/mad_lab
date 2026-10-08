import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from pathlib import Path

# 2. Провести дескриптивный анализ, оценить согласованность выборок с
# нормальным распределением.
# 3. Построить гистограммы выбранных атрибутов. Длину интервала
# рассчитать с помощью формулы Стерджесса.

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

# Выбранные переменные для дисперсионного анализа
numeric_cols = [
    "Rented_Bike_Count"
]

labels = {
    "Rented_Bike_Count": "Rented Bike Count"
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
        "Нормальность (alpha=0.05)": (
            "не подтверждена" if p < 0.05 else "не отвергается"
        )
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


# Дополнительная описательная статистика по уровням фактора
print("\nОписательная статистика Rented Bike Count по сезонам:")

season_stats = df.groupby("Seasons")["Rented_Bike_Count"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

season_stats.columns = [
    "N", "Среднее", "Медиана", "Std", "Min", "Max"
]

print(season_stats.round(4).to_string())

print("\nКоличество наблюдений по сезонам:")
print(df["Seasons"].value_counts().to_string())


# 3. ГИСТОГРАММА + СТЕРДЖЕСС
print("\n" + "=" * 90)
print("3. ГИСТОГРАММА И ФОРМУЛА СТЕРДЖЕССА")
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
