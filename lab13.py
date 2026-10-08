import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("forestfires.csv")
variables = ["temp", "RH", "wind", "area"]

# Папка для сохранения графиков
output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)

# 1. Число интервалов по формуле Стерджеса
# k = 1 + 3.322 * log10(n)
n = len(df)
k = int(np.ceil(1 + 3.322 * np.log10(n)))

print("=" * 60)
print("ЗАДАНИЕ 3. ГИСТОГРАММЫ И ФОРМУЛА СТЕРДЖЕСА")
print("=" * 60)

print(f"\nКоличество наблюдений n = {n}")
print(f"Число интервалов по формуле Стерджеса k = {k}")

# 2. Рассчитываем ширину интервала
# h = (x_max - x_min) / k
print("\n" + "=" * 60)
print("РАСЧЁТ ИНТЕРВАЛОВ")
print("=" * 60)

for variable in variables:
    x_min = df[variable].min()
    x_max = df[variable].max()
    data_range = x_max - x_min
    h = data_range / k

    print(f"\nПеременная: {variable}")
    print(f"Минимум = {x_min:.4f}")
    print(f"Максимум = {x_max:.4f}")
    print(f"Размах = {data_range:.4f}")
    print(f"Ширина интервала h = {h:.4f}")

# 3. Строим гистограммы
print("\n" + "=" * 60)
print("ПОСТРОЕНИЕ ГИСТОГРАММ")
print("=" * 60)

labels = {
    "temp": "Температура (°C)",
    "RH": "Относительная влажность (%)",
    "wind": "Скорость ветра",
    "area": "Площадь пожара"
}

for variable in variables:
    plt.figure(figsize=(8, 5))

    plt.hist(df[variable], bins=k, edgecolor="black")

    plt.title(f"Гистограмма: {labels[variable]}")
    plt.xlabel(labels[variable])
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.3)

    # Сохраняем график
    file_path = output_dir / f"hist_{variable}.png"
    plt.savefig(file_path, dpi=150, bbox_inches="tight")

    print(f"График сохранён: {file_path}")

    plt.show()
    plt.close()

print("\nГотово!")
print(f"Все гистограммы сохранены в папке: {output_dir}")
