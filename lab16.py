# Задание 9. Ложная (мнимая) корреляция
# Данные Tyler Vigen: маргарин и разводы в Maine, 2000–2009.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.DataFrame({
    "Год": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    "Маргарин": [8.2, 7.0, 6.5, 5.3, 5.2, 4.0, 4.6, 4.5, 4.2, 3.7],
    "Разводы в Maine": [5.0, 4.7, 4.6, 4.4, 4.3, 4.1, 4.2, 4.2, 4.2, 4.1]
})
df["X"] = np.arange(1, 11)
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 70)
print("ЗАДАНИЕ 9. ЛОЖНАЯ (МНИМАЯ) КОРРЕЛЯЦИЯ")
print("=" * 70)
print("\n9.1. ИСХОДНЫЕ ДАННЫЕ")
print(df[["Год", "X", "Маргарин", "Разводы в Maine"]].to_string(index=False))

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    df["X"],
    df["Маргарин"],
    marker="o",
    color="blue",
    label="Маргарин"
)

ax1.set_xlabel("Год")
ax1.set_ylabel("Маргарин")

ax1.set_xticks(df["X"])
ax1.set_xticklabels(df["Год"])

ax2 = ax1.twinx()

ax2.plot(
    df["X"],
    df["Разводы в Maine"],
    marker="s",
    color="orange",
    label="Разводы в Maine"
)

ax2.set_ylabel("Разводы в Maine")

ax1.set_title("Потребление маргарина и уровень разводов в Maine")

ax1.grid(True, alpha=0.3)

# Общая легенда
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()

ax1.legend(l1 + l2, lab1 + lab2)

fig.tight_layout()

fig.savefig("graphs/9_1_исходные_ряды.png", dpi=200)

plt.close(fig)

x = df["X"].to_numpy()
reg_m = stats.linregress(x, df["Маргарин"])
reg_d = stats.linregress(x, df["Разводы в Maine"])

def eq(reg):
    sign = "+" if reg.intercept >= 0 else "-"
    return f"y = {reg.slope:.4f}x {sign} {abs(reg.intercept):.4f}"

print("\n9.2. ЛИНЕЙНЫЕ ТРЕНДЫ")
print("Маргарин:", eq(reg_m), f", R² = {reg_m.rvalue**2:.4f}")
print("Разводы:", eq(reg_d), f", R² = {reg_d.rvalue**2:.4f}")

df["Маргарин предсказанный"] = reg_m.intercept + reg_m.slope * x
df["Разводы предсказанные"] = reg_d.intercept + reg_d.slope * x

fig, ax1 = plt.subplots(figsize=(10, 6))

# Маргарин — синяя основная линия + синяя пунктирная линия тренда
ax1.plot(x, df["Маргарин"], marker="o", label="Маргарин")
ax1.plot(
    x,
    df["Маргарин предсказанный"],
    linestyle="--",
    color="blue",
    label="Тренд маргарина"
)

ax1.set_xlabel("Год")
ax1.set_ylabel("Маргарин")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Год"])

# Разводы — оранжевая основная линия + красная пунктирная линия тренда
ax2 = ax1.twinx()

ax2.plot(
    x,
    df["Разводы в Maine"],
    marker="s",
    color="orange",
    label="Разводы в Maine"
)

ax2.plot(
    x,
    df["Разводы предсказанные"],
    linestyle="--",
    color="red",
    label="Тренд разводов"
)

ax2.set_ylabel("Разводы в Maine")

ax1.set_title("Линейные тренды исследуемых рядов")
ax1.grid(True, alpha=0.3)

# Общая легенда
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2)

fig.tight_layout()
fig.savefig("graphs/9_2_линии_тренда.png", dpi=200)

plt.close(fig)

print("\n9.3. ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ")
print(df[["Год", "X", "Маргарин предсказанный", "Разводы предсказанные"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

df["Остаток маргарина"] = df["Маргарин"] - df["Маргарин предсказанный"]
df["Остаток разводов"] = df["Разводы в Maine"] - df["Разводы предсказанные"]
print("\n9.4. ОСТАТКИ")
print(df[["Год", "Остаток маргарина", "Остаток разводов"]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

std_m = df["Остаток маргарина"].std(ddof=1)
std_d = df["Остаток разводов"].std(ddof=1)
df["Станд. остаток маргарина"] = df["Остаток маргарина"] / std_m
df["Станд. остаток разводов"] = df["Остаток разводов"] / std_d
print("\n9.5. СТАНДАРТИЗИРОВАННЫЕ ОСТАТКИ")
print(df[["Год", "Станд. остаток маргарина", "Станд. остаток разводов"]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

for col, title, fname in [
    ("Остаток маргарина", "График остатков для потребления маргарина", "9_6_остатки_маргарина.png"),
    ("Остаток разводов", "График остатков для уровня разводов в Maine", "9_6_остатки_разводов.png")]:
    plt.figure(figsize=(9, 5))
    plt.plot(x, df[col], marker="o")
    plt.axhline(0, linestyle="--")
    plt.xticks(x, df["Год"])
    plt.xlabel("Год")
    plt.ylabel("Остаток")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphs/" + fname, dpi=200)
    plt.close()

for col, title, fname in [
    ("Станд. остаток маргарина", "Гистограмма стандартизированных остатков маргарина", "9_6_гистограмма_маргарина.png"),
    ("Станд. остаток разводов", "Гистограмма стандартизированных остатков разводов", "9_6_гистограмма_разводов.png")]:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=5, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Стандартизированный остаток")
    plt.ylabel("Частота")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("graphs/" + fname, dpi=200)
    plt.close()

r, p = stats.pearsonr(df["Остаток маргарина"], df["Остаток разводов"])
print("\n9.7. КОРРЕЛЯЦИЯ ОСТАТКОВ")
print(f"r = {r:.6f}")
print(f"p-value = {p:.10f}")

n = len(df)
dfree = n - 2
t_calc = r * np.sqrt(dfree / (1 - r**2))
t_critical = stats.t.ppf(1 - 0.05 / 2, dfree)
print("\n9.8. ОЦЕНКА ЗНАЧИМОСТИ")
print(f"n = {n}")
print(f"df = {dfree}")
print("alpha = 0.05")
print(f"t расчётное = {t_calc:.6f}")
print(f"t критическое = {t_critical:.6f}")
print(f"p-value = {p:.10f}")
print("Вывод:", "корреляция остатков статистически значима." if p < 0.05 else "корреляция остатков статистически незначима.")

df.to_csv("results/lab16_полные_результаты.csv", index=False, sep=";", decimal=",")