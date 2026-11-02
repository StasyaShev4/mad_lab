from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
FILE_NAME = BASE / "SeoulBikeData.csv"
PLOTS = BASE / "plots_lab71"
PLOTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
ALPHA = 0.05
RANDOM_STATE = 42

def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)

def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Ортогональное вращение Varimax для матрицы факторных нагрузок."""
    p, k = Phi.shape
    R = np.eye(k)
    d_old = 0
    for _ in range(q):
        L = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L)))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old and d - d_old < tol:
            break
        d_old = d
    return Phi @ R

print("=" * 88)
print("ЛАБОРАТОРНАЯ РАБОТА №7. ВИЗУАЛИЗАЦИЯ ДАННЫХ — ЧАСТЬ 1")
print("=" * 88)

if not FILE_NAME.exists():
    raise SystemExit(f"ОШИБКА: положи SeoulBikeData.csv рядом с lab71.py\nПуть: {FILE_NAME}")

df = None
for enc in ("utf-8", "cp1252", "ISO-8859-1"):
    try:
        df = pd.read_csv(FILE_NAME, encoding=enc)
        break
    except UnicodeDecodeError:
        pass
if df is None:
    raise SystemExit("Не удалось прочитать CSV-файл.")

df = df.rename(columns={
    "Rented Bike Count": "Rented_Bike_Count",
    "Temperature(°C)": "Temperature",
    "Humidity(%)": "Humidity",
    "Wind speed (m/s)": "Wind_Speed",
    "Visibility (10m)": "Visibility",
    "Dew point temperature(°C)": "Dew_Point_Temperature",
    "Solar Radiation (MJ/m2)": "Solar_Radiation",
    "Rainfall(mm)": "Rainfall",
    "Snowfall (cm)": "Snowfall",
    "Functioning Day": "Functioning_Day",
})
required = [
    "Date", "Rented_Bike_Count", "Hour", "Temperature", "Humidity",
    "Wind_Speed", "Visibility", "Dew_Point_Temperature",
    "Solar_Radiation", "Rainfall", "Snowfall", "Seasons",
    "Holiday", "Functioning_Day",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"В датасете отсутствуют столбцы: {missing}")

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
numeric_cols = [
    "Rented_Bike_Count", "Hour", "Temperature", "Humidity", "Wind_Speed",
    "Visibility", "Dew_Point_Temperature", "Solar_Radiation", "Rainfall", "Snowfall"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

start_rows = len(df)
df = df.dropna(subset=["Date"] + numeric_cols).copy()
df = df[df["Rented_Bike_Count"] >= 0].copy()
print(f"Исходных строк: {start_rows}; после очистки: {len(df)}; удалено: {start_rows-len(df)}")
print(f"Период: {df.Date.min().date()} — {df.Date.max().date()}")
print(f"Пропусков в числовых столбцах: {int(df[numeric_cols].isna().sum().sum())}")

# 1. ОПИСАТЕЛЬНЫЙ АНАЛИЗ
print("\n1. ДЕСКРИПТИВНЫЙ АНАЛИЗ")
desc = df[numeric_cols].describe().T
desc["Медиана"] = df[numeric_cols].median()
desc["Асимметрия"] = df[numeric_cols].skew()
desc["Эксцесс"] = df[numeric_cols].kurt()
print(desc.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["Rented_Bike_Count"], bins="sturges", kde=True, color="steelblue", ax=ax)
ax.set(title="Распределение количества аренд", xlabel="Количество аренд за час", ylabel="Частота")
save_plot(fig, "01_rental_distribution.png")

hourly = df.groupby("Hour", as_index=False)["Rented_Bike_Count"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=hourly, x="Hour", y="Rented_Bike_Count", marker="o", ax=ax)
ax.set(title="Среднее количество аренд по часам", xlabel="Час суток", ylabel="Среднее количество аренд")
ax.set_xticks(range(24))
save_plot(fig, "02_hourly_rentals.png")

# 2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
print("\n2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ПИРСОНА")
corr_cols = [
    "Rented_Bike_Count", "Temperature", "Humidity", "Wind_Speed",
    "Visibility", "Dew_Point_Temperature", "Solar_Radiation", "Rainfall", "Snowfall"
]
corr = df[corr_cols].corr()
print(corr.round(3).to_string())
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title("Тепловая карта корреляций")
save_plot(fig, "03_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Temperature", y="Rented_Bike_Count", hue="Seasons", alpha=.4, s=22, ax=ax)
ax.set(title="Температура и количество аренд", xlabel="Температура (°C)", ylabel="Количество аренд")
save_plot(fig, "04_temperature_vs_rentals.png")

# 3. ANOVA — различия спроса между сезонами
print("\n3. ДИСПЕРСИОННЫЙ АНАЛИЗ (ОДНОФАКТОРНАЯ ANOVA)")
season_order = [s for s in ["Spring", "Summer", "Autumn", "Winter"] if s in df["Seasons"].unique()]
season_groups = [df.loc[df["Seasons"] == s, "Rented_Bike_Count"].values for s in season_order]
if len(season_groups) >= 2:
    f_stat, p_value = stats.f_oneway(*season_groups)
    print("Средние значения по сезонам:")
    print(df.groupby("Seasons")["Rented_Bike_Count"].agg(["count", "mean", "std"]).reindex(season_order).round(2))
    print(f"F-критерий = {f_stat:.4f}; p-value = {p_value:.6g}; alpha = {ALPHA}")
    print("Вывод:", "есть статистически значимые различия между средними" if p_value < ALPHA else "статистически значимые различия не выявлены")
else:
    print("Недостаточно сезонных групп для ANOVA.")

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="Seasons", y="Rented_Bike_Count", order=season_order, ax=ax)
ax.set(title="Распределение аренд по сезонам", xlabel="Сезон", ylabel="Количество аренд за час")
save_plot(fig, "05_rentals_by_season_boxplot.png")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=df, x="Seasons", y="Rented_Bike_Count", order=season_order, errorbar=None, color="steelblue", ax=ax)
ax.set(title="Среднее количество аренд по сезонам", xlabel="Сезон", ylabel="Среднее количество аренд")
save_plot(fig, "06_mean_rentals_by_season.png")

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="Functioning_Day", y="Rented_Bike_Count", ax=ax)
ax.set(title="Аренда в функционирующие и нерабочие дни", xlabel="Работа системы проката", ylabel="Количество аренд")
save_plot(fig, "07_rentals_by_functioning_day.png")

# 4. РЕГРЕССИОННЫЙ АНАЛИЗ
print("\n4. МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ")
reg_features = ["Temperature", "Humidity", "Wind_Speed", "Visibility",
                "Dew_Point_Temperature", "Solar_Radiation", "Rainfall", "Snowfall", "Hour"]
X = df[reg_features].copy()
y = df["Rented_Bike_Count"].copy()
# Хронологическое разделение: последние 20% наблюдений — тестовая выборка.
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"Размер обучающей выборки: {len(X_train)}; тестовой: {len(X_test)}")
print(f"R² на тестовой выборке = {r2:.4f}")
print(f"RMSE на тестовой выборке = {rmse:.2f}")
coef_table = pd.DataFrame({"Признак": reg_features, "Коэффициент": reg.coef_})
print("Коэффициенты модели (знак показывает направление линейной связи при прочих равных):")
print(coef_table.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, pred, alpha=.35, s=18)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax.plot(lims, lims, "r--", label="Идеальное совпадение")
ax.set(xlabel="Фактическое количество аренд", ylabel="Предсказанное количество аренд",
       title="Регрессия: фактические и предсказанные значения")
ax.legend()
save_plot(fig, "08_regression_actual_vs_predicted.png")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=coef_table, x="Коэффициент", y="Признак", color="steelblue", ax=ax)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Коэффициенты линейной регрессии")
save_plot(fig, "09_regression_coefficients.png")

# 5. ФАКТОРНЫЙ АНАЛИЗ (PCA + Varimax)
print("\n5. ФАКТОРНЫЙ АНАЛИЗ ПОГОДНЫХ ПРИЗНАКОВ")
factor_features = ["Humidity", "Wind_Speed", "Visibility", "Solar_Radiation", "Rainfall", "Snowfall"]
factor_data = df[factor_features].copy()
scaler = StandardScaler()
Z = scaler.fit_transform(factor_data)
pca_all = PCA().fit(Z)
eigenvalues = pca_all.explained_variance_  # для стандартизованных переменных
n_factors = max(1, min(2, int(np.sum(eigenvalues > 1))))
pca = PCA(n_components=n_factors)
scores = pca.fit_transform(Z)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
rotated = varimax(loadings)
communalities = np.sum(rotated**2, axis=1)
print("Собственные значения:", np.round(eigenvalues, 4))
print(f"Число факторов по критерию Кайзера (с ограничением максимум 2 для наглядности): {n_factors}")
loading_df = pd.DataFrame(rotated, index=factor_features,
                          columns=[f"Фактор {i+1}" for i in range(n_factors)])
loading_df["Общность"] = communalities
print("Повернутые факторные нагрузки и общности:")
print(loading_df.round(3).to_string())
print(f"Доля дисперсии выбранных компонент: {pca.explained_variance_ratio_.sum()*100:.2f}%")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(eigenvalues)+1), eigenvalues, marker="o")
ax.axhline(1, color="red", linestyle="--", label="Критерий Кайзера (λ = 1)")
ax.set(title="График собственных значений (Scree plot)", xlabel="Номер компоненты", ylabel="Собственное значение")
ax.set_xticks(range(1, len(eigenvalues)+1))
ax.legend()
save_plot(fig, "10_factor_scree_plot.png")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(loading_df.iloc[:, :n_factors], annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Факторные нагрузки после Varimax")
save_plot(fig, "11_factor_loadings_varimax.png")

if n_factors >= 2:
    factor_df = pd.DataFrame(scores, columns=["Фактор 1", "Фактор 2"])
    factor_df["Количество аренд"] = y.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=factor_df.sample(min(2500, len(factor_df)), random_state=RANDOM_STATE),
                    x="Фактор 1", y="Фактор 2", hue="Количество аренд", palette="viridis",
                    alpha=.6, s=20, legend=False, ax=ax)
    ax.set_title("Наблюдения в пространстве двух факторов")
    save_plot(fig, "12_factor_scores_scatter.png")

# 6. ВРЕМЕННОЙ РЯД
print("\n6. АНАЛИЗ ВРЕМЕННОГО РЯДА")
daily = df.groupby("Date", as_index=False)["Rented_Bike_Count"].sum().sort_values("Date")
daily["Rolling_7d"] = daily["Rented_Bike_Count"].rolling(7, min_periods=1).mean()
print(f"Число дней: {len(daily)}; окно скользящего среднего: 7 дней")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily["Date"], daily["Rented_Bike_Count"], alpha=.4, linewidth=.8, label="Сумма за день")
ax.plot(daily["Date"], daily["Rolling_7d"], linewidth=2, label="Скользящее среднее (7 дней)")
ax.set(title="Динамика аренды велосипедов по дням", xlabel="Дата", ylabel="Количество аренд")
ax.legend()
save_plot(fig, "13_daily_time_series.png")

# 7. КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ
print("\n7. КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ")
total = int(df["Rented_Bike_Count"].sum())
mean_hour = df["Rented_Bike_Count"].mean()
peak_hour = int(hourly.loc[hourly["Rented_Bike_Count"].idxmax(), "Hour"])
best_season = df.groupby("Seasons")["Rented_Bike_Count"].mean().idxmax()
print(f"Всего аренд за период: {total:,}")
print(f"Среднее количество аренд за час: {mean_hour:.2f}")
print(f"Час с максимальным средним спросом: {peak_hour}:00")
print(f"Сезон с максимальным средним спросом: {best_season}")
print("\nГОТОВО. CSV-файлы не создаются.")
print(f"Графики сохранены в: {PLOTS}")
for path in sorted(PLOTS.glob("*.png")):
    print(" -", path.name)
