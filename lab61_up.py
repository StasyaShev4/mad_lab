from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. НАСТРОЙКИ
# ---------------------------------------------------------------------

DATA_FILE = "SeoulBikeData.csv"
OUTPUT_DIR = Path("plots_lab61")

WEATHER_COLUMNS = {
    "Влажность": ["Humidity(%)", "Humidity", "Влажность"],
    "Скорость ветра": ["Wind speed (m/s)", "Wind_Speed", "Wind speed", "Скорость ветра"],
    "Видимость": ["Visibility (10m)", "Visibility", "Видимость"],
    "Солнечная радиация": ["Solar Radiation (MJ/m2)", "Solar_Radiation", "Solar Radiation", "Солнечная радиация"],
    "Осадки": ["Rainfall(mm)", "Rainfall", "Осадки"],
    "Снегопад": ["Snowfall (cm)", "Snowfall", "Снегопад"],
}

TARGET_CANDIDATES = [
    "Rented Bike Count", "Rented_Bike_Count",
    "Количество арендованных велосипедов", "Количество велосипедов"
]

ALPHA = 0.05


# ---------------------------------------------------------------------
# 2. ЗАГРУЗКА ДАННЫХ
# ---------------------------------------------------------------------

def load_dataset():
    candidates = [
        Path(DATA_FILE),
        Path(__file__).resolve().parent / DATA_FILE,
        Path.cwd() / DATA_FILE,
    ]

    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        print(f"ОШИБКА: файл {DATA_FILE} не найден.")
        print("Положите CSV-файл в ту же папку, что и lab61_up.py.")
        sys.exit(1)

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin1"):
        try:
            data = pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
            if data.shape[1] > 1:
                print(f"Загружен файл: {file_path}")
                print(f"Кодировка: {encoding}")
                return data
        except Exception as error:
            last_error = error

    raise RuntimeError(f"Не удалось прочитать CSV-файл: {last_error}")


def find_column(data, candidates):
    normalized = {str(col).strip().lower(): col for col in data.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    compact = {
        str(col).strip().lower().replace(" ", "").replace("_", ""): col
        for col in data.columns
    }
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "").replace("_", "")
        if key in compact:
            return compact[key]
    return None


# ---------------------------------------------------------------------
# 3. СТАТИСТИЧЕСКИЕ ФУНКЦИИ
# ---------------------------------------------------------------------

def calculate_kmo(correlation):
    """Возвращает общий KMO и индивидуальные значения KMO."""
    inverse = np.linalg.pinv(correlation)
    diagonal = np.sqrt(np.clip(np.diag(inverse), 1e-12, None))
    partial = -inverse / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)

    corr = correlation.copy()
    np.fill_diagonal(corr, 0.0)

    corr_sq = corr ** 2
    partial_sq = partial ** 2
    denominator = corr_sq + partial_sq

    overall = corr_sq.sum() / denominator.sum()
    per_variable = corr_sq.sum(axis=0) / denominator.sum(axis=0)
    return float(overall), per_variable


def bartlett_test(correlation, n):
    """Критерий сферичности Бартлетта."""
    p = correlation.shape[0]
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0:
        raise ValueError("Корреляционная матрица не является положительно определённой.")
    statistic = -(n - 1 - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) // 2
    p_value = chi2.sf(statistic, df)
    return float(statistic), int(df), float(p_value)


def varimax(loadings, gamma=1.0, max_iter=500, tol=1e-7):
    """Ортогональное вращение Varimax."""
    rows, cols = loadings.shape
    rotation = np.eye(cols)
    previous = 0.0

    for _ in range(max_iter):
        rotated = loadings @ rotation
        gram = rotated.T @ rotated
        target = loadings.T @ (
            rotated ** 3 - (gamma / rows) * rotated @ np.diag(np.diag(gram))
        )
        u, singular, vh = np.linalg.svd(target, full_matrices=False)
        rotation = u @ vh
        objective = singular.sum()

        if previous and objective - previous < tol:
            break
        previous = objective

    return loadings @ rotation


def interpret_kmo(value):
    if value >= 0.90:
        return "превосходная"
    if value >= 0.80:
        return "очень хорошая"
    if value >= 0.70:
        return "хорошая"
    if value >= 0.60:
        return "приемлемая"
    if value >= 0.50:
        return "слабая, но допустимая"
    return "недостаточная"


# ---------------------------------------------------------------------
# 4. ОСНОВНОЙ АНАЛИЗ
# ---------------------------------------------------------------------

def main():
    # Создаётся только папка с графиками. Таблицы CSV не сохраняются.
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_dataset()

    target_col = find_column(data, TARGET_CANDIDATES)
    if target_col is None:
        print("\nНе удалось найти столбец с количеством арендованных велосипедов.")
        print("Найденные столбцы:", list(data.columns))
        sys.exit(1)

    selected = {}
    missing = []
    for russian_name, aliases in WEATHER_COLUMNS.items():
        col = find_column(data, aliases)
        if col is None:
            missing.append((russian_name, aliases))
        else:
            selected[russian_name] = col

    if missing:
        print("\nНе найдены необходимые погодные столбцы:")
        for name, aliases in missing:
            print(f"- {name}: варианты {aliases}")
        print("\nФактические названия столбцов:", list(data.columns))
        sys.exit(1)

    weather = pd.DataFrame({
        russian_name: pd.to_numeric(data[source_col], errors="coerce")
        for russian_name, source_col in selected.items()
    })
    target = pd.to_numeric(data[target_col], errors="coerce")

    combined = weather.copy()
    combined["Количество арендованных велосипедов"] = target
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    weather_clean = combined[list(WEATHER_COLUMNS.keys())]
    target_clean = combined["Количество арендованных велосипедов"]

    n, p = weather_clean.shape

    print("=" * 88)
    print("ФАКТОРНЫЙ АНАЛИЗ ПОГОДНЫХ УСЛОВИЙ")
    print("=" * 88)
    print(f"Количество наблюдений после удаления пропусков: {n}")
    print(f"Количество признаков в факторном анализе: {p}")
    print("\nВ факторный анализ включены только погодные признаки:")
    for i, name in enumerate(weather_clean.columns, 1):
        print(f"{i}. {name}")
    print("\nЗависимая переменная анализируется отдельно:")
    print("Количество арендованных велосипедов")
    print("Она НЕ участвует в расчёте корреляционной матрицы погодных признаков,")
    print("KMO, критерия Бартлетта, компонент и факторных нагрузок.")

    # 1. Дескриптивная статистика
    print("\n" + "=" * 88)
    print("1. ДЕСКРИПТИВНЫЙ АНАЛИЗ ПОГОДНЫХ ПРИЗНАКОВ")
    print("=" * 88)
    desc = weather_clean.describe().T
    desc["Асимметрия"] = weather_clean.skew()
    desc["Эксцесс"] = weather_clean.kurtosis()
    print(desc.round(4).to_string())

    # 2. Стандартизация
    scaler = StandardScaler()
    z = scaler.fit_transform(weather_clean)
    z_df = pd.DataFrame(z, columns=weather_clean.columns, index=weather_clean.index)

    print("\n" + "=" * 88)
    print("2. СТАНДАРТИЗАЦИЯ")
    print("=" * 88)
    print("Выполнено z-преобразование: z = (x - среднее) / стандартное отклонение.")
    print("Средние стандартизированных признаков:")
    print(z_df.mean().round(4).to_string())
    print("Стандартные отклонения:")
    print(z_df.std(ddof=0).round(4).to_string())

    # 3. Корреляционная матрица
    corr = weather_clean.corr()
    print("\n" + "=" * 88)
    print("3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ПОГОДНЫХ ПРИЗНАКОВ")
    print("=" * 88)
    print(corr.round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(p):
        for j in range(p):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Корреляционная матрица погодных признаков")
    fig.colorbar(image, ax=ax, label="Коэффициент корреляции")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_correlation_matrix.png", dpi=180)
    plt.close(fig)

    # 4. KMO
    kmo, kmo_each = calculate_kmo(corr.to_numpy())
    print("\n" + "=" * 88)
    print("4. МЕРA КАЙЗЕРА—МЕЙЕРА—ОЛКИНА (KMO)")
    print("=" * 88)
    print(f"Общий KMO = {kmo:.4f}")
    print(f"Интерпретация: {interpret_kmo(kmo)} пригодность данных.")
    kmo_table = pd.DataFrame({"Признак": corr.columns, "Индивидуальный KMO": kmo_each})
    print(kmo_table.round(4).to_string(index=False))

    # 5. Bartlett
    bartlett_stat, bartlett_df, bartlett_p = bartlett_test(corr.to_numpy(), n)
    print("\n" + "=" * 88)
    print("5. КРИТЕРИЙ СФЕРИЧНОСТИ БАРТЛЕТТА")
    print("=" * 88)
    print(f"χ² = {bartlett_stat:.4f}")
    print(f"Степени свободы = {bartlett_df}")
    print(f"p-value = {bartlett_p:.8g}")
    if bartlett_p < ALPHA:
        print("Нулевая гипотеза отвергается: корреляционная матрица отличается от единичной.")
    else:
        print("Нет оснований отвергнуть нулевую гипотезу.")
    print("KMO и Бартлетт оценивают пригодность именно погодных признаков.")

    # 6. PCA
    eigenvalues, eigenvectors = np.linalg.eigh(corr.to_numpy())
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained = eigenvalues / p
    cumulative = np.cumsum(explained)

    eigen_table = pd.DataFrame({
        "Компонента": np.arange(1, p + 1),
        "Собственное число": eigenvalues,
        "Доля дисперсии": explained,
        "Накопленная доля": cumulative,
    })
    print("\n" + "=" * 88)
    print("6. МЕТОД ГЛАВНЫХ КОМПОНЕНТ")
    print("=" * 88)
    print(eigen_table.round(4).to_string(index=False))

    # Scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    components = np.arange(1, p + 1)
    ax.plot(components, eigenvalues, marker="o")
    ax.axhline(1, linestyle="--", label="Собственное число = 1")
    ax.set_title("График каменистой осыпи")
    ax.set_xlabel("Номер компоненты")
    ax.set_ylabel("Собственное число")
    ax.set_xticks(components)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_scree_plot.png", dpi=180)
    plt.close(fig)

    # 7. Число факторов
    n_factors = max(1, int(np.sum(eigenvalues > 1.0)))
    print("\n" + "=" * 88)
    print("7. ВЫБОР КОЛИЧЕСТВА ФАКТОРОВ")
    print("=" * 88)
    print(f"Количество факторов по критерию Кайзера: {n_factors}")
    print("Окончательное решение следует сопоставить с точкой излома scree plot.")

    # 8–9. Loadings and Varimax
    unrotated = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])
    rotated = varimax(unrotated)

    factor_names = [f"Фактор {i}" for i in range(1, n_factors + 1)]
    loadings_before = pd.DataFrame(unrotated, index=weather_clean.columns, columns=factor_names)
    loadings_after = pd.DataFrame(rotated, index=weather_clean.columns, columns=factor_names)

    print("\n" + "=" * 88)
    print("8. МАТРИЦА ФАКТОРНЫХ НАГРУЗОК ДО ВРАЩЕНИЯ")
    print("=" * 88)
    print(loadings_before.round(4).to_string())

    print("\n" + "=" * 88)
    print("9. МАТРИЦА ФАКТОРНЫХ НАГРУЗОК ПОСЛЕ VARIMAX")
    print("=" * 88)
    print(loadings_after.round(4).to_string())

    # Графики факторных нагрузок
    for filename, matrix, title in [
        ("03_loadings_before_rotation.png", loadings_before, "Факторные нагрузки до вращения"),
        ("04_loadings_after_varimax.png", loadings_after, "Факторные нагрузки после Varimax"),
    ]:
        fig, ax = plt.subplots(figsize=(9, max(5, 0.7 * p)))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factor_names)
        ax.set_yticks(range(p))
        ax.set_yticklabels(matrix.index)
        for i in range(p):
            for j in range(n_factors):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Факторная нагрузка")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)

    # 10. Общности
    communalities = (rotated ** 2).sum(axis=1)
    communalities_table = pd.DataFrame({"Общность": communalities}, index=weather_clean.columns)
    print("\n" + "=" * 88)
    print("10. ОБЩНОСТИ")
    print("=" * 88)
    print(communalities_table.round(4).to_string())

    # 11. Информативность факторов
    rotated_ss = (rotated ** 2).sum(axis=0)
    rotated_share = rotated_ss / p
    rotated_cumulative = np.cumsum(rotated_share)
    factor_info = pd.DataFrame({
        "Сумма квадратов нагрузок": rotated_ss,
        "Доля дисперсии": rotated_share,
        "Накопленная доля": rotated_cumulative,
    }, index=factor_names)
    completeness = float(communalities.sum() / p)

    print("\n" + "=" * 88)
    print("11. ИНФОРМАТИВНОСТЬ ФАКТОРОВ")
    print("=" * 88)
    print(factor_info.round(4).to_string())
    print(f"\nПолнота факторизации: {completeness * 100:.2f}%")

    # 12. Факторные оценки и отдельная связь с целевой переменной
    score_coefficients = np.linalg.pinv(rotated.T @ rotated) @ rotated.T
    scores = z @ score_coefficients.T
    scores_df = pd.DataFrame(scores, columns=factor_names, index=weather_clean.index)

    print("\n" + "=" * 88)
    print("12. ОТДЕЛЬНЫЙ АНАЛИЗ СВЯЗИ ФАКТОРОВ СО СПРОСОМ НА ВЕЛОСИПЕДЫ")
    print("=" * 88)
    print("Количество арендованных велосипедов не участвовало в факторизации.")
    print("Ниже оценивается его связь с уже рассчитанными факторными оценками.")

    association_rows = []
    for factor in factor_names:
        r, p_value = pearsonr(scores_df[factor], target_clean)
        association_rows.append({
            "Фактор": factor,
            "Корреляция Пирсона": r,
            "p-value": p_value
        })

    association = pd.DataFrame(association_rows).set_index("Фактор")
    print("\nКорреляция факторных оценок с количеством арендованных велосипедов:")
    print(association.round(6).to_string())

    regression = LinearRegression()
    regression.fit(scores_df, target_clean)
    predicted = regression.predict(scores_df)
    r2 = r2_score(target_clean, predicted)

    regression_table = pd.DataFrame({
        "Показатель": ["R²", "Свободный член"] +
                      [f"Коэффициент {factor}" for factor in factor_names],
        "Значение": [r2, regression.intercept_] + list(regression.coef_)
    })
    print("\nМножественная линейная регрессия количества велосипедов на факторные оценки:")
    print(f"R² = {r2:.4f}")
    print(regression_table.round(6).to_string(index=False))

    # Scatter plots: один график на каждый фактор
    for i, factor in enumerate(factor_names, 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(scores_df[factor], target_clean, alpha=0.25, s=12)
        ax.set_title(f"{factor}: связь с количеством арендованных велосипедов")
        ax.set_xlabel("Факторная оценка")
        ax.set_ylabel("Количество арендованных велосипедов")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"{4 + i:02d}_{factor.lower().replace(' ', '_')}_vs_bikes.png",
            dpi=180
        )
        plt.close(fig)

    # 13. Итоговая сводка
    print("\n" + "=" * 88)
    print("13. ИТОГОВАЯ СВОДКА")
    print("=" * 88)
    print(f"Количество наблюдений: {n}")
    print(f"Количество погодных признаков в факторном анализе: {p}")
    print(f"KMO: {kmo:.4f} ({interpret_kmo(kmo)} пригодность)")
    print(f"Bartlett: χ²({bartlett_df}) = {bartlett_stat:.4f}, p = {bartlett_p:.8g}")
    print(f"Факторов по Кайзеру: {n_factors}")
    print(f"Полнота факторизации: {completeness * 100:.2f}%")
    print(f"R² отдельной регрессии спроса на факторные оценки: {r2:.4f}")
    print(f"\nГрафики сохранены в папке {OUTPUT_DIR}/")
    print("Таблицы выводятся в консоль и не сохраняются в CSV.")
    print("Анализ завершён.")


if __name__ == "__main__":
    main()
