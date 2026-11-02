import os
import time
import requests
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №5. ТЕХНИЧЕСКИЙ АНАЛИЗ
# BTC/USDT, дневные свечи
# ============================================================

DATA_FILE = "btc_usdt_daily.csv"
RSI_DATA_FILE = "btc_usdt_analysis_rsi.csv"
PATTERNS_FILE = "double_bottom_patterns.csv"

CANDLESTICKS_CHART = "btc_candlestick.png"
TREND_LEVELS_CHART = "btc_trend_levels.png"
RSI_CHART = "btc_rsi.png"
PATTERNS_CHART = "btc_patterns.png"
SUMMARY_CHART = "btc_full_analysis.png"

LIMIT = 365
SYMBOL = "BTCUSDT"
INTERVAL = "1d"


def download_quotes():
    """Загрузить дневные свечи с Binance; при ошибке использовать локальный CSV."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        raw = response.json()

        if not isinstance(raw, list) or not raw:
            raise ValueError("Binance вернул пустой или неожиданный ответ.")

        columns = [
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_asset_volume", "Number_of_trades",
            "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ]
        df = pd.DataFrame(raw, columns=columns)
        df["Date"] = pd.to_datetime(df["Open_time"], unit="ms").dt.normalize()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna().drop_duplicates(subset="Date").sort_values("Date")
        df.to_csv(DATA_FILE, index=False)
        print(f"Котировки загружены с Binance и сохранены в {DATA_FILE}")
        return df

    except Exception as exc:
        if os.path.exists(DATA_FILE):
            print(f"Не удалось загрузить данные с Binance: {exc}")
            print(f"Используется ранее сохранённый файл {DATA_FILE}")
            df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
            return prepare_data(df)
        raise RuntimeError(
            "Не удалось загрузить котировки и локальный CSV не найден. "
            "Проверьте интернет или наличие btc_usdt_daily.csv."
        ) from exc


def prepare_data(df):
    """Проверка и подготовка структуры данных."""
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В файле отсутствуют столбцы: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df.drop_duplicates(subset="Date").sort_values("Date")
    df = df.set_index("Date")
    df.index.name = "Date"
    if df.empty:
        raise ValueError("После очистки в таблице не осталось данных.")
    return df


def calculate_indicators(df):
    """Рассчитать MA20, MA50 и RSI(14)."""
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    df["RSI14"] = rsi
    return df


def determine_trend(df):
    last = df.iloc[-1]
    ma20, ma50, close = last["MA20"], last["MA50"], last["Close"]

    if pd.isna(ma20) or pd.isna(ma50):
        return "Недостаточно данных для определения тренда"

    if close > ma20 > ma50:
        return "Восходящий"
    if close < ma20 < ma50:
        return "Нисходящий"
    return "Боковой или неопределённый"


def find_support_resistance(df, tolerance=0.015, window=5):
    """Найти локальные экстремумы и сгруппировать близкие уровни."""
    data = df.copy()
    data["Local_Min"] = data["Low"].where(
        data["Low"] == data["Low"].rolling(2 * window + 1, center=True).min()
    )
    data["Local_Max"] = data["High"].where(
        data["High"] == data["High"].rolling(2 * window + 1, center=True).max()
    )

    current_price = float(data["Close"].iloc[-1])

    def cluster_levels(values):
        values = sorted(float(v) for v in values if pd.notna(v))
        clusters = []
        for value in values:
            if not clusters:
                clusters.append([value])
            else:
                center = float(np.mean(clusters[-1]))
                if abs(value - center) / center <= tolerance:
                    clusters[-1].append(value)
                else:
                    clusters.append([value])
        levels = [float(np.mean(c)) for c in clusters]
        return levels

    supports = [
        x for x in cluster_levels(data["Local_Min"].dropna().tolist())
        if x < current_price
    ]
    resistances = [
        x for x in cluster_levels(data["Local_Max"].dropna().tolist())
        if x > current_price
    ]

    # Сохраняем уровни для общего графика.
    return data, supports, resistances


def calculate_double_bottoms(df, min_gap=10, max_gap=90, tolerance=0.03, window=5):
    """Поиск пар локальных минимумов, похожих на фигуру «Двойное дно»."""
    data = df.copy()
    data["Local_Min"] = data["Low"].where(
        data["Low"] == data["Low"].rolling(2 * window + 1, center=True).min()
    )
    minima = data.dropna(subset=["Local_Min"])
    patterns = []

    dates = list(minima.index)
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            date1, date2 = dates[i], dates[j]
            gap = (date2 - date1).days
            if gap < min_gap:
                continue
            if gap > max_gap:
                break

            price1 = float(data.loc[date1, "Low"])
            price2 = float(data.loc[date2, "Low"])
            average_price = (price1 + price2) / 2
            difference_pct = abs(price1 - price2) / average_price

            if difference_pct > tolerance:
                continue

            between = data.loc[date1:date2]
            neckline = float(between["High"].max())

            after = data.loc[data.index > date2]
            breakout = bool((after["Close"] > neckline).any())
            breakout_date = None
            if breakout:
                breakout_date = after.index[after["Close"] > neckline][0]

            patterns.append({
                "First_Bottom_Date": date1.strftime("%Y-%m-%d"),
                "First_Bottom_Price": round(price1, 2),
                "Second_Bottom_Date": date2.strftime("%Y-%m-%d"),
                "Second_Bottom_Price": round(price2, 2),
                "Neckline": round(neckline, 2),
                "Days_Between": gap,
                "Difference_Percent": round(difference_pct * 100, 2),
                "Breakout_Detected": "Да" if breakout else "Нет",
                "Breakout_Date": breakout_date.strftime("%Y-%m-%d") if breakout_date is not None else "",
            })

    return data, pd.DataFrame(patterns)


def save_candlestick_chart(df):
    plot_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    mpf.plot(
        plot_df.tail(180),
        type="candle",
        volume=True,
        style="yahoo",
        title="BTC/USDT — дневные свечи (последние 180 дней)",
        ylabel="Цена, USDT",
        ylabel_lower="Объём",
        figsize=(14, 8),
        savefig=dict(fname=CANDLESTICKS_CHART, dpi=150, bbox_inches="tight"),
    )


def save_trend_levels_chart(df, supports, resistances):
    plot_df = df[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50"]].tail(180).copy()
    addplots = [
        mpf.make_addplot(plot_df["MA20"], color="blue", width=1.0),
        mpf.make_addplot(plot_df["MA50"], color="orange", width=1.0),
    ]

    # Горизонтальные уровни на ценовой панели.
    levels = []
    for level in supports:
        if plot_df["Low"].min() <= level <= plot_df["High"].max():
            levels.append(level)
    for level in resistances:
        if plot_df["Low"].min() <= level <= plot_df["High"].max():
            levels.append(level)

    kwargs = {}
    if levels:
        kwargs["hlines"] = dict(hlines=levels, linestyle="--", linewidths=0.8)

    mpf.plot(
        plot_df[["Open", "High", "Low", "Close", "Volume"]],
        type="candle",
        volume=True,
        style="yahoo",
        addplot=addplots,
        title="BTC/USDT — скользящие средние и уровни",
        ylabel="Цена, USDT",
        figsize=(14, 8),
        savefig=dict(fname=TREND_LEVELS_CHART, dpi=150, bbox_inches="tight"),
        **kwargs,
    )


def save_rsi_chart(df):
    plot_df = df[["Open", "High", "Low", "Close", "Volume"]].tail(180).copy()
    rsi = df["RSI14"].tail(180)
    addplots = [
        mpf.make_addplot(rsi, panel=2, ylabel="RSI", ylim=(0, 100), color="purple"),
        mpf.make_addplot(pd.Series(70, index=rsi.index), panel=2, color="red", linestyle="--"),
        mpf.make_addplot(pd.Series(30, index=rsi.index), panel=2, color="green", linestyle="--"),
    ]
    mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="yahoo",
        addplot=addplots,
        panel_ratios=(3, 1, 1),
        num_panels=3,
        title="BTC/USDT — свечи и RSI(14)",
        ylabel="Цена, USDT",
        figsize=(14, 9),
        savefig=dict(fname=RSI_CHART, dpi=150, bbox_inches="tight"),
    )


def save_patterns_chart(df, patterns):
    plot_df = df[["Open", "High", "Low", "Close", "Volume"]].tail(180).copy()
    bottom1 = pd.Series(np.nan, index=plot_df.index, dtype=float)
    bottom2 = pd.Series(np.nan, index=plot_df.index, dtype=float)
    neckpoints = pd.Series(np.nan, index=plot_df.index, dtype=float)

    if not patterns.empty:
        for _, row in patterns.iterrows():
            d1 = pd.to_datetime(row["First_Bottom_Date"])
            d2 = pd.to_datetime(row["Second_Bottom_Date"])
            if d1 in plot_df.index:
                bottom1.loc[d1] = row["First_Bottom_Price"]
            if d2 in plot_df.index:
                bottom2.loc[d2] = row["Second_Bottom_Price"]

            # Отмечаем линию шеи на дате максимума High между минимумами.
            section = df.loc[d1:d2]
            if not section.empty:
                neckline_date = section["High"].idxmax()
                if neckline_date in plot_df.index:
                    neckpoints.loc[neckline_date] = row["Neckline"]

    addplots = [
        mpf.make_addplot(bottom1, type="scatter", marker="v", markersize=80, color="blue"),
        mpf.make_addplot(bottom2, type="scatter", marker="v", markersize=80, color="purple"),
        mpf.make_addplot(neckpoints, type="scatter", marker="o", markersize=45, color="red"),
    ]
    mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="yahoo",
        addplot=addplots,
        title="BTC/USDT — возможные фигуры «Двойное дно»",
        ylabel="Цена, USDT",
        figsize=(14, 8),
        savefig=dict(fname=PATTERNS_CHART, dpi=150, bbox_inches="tight"),
    )


def save_combined_chart(df):
    """Единый график: цена, MA20/MA50, объём и RSI."""
    plot_df = df[["Open", "High", "Low", "Close", "Volume"]].tail(180).copy()
    ma20 = df["MA20"].tail(180)
    ma50 = df["MA50"].tail(180)
    rsi = df["RSI14"].tail(180)

    addplots = [
        mpf.make_addplot(ma20, color="blue", width=1.0),
        mpf.make_addplot(ma50, color="orange", width=1.0),
        mpf.make_addplot(rsi, panel=2, ylabel="RSI(14)", ylim=(0, 100), color="purple"),
        mpf.make_addplot(pd.Series(70, index=rsi.index), panel=2, color="red", linestyle="--"),
        mpf.make_addplot(pd.Series(30, index=rsi.index), panel=2, color="green", linestyle="--"),
    ]
    mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="yahoo",
        addplot=addplots,
        panel_ratios=(3, 1, 1),
        num_panels=3,
        title="BTC/USDT — полный технический анализ",
        ylabel="Цена, USDT",
        figsize=(15, 10),
        savefig=dict(fname=SUMMARY_CHART, dpi=160, bbox_inches="tight"),
    )


def main():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №5. ТЕХНИЧЕСКИЙ АНАЛИЗ")
    print("=" * 70)

    raw = download_quotes()
    df = prepare_data(raw)
    df = calculate_indicators(df)

    # Исходные свечи и общий CSV с индикаторами.
    df.reset_index().to_csv(RSI_DATA_FILE, index=False)
    save_candlestick_chart(df)

    # Тренд, поддержка и сопротивление.
    levels_data, supports, resistances = find_support_resistance(df)
    trend = determine_trend(df)
    save_trend_levels_chart(df, supports, resistances)

    # Паттерны.
    _, patterns = calculate_double_bottoms(df)
    patterns.to_csv(PATTERNS_FILE, index=False)

    # RSI и общий график.
    save_rsi_chart(df)
    save_patterns_chart(df, patterns)
    save_combined_chart(df)

    last = df.iloc[-1]
    print("\nОБЩИЕ РЕЗУЛЬТАТЫ")
    print("-" * 70)
    print(f"Количество свечей: {len(df)}")
    print(f"Период: {df.index.min().date()} — {df.index.max().date()}")
    print(f"Последняя цена закрытия: {last['Close']:.2f} USDT")
    print(f"MA20: {last['MA20']:.2f}")
    print(f"MA50: {last['MA50']:.2f}")
    print(f"Определённый тренд: {trend}")
    print(f"RSI(14): {last['RSI14']:.2f}")

    if last["RSI14"] >= 70:
        rsi_text = "зона выше 70 (часто трактуется как перекупленность)"
    elif last["RSI14"] <= 30:
        rsi_text = "зона ниже 30 (часто трактуется как перепроданность)"
    else:
        rsi_text = "между уровнями 30 и 70"
    print(f"Интерпретация RSI: {rsi_text}")

    print("\nУРОВНИ ПОДДЕРЖКИ (ниже текущей цены)")
    if supports:
        for level in supports:
            print(f"{level:.2f} USDT")
    else:
        print("Не найдены")

    print("\nУРОВНИ СОПРОТИВЛЕНИЯ (выше текущей цены)")
    if resistances:
        for level in resistances:
            print(f"{level:.2f} USDT")
    else:
        print("Не найдены")

    print("\nПОИСК «ДВОЙНОГО ДНА»")
    print(f"Количество локальных минимумов: {int(levels_data['Local_Min'].notna().sum())}")
    print(f"Количество найденных структур: {len(patterns)}")
    if not patterns.empty:
        print(patterns.to_string(index=False))

    print("\nСОХРАНЁННЫЕ ФАЙЛЫ")
    for filename in [
        DATA_FILE, RSI_DATA_FILE, PATTERNS_FILE,
        CANDLESTICKS_CHART, TREND_LEVELS_CHART, RSI_CHART,
        PATTERNS_CHART, SUMMARY_CHART
    ]:
        print(f"- {filename}")

    print("\nГотово. Файлы сохранены в текущей папке.")


if __name__ == "__main__":
    main()
