from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("dataset-HeartDisease") / "processed.cleveland.data"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл датасета: {DATA_PATH.resolve()}\n"
        "Положите папку dataset-HeartDisease рядом с lab43.py."
    )

df = pd.read_csv(DATA_PATH, header=None, na_values="?")
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
classes = [0, 1, 2, 3, 4]

X = df[features].copy()
y = pd.to_numeric(df["target"], errors="raise").astype(int)


# 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 72)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 72)
print(f"Обучающая выборка: {len(X_train_raw)} объектов")
print(f"Тестовая выборка:  {len(X_test_raw)} объектов")
print("Для всех сравнений используются одни и те же train/test-индексы.")


# 3. ОБРАБОТКА ПРОПУСКОВ БЕЗ УТЕЧКИ ДАННЫХ
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    columns=features,
    index=X_train_raw.index
)
X_test = pd.DataFrame(
    imputer.transform(X_test_raw),
    columns=features,
    index=X_test_raw.index
)

# Для SVM scaler обучается только на обучающей выборке.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=features,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=features,
    index=X_test.index
)


# 4. ОБУЧЕНИЕ ОСНОВНЫХ МОДЕЛЕЙ
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
tree_model.fit(X_train, y_train)

svm_base = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)
svm_model = CalibratedClassifierCV(
    svm_base,
    method="sigmoid",
    cv=5
)
svm_model.fit(X_train_scaled, y_train)

# Предсказания на train и test считаются отдельно.
predictions = {
    ("Дерево решений", "Обучающая"): tree_model.predict(X_train),
    ("Дерево решений", "Тестовая"): tree_model.predict(X_test),
    ("SVM", "Обучающая"): svm_model.predict(X_train_scaled),
    ("SVM", "Тестовая"): svm_model.predict(X_test_scaled),
}
probabilities = {
    ("Дерево решений", "Обучающая"): tree_model.predict_proba(X_train),
    ("Дерево решений", "Тестовая"): tree_model.predict_proba(X_test),
    ("SVM", "Обучающая"): svm_model.predict_proba(X_train_scaled),
    ("SVM", "Тестовая"): svm_model.predict_proba(X_test_scaled),
}
true_labels = {
    "Обучающая": y_train,
    "Тестовая": y_test,
}


# 5. ФУНКЦИЯ РАСЧЁТА МЕТРИК
def calculate_metrics(y_true, y_pred, y_score):
    """Возвращает набор метрик для многоклассовой классификации."""
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
    }
    try:
        result["ROC_AUC_macro"] = roc_auc_score(
            y_true, y_score, labels=classes,
            multi_class="ovr", average="macro"
        )
    except ValueError:
        result["ROC_AUC_macro"] = float("nan")
    return result


# 6. СРАВНЕНИЕ КАЧЕСТВА НА ОБУЧАЮЩЕЙ И ТЕСТОВОЙ ВЫБОРКАХ
metric_rows = []
for model_name in ["Дерево решений", "SVM"]:
    for split_name in ["Обучающая", "Тестовая"]:
        key = (model_name, split_name)
        row = {
            "Model": model_name,
            "Dataset": split_name,
            **calculate_metrics(
                true_labels[split_name],
                predictions[key],
                probabilities[key]
            )
        }
        metric_rows.append(row)

metrics_df = pd.DataFrame(metric_rows)

print("\n" + "=" * 72)
print("СРАВНЕНИЕ МЕТРИК: TRAIN И TEST")
print("=" * 72)
print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
metrics_df.to_csv(PLOTS_DIR / "classification_metrics_train_test.csv", index=False)


# 7. ДЕТАЛЬНЫЕ ОТЧЁТЫ ПО КЛАССАМ — ОТДЕЛЬНО TRAIN И TEST
print("\n" + "=" * 72)
print("ДЕТАЛЬНЫЕ ОТЧЁТЫ ПО КЛАССАМ")
print("=" * 72)

for model_name in ["Дерево решений", "SVM"]:
    for split_name in ["Обучающая", "Тестовая"]:
        key = (model_name, split_name)
        print(f"\n{model_name} — {split_name.lower()} выборка")
        print(classification_report(
            true_labels[split_name],
            predictions[key],
            labels=classes,
            zero_division=0,
            digits=4
        ))

# 8. МАТРИЦЫ ОШИБОК — ЧЕТЫРЕ ОТДЕЛЬНЫХ ГРАФИКА
print("\n" + "=" * 72)
print("МАТРИЦЫ ОШИБОК")
print("=" * 72)

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"\n{title}")
    print("Строки — истинные классы; столбцы — предсказанные классы.")
    print(pd.DataFrame(
        cm,
        index=[f"Истинный {c}" for c in classes],
        columns=[f"Предсказан {c}" for c in classes]
    ).to_string())

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(title)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    filepath = PLOTS_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранён: {filepath}")


for model_name, prefix in [
    ("Дерево решений", "tree"),
    ("SVM", "svm")
]:
    for split_name, split_suffix in [
        ("Обучающая", "train"),
        ("Тестовая", "test")
    ]:
        key = (model_name, split_name)
        save_confusion_matrix(
            true_labels[split_name],
            predictions[key],
            f"{model_name} — {split_name.lower()} выборка",
            f"cm_{prefix}_{split_suffix}.png"
        )


# 9. ВИЗУАЛИЗАЦИЯ САМОГО ДЕРЕВА РЕШЕНИЙ
plt.figure(figsize=(24, 13))
plot_tree(
    tree_model,
    feature_names=features,
    class_names=[str(c) for c in classes],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Дерево решений — обученная модель")
plt.tight_layout()
tree_plot_path = PLOTS_DIR / "decision_tree.png"
plt.savefig(tree_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nВизуализация дерева решений сохранена: {tree_plot_path}")


# 10. ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ ДЕРЕВА РЕШЕНИЙ
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": tree_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n" + "=" * 72)
print("ВАЖНОСТЬ ПРИЗНАКОВ — ДЕРЕВО РЕШЕНИЙ")
print("=" * 72)
print(feature_importance.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
feature_importance.to_csv(PLOTS_DIR / "feature_importance_tree.csv", index=False)

plt.figure(figsize=(10, 7))
plot_data = feature_importance.sort_values("importance", ascending=True)
plt.barh(plot_data["feature"], plot_data["importance"])
plt.xlabel("Важность")
plt.ylabel("Признак")
plt.title("Важность признаков — дерево решений")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importance_tree.png", dpi=300, bbox_inches="tight")
plt.close()


# 10.1. ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ SVM
print("\n" + "=" * 72)
print("ВАЖНОСТЬ ПРИЗНАКОВ — SVM (PERMUTATION IMPORTANCE)")
print("=" * 72)

# Оценка важности признаков на тестовой выборке
svm_importance = permutation_importance(
    svm_model,
    X_test_scaled,
    y_test,
    scoring="f1_macro",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

svm_feature_importance = pd.DataFrame({
    "feature": features,
    "importance": svm_importance.importances_mean,
    "std": svm_importance.importances_std
}).sort_values("importance", ascending=False)

print("\nВажность признаков SVM:")
print(
    svm_feature_importance.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

# Сохранение таблицы
svm_feature_importance.to_csv(
    PLOTS_DIR / "feature_importance_svm.csv",
    index=False
)

# Построение графика
plt.figure(figsize=(10, 7))

plot_data = svm_feature_importance.sort_values(
    "importance",
    ascending=True
)

plt.barh(
    plot_data["feature"],
    plot_data["importance"],
    xerr=plot_data["std"],
    capsize=3
)

plt.xlabel("Важность (снижение F1-macro)")
plt.ylabel("Признак")
plt.title("Важность признаков — SVM (Permutation Importance)")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "feature_importance_svm.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nСохранено:")
print("plots/feature_importance_svm.csv")
print("plots/feature_importance_svm.png")


# 11. ИЗМЕНЕНИЕ МАКСИМАЛЬНОЙ ГЛУБИНЫ ДЕРЕВА
print("\n" + "=" * 72)
print("СРАВНЕНИЕ ГЛУБИНЫ ДЕРЕВА РЕШЕНИЙ")
print("=" * 72)

depth_rows = []

# Список глубин: от простой модели к сложной, включая None (без ограничения)
depth_values = [2, 3, 4, 5, 6, 7, 10, None]

for depth in depth_values:
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Метрики на ТЕСТОВОЙ выборке
    pred_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)
    metrics_test = calculate_metrics(y_test, pred_test, proba_test)

    # Метрики на ОБУЧАЮЩЕЙ выборке (для контроля переобучения)
    pred_train = model.predict(X_train)
    proba_train = model.predict_proba(X_train)
    metrics_train = calculate_metrics(y_train, pred_train, proba_train)

    depth_label = depth if depth is not None else "None"
    depth_rows.append({
        "Model": f"Дерево решений, max_depth={depth_label}",
        "Train_Accuracy": metrics_train["Accuracy"],
        "Test_Accuracy":  metrics_test["Accuracy"],
        "Test_Precision_macro": metrics_test["Precision_macro"],
        "Test_Recall_macro":    metrics_test["Recall_macro"],
        "Test_F1_macro":        metrics_test["F1_macro"],
        "Test_ROC_AUC_macro":   metrics_test["ROC_AUC_macro"],
    })

depth_comparison = pd.DataFrame(depth_rows)
print(depth_comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

depth_comparison.to_csv(PLOTS_DIR / "tree_parameter_comparison.csv", index=False)

# 12. ИСКЛЮЧЕНИЕ ПРИЗНАКОВ С НУЛЕВОЙ ВАЖНОСТЬЮ
print("\n" + "=" * 72)
print("СРАВНЕНИЕ ДЕРЕВА ДО И ПОСЛЕ ИСКЛЮЧЕНИЯ ПРИЗНАКОВ")
print("=" * 72)

# Важно: список признаков определяется по основной модели.
# В этой версии исключаются признаки с нулевой важностью.
insignificant_features = feature_importance.loc[
    feature_importance["importance"] == 0, "feature"
].tolist()

print("Признаки с нулевой важностью:", insignificant_features or "нет")
selected_features = [f for f in features if f not in insignificant_features]
print(f"Количество исходных признаков: {len(features)}")
print(f"Количество оставшихся признаков: {len(selected_features)}")

# Используем те же индексы train/test, что и в основной модели.
X_train_reduced_raw = X_train_raw[selected_features]
X_test_reduced_raw = X_test_raw[selected_features]

reduced_imputer = SimpleImputer(strategy="median")
X_train_reduced = pd.DataFrame(
    reduced_imputer.fit_transform(X_train_reduced_raw),
    columns=selected_features,
    index=X_train_reduced_raw.index
)
X_test_reduced = pd.DataFrame(
    reduced_imputer.transform(X_test_reduced_raw),
    columns=selected_features,
    index=X_test_reduced_raw.index
)

tree_reduced = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
tree_reduced.fit(X_train_reduced, y_train)
pred_reduced = tree_reduced.predict(X_test_reduced)
proba_reduced = tree_reduced.predict_proba(X_test_reduced)

original_test_metrics = calculate_metrics(
    y_test,
    predictions[("Дерево решений", "Тестовая")],
    probabilities[("Дерево решений", "Тестовая")]
)
reduced_test_metrics = calculate_metrics(y_test, pred_reduced, proba_reduced)

feature_comparison = pd.DataFrame([
    {
        "Model": "Дерево: все признаки",
        "Количество признаков": len(features),
        **original_test_metrics
    },
    {
        "Model": "Дерево: без признаков с нулевой важностью",
        "Количество признаков": len(selected_features),
        **reduced_test_metrics
    }
])
print(feature_comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
feature_comparison.to_csv(PLOTS_DIR / "tree_feature_removal_comparison.csv", index=False)

# Матрица ошибок для сокращённой модели, чтобы можно было сравнить её с исходной.
save_confusion_matrix(
    y_test,
    pred_reduced,
    "Дерево решений без признаков с нулевой важностью — тестовая выборка",
    "cm_tree_reduced_test.png"
)

print("\n" + "=" * 72)
print("РАБОТА ЗАВЕРШЕНА")
print(f"Все таблицы и графики сохранены в папке: {PLOTS_DIR.resolve()}")
print("=" * 72)
