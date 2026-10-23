import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 7. Решить задачу классификации двумя методами (нейронная сеть,
# дерево решений, метод опорных векторов, дискриминантная
# функция и т.д.). Аргументировать выбор значений параметров метода.
# 8. Оценить качество построенных моделей. Проанализировать confusion
# matrix, метрики на тестовом и обучающем множествах (accuracy,
# precision, recall, AUC, F-мера; макро- и микроусреднение). Провести
# сравнительный анализ решений.
# 9. Оценить значимость признаков.
# 10. Изменить параметры одного из методов, оценить полученные результаты.
# 11. В случае выявления малозначимых признаков – удалить их из
# массива; оценить изменение точности классификатора при решении
# задачи с исключенными из датасета признаками (для одного из методов).

df = pd.read_csv(
    "dataset-HeartDisease/processed.cleveland.data",
    header=None,
    na_values="?"
)

df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X = df[features]
y = df["target"]

# 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 70)
print("7. КЛАССИФИКАЦИЯ")
print("=" * 70)
print(f"\nОбучающая выборка: {len(X_train)} объектов")
print(f"Тестовая выборка: {len(X_test)} объектов")


# 3. ОБРАБОТКА ПРОПУСКОВ
imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=features,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=features,
    index=X_test.index
)

# 4. СТАНДАРТИЗАЦИЯ ДЛЯ SVM
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),
    columns=features,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed),
    columns=features,
    index=X_test.index
)

# 5. ДЕРЕВО РЕШЕНИЙ
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

tree_model.fit(X_train_imputed, y_train)
y_pred_tree = tree_model.predict(X_test_imputed)
tree_scores = tree_model.predict_proba(X_test_imputed)

# 6. SVM
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
y_pred_svm = svm_model.predict(X_test_scaled)
svm_scores = svm_model.predict_proba(X_test_scaled)


# 7. ПРЕДСКАЗАНИЯ
print("\nДерево решений:")
print("Предсказания:", y_pred_tree)

print("\nSVM:")
print("Предсказания:", y_pred_svm)

print(f"\nКоличество предсказаний дерева: {len(y_pred_tree)}")
print(f"Количество предсказаний SVM: {len(y_pred_svm)}")

# 8. ФУНКЦИЯ ОЦЕНКИ МОДЕЛИ
def evaluate_model(model_name, y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    f1_micro = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    roc_auc = roc_auc_score(
        y_true, y_score, multi_class="ovr", average="macro"
    )

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)
    print(f"Accuracy:              {accuracy:.4f}")

    print("\nMacro averaging:")
    print(f"Precision (macro):     {precision_macro:.4f}")
    print(f"Recall (macro):        {recall_macro:.4f}")
    print(f"F1-score (macro):      {f1_macro:.4f}")

    print("\nMicro averaging:")
    print(f"Precision (micro):     {precision_micro:.4f}")
    print(f"Recall (micro):        {recall_micro:.4f}")
    print(f"F1-score (micro):      {f1_micro:.4f}")

    print(f"\nROC-AUC (macro, OVR):  {roc_auc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
        "Precision_micro": precision_micro,
        "Recall_micro": recall_micro,
        "F1_micro": f1_micro,
        "ROC_AUC_macro": roc_auc
    }

# 9. ОЦЕНКА ДВУХ МОДЕЛЕЙ
tree_results = evaluate_model(
    "Дерево решений", y_test, y_pred_tree, tree_scores
)

svm_results = evaluate_model(
    "SVM", y_test, y_pred_svm, svm_scores
)

# 10. СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК
results = pd.DataFrame([tree_results, svm_results])

print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
print("=" * 70)
print(results.to_string(index=False))

# 11. ДЕТАЛЬНЫЙ ОТЧЁТ ПО КАЖДОМУ КЛАССУ
print("\n" + "=" * 70)
print("ДЕТАЛЬНЫЙ ОТЧЁТ ПО КЛАССАМ")
print("=" * 70)

print("\nДерево решений:")
print(classification_report(
    y_test, y_pred_tree,
    labels=[0, 1, 2, 3, 4],
    zero_division=0
))

print("\nSVM:")
print(classification_report(
    y_test, y_pred_svm,
    labels=[0, 1, 2, 3, 4],
    zero_division=0
))

# 12. МАТРИЦЫ ОШИБОК
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

cm_tree = confusion_matrix(y_test, y_pred_tree, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_tree, annot=True, fmt="d", cmap="Blues",
    xticklabels=[0, 1, 2, 3, 4],
    yticklabels=[0, 1, 2, 3, 4]
)
plt.title("Матрица ошибок — Дерево решений")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.tight_layout()
tree_cm_path = os.path.join(plots_dir, "confusion_matrix_tree.png")
plt.savefig(tree_cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nМатрица ошибок дерева сохранена: {tree_cm_path}")

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_svm, annot=True, fmt="d", cmap="Blues",
    xticklabels=[0, 1, 2, 3, 4],
    yticklabels=[0, 1, 2, 3, 4]
)
plt.title("Матрица ошибок — SVM")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.tight_layout()
svm_cm_path = os.path.join(plots_dir, "confusion_matrix_svm.png")
plt.savefig(svm_cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Матрица ошибок SVM сохранена: {svm_cm_path}")

# 13. ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ ДЕРЕВА
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": tree_model.feature_importances_
}).sort_values(by="importance", ascending=False)


# 8. ОЦЕНКА КАЧЕСТВА ПОСТРОЕННЫХ МОДЕЛЕЙ
print("\n" + "=" * 70)
print("8. ОЦЕНКА КАЧЕСТВА ПОСТРОЕННЫХ МОДЕЛЕЙ")
print("=" * 70)


def evaluate_model_full(model_name, dataset_name, y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_micro = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    f1_micro = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true, y_score, multi_class="ovr", average="macro"
    )

    print("\n" + "-" * 70)
    print(f"{model_name} — {dataset_name}")
    print("-" * 70)
    print(f"Accuracy:              {accuracy:.4f}")

    print("\nMacro averaging:")
    print(f"Precision (macro):     {precision_macro:.4f}")
    print(f"Recall (macro):        {recall_macro:.4f}")
    print(f"F1-score (macro):      {f1_macro:.4f}")

    print("\nMicro averaging:")
    print(f"Precision (micro):     {precision_micro:.4f}")
    print(f"Recall (micro):        {recall_micro:.4f}")
    print(f"F1-score (micro):      {f1_micro:.4f}")

    print(f"\nROC-AUC (macro, OVR):  {roc_auc:.4f}")

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
        "Precision_micro": precision_micro,
        "Recall_micro": recall_micro,
        "F1_micro": f1_micro,
        "ROC_AUC_macro": roc_auc
    }


# Оценка дерева на обучающей и тестовой выборках
tree_train_results = evaluate_model_full(
    "Дерево решений",
    "Обучающая выборка",
    y_train,
    tree_model.predict(X_train_imputed),
    tree_model.predict_proba(X_train_imputed)
)

tree_test_results = evaluate_model_full(
    "Дерево решений",
    "Тестовая выборка",
    y_test,
    y_pred_tree,
    tree_scores
)

# Оценка SVM на обучающей и тестовой выборках
svm_train_results = evaluate_model_full(
    "SVM",
    "Обучающая выборка",
    y_train,
    svm_model.predict(X_train_scaled),
    svm_model.predict_proba(X_train_scaled)
)

svm_test_results = evaluate_model_full(
    "SVM",
    "Тестовая выборка",
    y_test,
    y_pred_svm,
    svm_scores
)


# Сравнительная таблица
results_8 = pd.DataFrame([
    tree_train_results,
    tree_test_results,
    svm_train_results,
    svm_test_results
])

print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
print("=" * 70)
print(results_8.to_string(index=False))

# Сохраняем результаты в отдельный файл,
# чтобы не перезаписывать существующие файлы.
metrics_path = os.path.join(plots_dir, "classification_metrics_train_test.csv")
results_8.to_csv(metrics_path, index=False)
print(f"\nТаблица метрик сохранена: {metrics_path}")


# CONFUSION MATRIX ДЛЯ TRAIN И TEST
print("\n" + "=" * 70)
print("МАТРИЦЫ ОШИБОК")
print("=" * 70)


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1, 2, 3, 4],
        yticklabels=[0, 1, 2, 3, 4]
    )

    plt.title(title)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()

    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Сохранено: {filepath}")


save_confusion_matrix(
    y_train,
    tree_model.predict(X_train_imputed),
    "Дерево решений — обучающая выборка",
    "cm_tree_train.png"
)

save_confusion_matrix(
    y_test,
    y_pred_tree,
    "Дерево решений — тестовая выборка",
    "cm_tree_test.png"
)

save_confusion_matrix(
    y_train,
    svm_model.predict(X_train_scaled),
    "SVM — обучающая выборка",
    "cm_svm_train.png"
)

save_confusion_matrix(
    y_test,
    y_pred_svm,
    "SVM — тестовая выборка",
    "cm_svm_test.png"
)


print("\n" + "=" * 70)
print("ДЕТАЛЬНЫЙ ОТЧЁТ ПО КЛАССАМ — ТЕСТОВАЯ ВЫБОРКА")
print("=" * 70)

print("\nДерево решений:")
print(classification_report(
    y_test,
    y_pred_tree,
    labels=[0, 1, 2, 3, 4],
    zero_division=0
))

print("SVM:")
print(classification_report(
    y_test,
    y_pred_svm,
    labels=[0, 1, 2, 3, 4],
    zero_division=0
))

print("\n" + "=" * 70)
print("9. ВАЖНОСТЬ ПРИЗНАКОВ — ДЕРЕВО РЕШЕНИЙ")
print("=" * 70)

print(feature_importance.to_string(index=False))

# Используем новый файл, чтобы не конфликтовать с уже существующим.
feature_importance_path = os.path.join(
    plots_dir,
    "feature_importance_tree_v2.csv"
)

feature_importance.to_csv(feature_importance_path, index=False)
print(f"\nТаблица важности признаков сохранена: {feature_importance_path}")

# График важности признаков
plt.figure(figsize=(9, 6))
plt.barh(
    feature_importance["feature"][::-1],
    feature_importance["importance"][::-1]
)
plt.xlabel("Важность")
plt.ylabel("Признак")
plt.title("Важность признаков — Дерево решений")
plt.tight_layout()

importance_plot_path = os.path.join(
    plots_dir,
    "feature_importance_tree_v2.png"
)
plt.savefig(importance_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"График важности признаков сохранён: {importance_plot_path}")


print("\n" + "=" * 70)
print("10. ИЗМЕНЕНИЕ ПАРАМЕТРОВ ДЕРЕВА РЕШЕНИЙ")
print("=" * 70)

# Исходная модель: max_depth = 5
tree_original = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

# Используем те же обработанные данные,
# что и для основной модели
tree_original.fit(X_train_imputed, y_train)

y_pred_original = tree_original.predict(X_test_imputed)
y_proba_original = tree_original.predict_proba(X_test_imputed)


# Изменённая модель: max_depth = 3
tree_changed = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

tree_changed.fit(X_train_imputed, y_train)

y_pred_changed = tree_changed.predict(X_test_imputed)
y_proba_changed = tree_changed.predict_proba(X_test_imputed)


def get_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(
            y_true, y_pred
        ),
        "Precision_macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall_macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "F1_macro": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "ROC_AUC_macro": roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro"
        )
    }


metrics_original = get_metrics(
    y_test,
    y_pred_original,
    y_proba_original
)

metrics_changed = get_metrics(
    y_test,
    y_pred_changed,
    y_proba_changed
)


comparison_params = pd.DataFrame([
    {
        "Model": "Дерево решений, max_depth=5",
        **metrics_original
    },
    {
        "Model": "Дерево решений, max_depth=3",
        **metrics_changed
    }
])


print("\nСравнение параметров дерева:")
print(comparison_params.to_string(index=False))

comparison_params.to_csv(
    "plots/tree_parameter_comparison.csv",
    index=False
)

print(
    "\nТаблица сохранена: "
    "plots\\tree_parameter_comparison.csv"
)

print("\n" + "=" * 70)
print("11. ИСКЛЮЧЕНИЕ МАЛОЗНАЧИМЫХ ПРИЗНАКОВ")
print("=" * 70)


# По результатам анализа важности признаков
insignificant_features = [
    "sex",
    "restecg",
    "slope"
]

print("\nИсключаемые признаки:")
print(insignificant_features)


# Формируем новый набор признаков
# Здесь используется features, а не feature_names
selected_features = [
    feature
    for feature in features
    if feature not in insignificant_features
]

print("\nПризнаки после удаления:")
print(selected_features)


# Новый массив данных только с выбранными признаками
X_reduced = X[selected_features]


# Разделяем данные на обучающую и тестовую выборки
# Используем тот же random_state и stratify,
# чтобы сравнение было корректным
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Заполняем пропуски тем же способом — медианой
imputer_reduced = SimpleImputer(strategy="median")

X_train_reduced_imputed = pd.DataFrame(
    imputer_reduced.fit_transform(X_train_reduced),
    columns=selected_features,
    index=X_train_reduced.index
)

X_test_reduced_imputed = pd.DataFrame(
    imputer_reduced.transform(X_test_reduced),
    columns=selected_features,
    index=X_test_reduced.index
)


# Обучаем дерево с теми же параметрами,
# что и исходное дерево
tree_reduced = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

tree_reduced.fit(
    X_train_reduced_imputed,
    y_train_reduced
)

y_pred_reduced = tree_reduced.predict(
    X_test_reduced_imputed
)

y_proba_reduced = tree_reduced.predict_proba(
    X_test_reduced_imputed
)


# Метрики сокращённой модели
metrics_reduced = get_metrics(
    y_test_reduced,
    y_pred_reduced,
    y_proba_reduced
)


# Метрики исходной модели
metrics_full = get_metrics(
    y_test,
    y_pred_original,
    y_proba_original
)


comparison_features = pd.DataFrame([
    {
        "Model": "Дерево: все признаки",
        "Количество признаков": len(features),
        **metrics_full
    },
    {
        "Model": "Дерево: без sex, restecg, slope",
        "Количество признаков": len(selected_features),
        **metrics_reduced
    }
])


print("\nСравнение качества классификации:")
print(
    comparison_features.to_string(index=False)
)


comparison_features.to_csv(
    "plots/tree_feature_removal_comparison.csv",
    index=False
)

print(
    "\nТаблица сохранена: "
    "plots\\tree_feature_removal_comparison.csv"
)
