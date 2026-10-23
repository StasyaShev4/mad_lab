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

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

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

# ============================================================
# 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 70)
print("7. КЛАССИФИКАЦИЯ")
print("=" * 70)
print(f"\nОбучающая выборка: {len(X_train)} объектов")
print(f"Тестовая выборка: {len(X_test)} объектов")

# ============================================================
# 3. ОБРАБОТКА ПРОПУСКОВ
# ============================================================

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

# ============================================================
# 4. СТАНДАРТИЗАЦИЯ ДЛЯ SVM
# ============================================================

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

# ============================================================
# 5. ДЕРЕВО РЕШЕНИЙ
# ============================================================

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

# ============================================================
# 6. SVM
# ============================================================

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

# ============================================================
# 7. ПРЕДСКАЗАНИЯ
# ============================================================

print("\nДерево решений:")
print("Предсказания:", y_pred_tree)

print("\nSVM:")
print("Предсказания:", y_pred_svm)

print(f"\nКоличество предсказаний дерева: {len(y_pred_tree)}")
print(f"Количество предсказаний SVM: {len(y_pred_svm)}")

# ============================================================
# 8. ФУНКЦИЯ ОЦЕНКИ МОДЕЛИ
# ============================================================

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

# ============================================================
# 9. ОЦЕНКА ДВУХ МОДЕЛЕЙ
# ============================================================

tree_results = evaluate_model(
    "Дерево решений", y_test, y_pred_tree, tree_scores
)

svm_results = evaluate_model(
    "SVM", y_test, y_pred_svm, svm_scores
)

# ============================================================
# 10. СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК
# ============================================================

results = pd.DataFrame([tree_results, svm_results])

print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
print("=" * 70)
print(results.to_string(index=False))

# ============================================================
# 11. ДЕТАЛЬНЫЙ ОТЧЁТ ПО КАЖДОМУ КЛАССУ
# ============================================================

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

# ============================================================
# 12. МАТРИЦЫ ОШИБОК
# ============================================================

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

# ============================================================
# 13. ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ ДЕРЕВА
# ============================================================

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": tree_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n" + "=" * 70)
print("ВАЖНОСТЬ ПРИЗНАКОВ — ДЕРЕВО РЕШЕНИЙ")
print("=" * 70)
print(feature_importance.to_string(index=False))

feature_importance_path = os.path.join(
    plots_dir, "feature_importance_tree.csv"
)
feature_importance.to_csv(feature_importance_path, index=False)
print(f"\nТаблица важности признаков сохранена: {feature_importance_path}")

