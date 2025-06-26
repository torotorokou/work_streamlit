import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate_stage1(stage1_eval, target_items):
    print("\n===== ステージ1評価結果 =====")
    for item in target_items:
        y_true = np.array(stage1_eval[item]["y_true"])
        y_pred = np.array(stage1_eval[item]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{item}: R² = {r2:.3f}, MAE = {mae:,.0f}kg")


def plot_feature_importances(names, importances, title="Feature Importance", top_k=15):
    sorted_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), np.array(importances)[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), np.array(names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
