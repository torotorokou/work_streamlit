import pandas as pd
import matplotlib.pyplot as plt
from logic.factory_manage.make_df import read_csv_hannnyuu

# --- データ読み込み ---
df1 = read_csv_hannnyuu()
df1["年"] = df1["伝票日付"].dt.year

# --- 年・品名ごとの集計 ---
summary = (
    df1.groupby(["年", "品名"])
    .agg(
        件数=("正味重量", "count"),
        合計重量=("正味重量", "sum"),
        平均重量=("正味重量", "mean"),
        最大重量=("正味重量", "max"),
        最小重量=("正味重量", "min"),
    )
    .reset_index()
)

# --- 年別集計 ---
件数_by_year = df1.groupby("年")["正味重量"].count()
重量_by_year = df1.groupby("年")["正味重量"].sum()
平均重量_by_year = df1.groupby("年")["正味重量"].mean()

# --- 品目別集計 ---
品目ランキング = df1["品名"].value_counts().head(10)
平均重量_by_item = (
    df1.groupby("品名")["正味重量"].mean().sort_values(ascending=False).head(10)
)


# --- グラフ描画関数 ---
def show_bar_chart(title, series, ylabel):
    plt.figure(figsize=(10, 4))
    series.plot(kind="bar", color="skyblue", edgecolor="blue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- 表示 ---
print("✅ 年別 件数:")
print(件数_by_year)
print("\n✅ 年別 合計重量:")
print(重量_by_year)
print("\n✅ 年別 平均重量:")
print(平均重量_by_year)
print("\n✅ 品目別 件数ランキング:")
print(品目ランキング)
print("\n✅ 品目別 平均重量（上位10件）:")
print(平均重量_by_item)

# --- グラフ表示 ---
show_bar_chart("Yearly Number of Entries", 件数_by_year, "Count")
show_bar_chart("Yearly Total Weight (kg)", 重量_by_year, "Weight")
show_bar_chart("Yearly Average Weight (kg)", 平均重量_by_year, "Average Weight")
show_bar_chart("Top 10 Frequent Items", 品目ランキング, "Count")
show_bar_chart("Top 10 Items by Avg Weight", 平均重量_by_item, "Avg Weight")
