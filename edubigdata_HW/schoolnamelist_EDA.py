import re
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------
# 0. 設定與常數
# ----------------------------------------------------
FONT_FAMILY: Final[list[str]] = ["Microsoft JhengHei", "Arial Unicode MS", "SimHei"]
DATA_DIR: Final[Path] = Path(__file__).resolve().parent
OUTPUT_DIR: Final[Path] = DATA_DIR.parent / "output"
STATS_FILE: Final[Path] = DATA_DIR / "高雄市114學校別_政府公開資料庫(移除新威).csv"
LIST_FILE: Final[Path] = DATA_DIR / "高雄市114學名錄_政府公開資料庫.csv"

CLASS_BINS: Final[list[float]] = [0.0, 6.0, 12.0, 24.0, 48.0, float("inf")]
CLASS_LABELS: Final[list[str]] = ["6班(含)以下", "7-12班", "13-24班", "25-48班", "49班以上"]
STUDENT_BINS: Final[list[float]] = [0.0, 50.0, 100.0, 300.0, 600.0, 1000.0, float("inf")]
STUDENT_LABELS: Final[list[str]] = ["50人以下", "51-100人", "101-300人", "301-600人", "601-1000人", "1000人以上"]

plt.rcParams["font.sans-serif"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")


def normalize_name(name: str) -> str:
    """清除學校名稱中的多餘空白，避免合併時因格式差異失敗。"""
    return re.sub(r"\s+", "", str(name).strip())


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀入兩份原始 CSV 檔並回傳整理後的 DataFrame。"""
    df_stats = pd.read_csv(STATS_FILE, encoding="utf-8-sig")
    df_list = pd.read_csv(LIST_FILE, encoding="utf-8-sig")

    for df in (df_stats, df_list):
        df.columns = [col.strip() for col in df.columns]

    return df_stats, df_list


def prepare_analysis_frame(df_stats: pd.DataFrame, df_list: pd.DataFrame) -> pd.DataFrame:
    """整合兩份資料並建立分析所需的欄位。"""
    df_stats = df_stats.copy()
    df_list = df_list.copy()

    df_stats["學校名稱_norm"] = df_stats["學校名稱"].apply(normalize_name)
    df_list["學校名稱_norm"] = df_list["學校名稱"].apply(normalize_name)

    df_stats["學校代碼"] = df_stats["學校代碼"].astype(str).str.strip()
    df_list = df_list.rename(columns={"代碼": "學校代碼"})
    df_list["學校代碼"] = df_list["學校代碼"].astype(str).str.strip()

    merged = df_stats.merge(df_list[["學校代碼", "公/私立", "地址", "網址"]], on="學校代碼", how="left")

    name_map = (
        df_list[["學校名稱_norm", "公/私立", "地址", "網址"]]
        .drop_duplicates("學校名稱_norm")
        .set_index("學校名稱_norm")
    )

    merged["公/私立"] = merged["公/私立"].fillna(merged["學校名稱_norm"].map(name_map["公/私立"]))
    merged["地址"] = merged["地址"].fillna(merged["學校名稱_norm"].map(name_map["地址"]))
    merged["網址"] = merged["網址"].fillna(merged["學校名稱_norm"].map(name_map["網址"]))
    merged["公/私立"] = merged["公/私立"].fillna("未標註")

    class_cols = [f"{i}年級班級數" for i in range(1, 7)]
    male_cols = [f"{i}年級男學生數" for i in range(1, 7)]
    female_cols = [f"{i}年級女學生數" for i in range(1, 7)]

    merged["總班級數"] = merged[class_cols].sum(axis=1)
    merged["男學生總數"] = merged[male_cols].sum(axis=1)
    merged["女學生總數"] = merged[female_cols].sum(axis=1)
    merged["總學生數"] = merged["男學生總數"] + merged["女學生總數"]
    merged["總教師數"] = merged["男專任教師"] + merged["女專任教師"]
    merged["生師比"] = merged["總學生數"] / merged["總教師數"].replace(0, np.nan)
    merged["偏遠程度"] = merged["偏遠程度"].fillna("一般")

    return merged


def build_summary_components(merged: pd.DataFrame) -> tuple[list[str], pd.Series, pd.Series, int, int]:
    """建立文字摘要與分層統計結果。"""
    merged["班級數分層"] = pd.cut(
        merged["總班級數"], bins=CLASS_BINS, labels=CLASS_LABELS, include_lowest=True
    )
    class_dist = merged["班級數分層"].value_counts().sort_index()

    merged["學生人數分層"] = pd.cut(
        merged["總學生數"], bins=STUDENT_BINS, labels=STUDENT_LABELS, include_lowest=True
    )
    student_dist = merged["學生人數分層"].value_counts().sort_index()

    district_summary = (
        merged.groupby("鄉鎮市區")
        .agg(學校數=("學校名稱", "count"), 學生人數=("總學生數", "sum"), 班級數=("總班級數", "sum"))
        .sort_values(["學生人數", "學校數"], ascending=False)
    )
    size_rank = merged[["學校名稱", "鄉鎮市區", "總學生數", "總班級數", "公/私立", "偏遠程度"]].sort_values("總學生數", ascending=False)
    remote_summary = (
        merged.groupby("偏遠程度")
        .agg(學校數=("學校名稱", "count"), 學生人數=("總學生數", "sum"), 平均學生數=("總學生數", "mean"))
        .reset_index()
        .sort_values("學生人數", ascending=False)
    )

    male_total = int(merged["男學生總數"].sum())
    female_total = int(merged["女學生總數"].sum())

    summary_lines: list[str] = []
    summary_lines.append("==== 高雄市 114 學年度國小 EDA 摘要 ====")
    summary_lines.append(f"總學校數: {len(merged)} 所")
    summary_lines.append(f"總學生數: {merged['總學生數'].sum():,} 人")
    summary_lines.append(f"總班級數: {merged['總班級數'].sum():,} 班")
    summary_lines.append(f"總教師數: {merged['總教師數'].sum():,} 人")
    summary_lines.append(f"平均生師比: {merged['總學生數'].sum() / merged['總教師數'].sum():.2f}")
    summary_lines.append(f"公立學校數: {merged[merged['公/私立'].str.contains('公立', na=False)].shape[0]} 所")
    summary_lines.append(f"私立學校數: {merged[merged['公/私立'].str.contains('私立', na=False)].shape[0]} 所")
    summary_lines.append("")
    summary_lines.append("1. 行政區分布（依學生人數排序）")
    summary_lines.append(district_summary.head(10).to_string())
    summary_lines.append("")
    summary_lines.append("2. 學校規模樣態（學生人數最多/最少前五）")
    summary_lines.append("前五多:")
    summary_lines.append(size_rank.head(5).to_string(index=False))
    summary_lines.append("")
    summary_lines.append("前五少:")
    summary_lines.append(size_rank.tail(5).to_string(index=False))
    summary_lines.append("")
    summary_lines.append("3. 偏遠程度分布")
    summary_lines.append(remote_summary.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("4. 每校班級數分層")
    summary_lines.append(class_dist.to_string())
    summary_lines.append("")
    summary_lines.append("5. 學生人數分層")
    summary_lines.append(student_dist.to_string())
    summary_lines.append("")
    summary_lines.append("6. 學生男女性別")
    summary_lines.append(f"男學生總數: {male_total:,} 人")
    summary_lines.append(f"女學生總數: {female_total:,} 人")
    summary_lines.append(f"性別比(男/女): {male_total / female_total:.3f}")

    return summary_lines, class_dist, student_dist, male_total, female_total


def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    """將圖表存成 PNG 檔並關閉圖表物件。"""
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_charts(
    merged: pd.DataFrame,
    class_dist: pd.Series,
    student_dist: pd.Series,
    male_total: int,
    female_total: int,
    output_dir: Path,
) -> None:
    """產生並儲存所有 EDA 圖表。"""
    district_students = merged.groupby("鄉鎮市區")["總學生數"].sum().sort_values(ascending=False).head(10)
    district_df = pd.DataFrame({"行政區": district_students.index, "學生人數": district_students.values})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=district_df,
        x="學生人數",
        y="行政區",
        hue="行政區",
        palette="viridis",
        ax=ax,
        legend=False,
        order=district_df["行政區"],
    )
    ax.set_title("高雄市 114 學年度國小行政區學生數 Top 10")
    ax.set_xlabel("學生總數")
    ax.set_ylabel("行政區")
    save_figure(fig, "高雄市114國小_行政區學生數.png", output_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    remote_counts = merged["偏遠程度"].value_counts().sort_index()
    remote_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, ax=ax, colors=sns.color_palette("pastel"))
    ax.set_title("高雄市 114 學年度國小偏遠程度分布")
    ax.set_ylabel("")
    save_figure(fig, "高雄市114國小_偏遠程度分布.png", output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    class_df = pd.DataFrame({"分    python educode.pyst.index, "學校數": class_dist.values})
    sns.barplot(data=class_df, x="分層", y="學校數", hue="分層", palette="viridis", ax=ax, legend=False)
    ax.set_title("高雄市 114 學年度國小每校班級數分層")
    ax.set_xlabel("班級數分層")
    ax.set_ylabel("學校數")
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, "高雄市114國小_班級數分層.png", output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    student_df = pd.DataFrame({"分層": student_dist.index, "學校數": student_dist.values})
    sns.barplot(data=student_df, x="分層", y="學校數", hue="分層", palette="magma", ax=ax, legend=False)
    ax.set_title("高雄市 114 學年度國小學生人數分層")
    ax.set_xlabel("學生人數分層")
    ax.set_ylabel("學校數")
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, "高雄市114國小_學生人數分層.png", output_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie([male_total, female_total], labels=["男學生", "女學生"], autopct="%1.1f%%", startangle=90, colors=["#55A868", "#C44E52"])
    ax.set_title("高雄市 114 學年度國小學生性別比例")
    save_figure(fig, "高雄市114國小_學生性別比例.png", output_dir)


def main() -> None:
    """執行完整 EDA 流程，輸出摘要檔與圖表。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_stats, df_list = load_raw_data()
    merged = prepare_analysis_frame(df_stats, df_list)

    summary_lines, class_dist, student_dist, male_total, female_total = build_summary_components(merged)
    print("\n".join(summary_lines))

    with open(OUTPUT_DIR / "高雄市114國小_EDA摘要.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    plot_charts(merged, class_dist, student_dist, male_total, female_total, OUTPUT_DIR)
    print(f"\n分析已完成，結果與圖表已輸出至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()