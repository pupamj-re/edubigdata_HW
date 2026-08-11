import re
import urllib.request
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# ----------------------------------------------------
# 0. 設定與常數
# ----------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "edubigdata_HW"
OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
STATS_FILE: Final[Path] = DATA_DIR / "高雄市114學校別_政府公開資料庫(移除新威).csv"
LIST_FILE: Final[Path] = DATA_DIR / "高雄市114學名錄_政府公開資料庫.csv"
FONT_URL: Final[str] = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_FILE: Final[Path] = BASE_DIR / "NotoSansCJKtc-Regular.otf"

CLASS_BINS: Final[list[float]] = [0.0, 6.0, 12.0, 24.0, 48.0, float("inf")]
CLASS_LABELS: Final[list[str]] = ["6班(含)以下", "7-12班", "13-24班", "25-48班", "49班以上"]
STUDENT_BINS: Final[list[float]] = [0.0, 50.0, 100.0, 300.0, 600.0, 1000.0, float("inf")]
STUDENT_LABELS: Final[list[str]] = ["50人以下", "51-100人", "101-300人", "301-600人", "601-1000人", "1000人以上"]
REMOTE_ORDER: Final[list[str]] = ["一般", "非山非市", "偏遠", "特偏", "極偏"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "微軟正黑體", "Taipei Sans TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


# ----------------------------------------------------
# 1. 基礎工具
# ----------------------------------------------------
def normalize_name(name: str) -> str:
    """清除學校名稱中的多餘空白，避免合併時因格式差異失敗。"""
    return re.sub(r"\s+", "", str(name).strip())


def ensure_font() -> str | None:
    """嘗試載入中文字型，若不存在則下載。"""
    if FONT_FILE.exists():
        return str(FONT_FILE)
    try:
        print("正在下載中文字型...")
        urllib.request.urlretrieve(FONT_URL, FONT_FILE)
        return str(FONT_FILE)
    except Exception:
        return None


def set_chinese_font() -> None:
    """設定 matplotlib 中文字型。"""
    font_file = ensure_font()
    font_candidates = ["Noto Sans CJK TC", "Microsoft JhengHei", "微軟正黑體", "Taipei Sans TC", "sans-serif"]
    if font_file and Path(font_file).exists():
        fm.fontManager.addfont(font_file)
        font_name = fm.FontProperties(fname=font_file).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name] + font_candidates
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = font_candidates
    plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------
# 2. 資料讀取與整理
# ----------------------------------------------------
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


# ----------------------------------------------------
# 3. 分層與摘要
# ----------------------------------------------------
def add_stratification_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """加入各種分層欄位。"""
    df = merged.copy()
    df["班級數分層"] = pd.cut(df["總班級數"], bins=CLASS_BINS, labels=CLASS_LABELS, include_lowest=True)
    df["學生人數分層"] = pd.cut(df["總學生數"], bins=STUDENT_BINS, labels=STUDENT_LABELS, include_lowest=True)
    return df


def build_summary_components(merged: pd.DataFrame) -> tuple[list[str], pd.Series, pd.Series, int, int, int, int]:
    """建立文字摘要與分層統計結果。"""
    merged = add_stratification_columns(merged)

    class_dist = merged["班級數分層"].value_counts().sort_index()
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
    male_teacher_total = int(merged["男專任教師"].sum())
    female_teacher_total = int(merged["女專任教師"].sum())

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
    summary_lines.append(f"男教師總數: {male_teacher_total:,} 人")
    summary_lines.append(f"女教師總數: {female_teacher_total:,} 人")

    return summary_lines, class_dist, student_dist, male_total, female_total, male_teacher_total, female_teacher_total


# ----------------------------------------------------
# 4. 圖表輸出
# ----------------------------------------------------
def compute_statistical_summary(merged: pd.DataFrame) -> dict[str, object]:
    """計算正式報告所需的統計量。"""
    district_summary = (
        merged.groupby("鄉鎮市區")
        .agg(學校數=("學校名稱", "count"), 學生人數=("總學生數", "sum"), 平均學生數=("總學生數", "mean"), 平均班級數=("總班級數", "mean"))
        .sort_values("學生人數", ascending=False)
    )
    remote_summary = (
        merged.groupby("偏遠程度")
        .agg(學校數=("學校名稱", "count"), 平均學生數=("總學生數", "mean"), 平均生師比=("生師比", "mean"))
        .reset_index()
    )
    corr_students_classes = float(merged["總學生數"].corr(merged["總班級數"]))
    top_school = merged.loc[merged["總學生數"].idxmax()]
    bottom_school = merged.loc[merged["總學生數"].idxmin()]

    return {
        "district_summary": district_summary,
        "remote_summary": remote_summary,
        "corr_students_classes": corr_students_classes,
        "top_school": top_school,
        "bottom_school": bottom_school,
    }


def analyze_regional_and_regression(merged: pd.DataFrame) -> dict[str, object]:
    """進一步分析區域比較與偏遠程度對生師比的回歸關係。"""
    regional_summary = (
        merged.groupby("鄉鎮市區")
        .agg(
            學校數=("學校名稱", "count"),
            總學生數=("總學生數", "sum"),
            平均學生數=("總學生數", "mean"),
            平均班級數=("總班級數", "mean"),
            平均生師比=("生師比", "mean"),
        )
        .sort_values(["總學生數", "平均生師比"], ascending=[False, True])
    )

    remoteness_levels = REMOTE_ORDER
    code_map = {level: idx for idx, level in enumerate(remoteness_levels)}
    regression_df = merged.copy()
    regression_df["偏遠程度編碼"] = regression_df["偏遠程度"].map(code_map)

    x = regression_df["偏遠程度編碼"].astype(float).to_numpy()
    y = regression_df["生師比"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(regression_df)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else np.nan

    return {
        "regional_summary": regional_summary,
        "regression_df": regression_df,
        "intercept": intercept,
        "slope": slope,
        "r_squared": r_squared,
        "remoteness_levels": remoteness_levels,
    }


def analyze_multivariable_regression(merged: pd.DataFrame) -> dict[str, object]:
    """使用多變量線性回歸分析偏遠程度、行政區與學校規模對生師比的共同影響。"""
    model_df = merged[["生師比", "偏遠程度", "鄉鎮市區", "總學生數", "總班級數"]].copy()
    model_df = model_df.dropna(subset=["生師比", "偏遠程度", "鄉鎮市區", "總學生數", "總班級數"])

    remoteness_map = {level: idx for idx, level in enumerate(REMOTE_ORDER)}
    model_df["偏遠程度編碼"] = model_df["偏遠程度"].map(remoteness_map)
    model_df["學校規模_log"] = np.log1p(model_df["總學生數"])
    district_dummies = pd.get_dummies(model_df["鄉鎮市區"], prefix="區域", drop_first=True)

    x_df = pd.concat(
        [
            pd.DataFrame({"截距": 1.0}, index=model_df.index),
            model_df[["偏遠程度編碼", "學校規模_log"]],
            district_dummies,
        ],
        axis=1,
    ).astype(float)
    y = model_df["生師比"].astype(float).to_numpy(dtype=float)
    X = x_df.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else np.nan

    coeff_df = pd.DataFrame({"變數": x_df.columns, "係數": coef})
    coeff_df = coeff_df[coeff_df["變數"] != "截距"].sort_values("係數", key=lambda s: s.abs(), ascending=False)

    return {
        "model_df": model_df,
        "x_df": x_df,
        "coef": coef,
        "coeff_df": coeff_df,
        "y_pred": y_pred,
        "r_squared": r_squared,
    }


def build_formal_report(
    merged: pd.DataFrame,
    male_total: int,
    female_total: int,
    male_teacher_total: int,
    female_teacher_total: int,
    analysis_result: dict[str, object],
    multivariable_result: dict[str, object],
) -> list[str]:
    """建立更像正式報告的文字摘要。"""
    stats = compute_statistical_summary(merged)
    district_summary = stats["district_summary"]
    remote_summary = stats["remote_summary"]
    corr_students_classes = stats["corr_students_classes"]
    top_school = stats["top_school"]
    bottom_school = stats["bottom_school"]
    regional_summary = analysis_result["regional_summary"]
    intercept = analysis_result["intercept"]
    slope = analysis_result["slope"]
    r_squared = analysis_result["r_squared"]
    multivariable_r_squared = multivariable_result["r_squared"]
    coeff_df = multivariable_result["coeff_df"]

    report_lines: list[str] = []
    report_lines.append("# 高雄市 114 學年度國小教育資源與學校規模正式報告")
    report_lines.append("")
    report_lines.append("## 1. 報告目的")
    report_lines.append("本報告依據高雄市 114 學年度國小學校資料，整合學校規模、班級規模、偏遠程度與性別結構等面向，檢視各區域與不同學校類型的教育資源分布現況。")
    report_lines.append("")
    report_lines.append("## 2. 資料概況")
    report_lines.append(f"- 分析學校數：{len(merged)} 所")
    report_lines.append(f"- 總學生數：{int(merged['總學生數'].sum()):,} 人")
    report_lines.append(f"- 總班級數：{int(merged['總班級數'].sum()):,} 班")
    report_lines.append(f"- 總教師數：{int(merged['總教師數'].sum()):,} 人")
    report_lines.append(f"- 平均生師比：{merged['總學生數'].sum() / merged['總教師數'].sum():.2f}")
    report_lines.append(f"- 學生性別分布：男生 {male_total:,} 人，女生 {female_total:,} 人")
    report_lines.append(f"- 教師性別分布：男教師 {male_teacher_total:,} 人，女教師 {female_teacher_total:,} 人")
    report_lines.append("")
    report_lines.append("## 3. 主要發現")
    report_lines.append(f"- 行政區層面，{district_summary.index[0]} 的學生總數最高，達 {int(district_summary.iloc[0]['學生人數']):,} 人，顯示其教育資源集中度較高。")
    report_lines.append(f"- 學校規模方面，{top_school['學校名稱']} 為全市規模最大者，總學生數為 {int(top_school['總學生數']):,} 人；{bottom_school['學校名稱']} 則為最小規模者，總學生數僅 {int(bottom_school['總學生數'])} 人。")
    report_lines.append(f"- 總學生數與總班級數之相關係數為 {corr_students_classes:.3f}，顯示學校規模與班級配置呈現高度同步。")
    report_lines.append("- 偏遠程度分布顯示，偏遠與極偏學校的平均學生規模明顯低於一般學校，反映出資源配置與學校規模之間的差異。")
    report_lines.append("")
    report_lines.append("## 4. 統計觀察")
    report_lines.append("以下為各偏遠程度的平均規模與平均生師比：")
    report_lines.append(remote_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("## 5. 區域比較")
    report_lines.append("以下列出各行政區的學校數、總學生數與平均生師比：")
    report_lines.append(regional_summary.head(10).to_string())
    report_lines.append("")
    report_lines.append("## 6. 偏遠程度與生師比回歸分析")
    report_lines.append(f"- 使用簡單線性回歸模型，將偏遠程度依序編碼為一般、非山非市、偏遠、特偏、極偏。")
    report_lines.append(f"- 回歸方程式可表示為：生師比 ≈ {intercept:.3f} + {slope:.3f} × 偏遠程度編碼")
    report_lines.append(f"- 決定係數 R² 約為 {r_squared:.3f}，顯示偏遠程度對生師比具有中等解釋能力。")
    report_lines.append("- 此結果提示，偏遠程度與生師比之間存在明顯相關，但仍建議納入教師配置、學校型態與區域人口結構等其他變數做進一步檢驗。")
    report_lines.append("")
    report_lines.append("## 7. 多變量分析：偏遠程度、區域與學校規模共同影響生師比")
    report_lines.append(f"- 以偏遠程度編碼、行政區虛擬變數與學校規模（取對數）進行多變量線性回歸，R² 約為 {multivariable_r_squared:.3f}。")
    report_lines.append("- 這表示偏遠程度、區域與學校規模三者共同能解釋生師比的部分變異。")
    report_lines.append("- 於模型中，影響較大的變數如下：")
    report_lines.append(coeff_df.head(10).to_string(index=False))
    report_lines.append("")
    report_lines.append("## 8. 管理與政策啟示")
    report_lines.append("1. 建議持續關注偏遠與極偏學校的規模與師資配置，以維持基本教育供給品質。")
    report_lines.append("2. 大型學校可透過班級與行政資源規劃，避免因規模過大造成班級與教師配置不均。")
    report_lines.append("3. 未來可結合區域發展、人口變遷與交通條件，並納入更完整的多變量分析，進一步評估學校整併與資源再配置的可行性。")
    return report_lines


def create_interactive_charts(merged: pd.DataFrame, output_dir: Path) -> None:
    """產生互動式 HTML 圖表。"""
    district_summary = (
        merged.groupby("鄉鎮市區")
        .agg(學校數=("學校名稱", "count"), 學生人數=("總學生數", "sum"), 平均學生數=("總學生數", "mean"))
        .reset_index()
        .sort_values("學生人數", ascending=False)
    )
    fig1 = px.bar(
        district_summary,
        x="鄉鎮市區",
        y="學生人數",
        color="學生人數",
        title="高雄市各行政區學生數分布（互動圖）",
        text="學生人數",
    )
    fig1.update_traces(texttemplate="%{text}", textposition="outside")
    fig1.write_html(str(output_dir / "interactive_district_students.html"))

    scatter_df = merged[["學校名稱", "鄉鎮市區", "總學生數", "總班級數", "偏遠程度"]].copy()
    fig2 = px.scatter(
        scatter_df,
        x="總班級數",
        y="總學生數",
        color="偏遠程度",
        size="總學生數",
        hover_name="學校名稱",
        title="學校規模與班級規模關係（互動散點圖）",
    )
    fig2.write_html(str(output_dir / "interactive_school_size_vs_class.html"))

    remote_summary = (
        merged.groupby("偏遠程度")
        .agg(學校數=("學校名稱", "count"), 平均學生數=("總學生數", "mean"), 平均生師比=("生師比", "mean"))
        .reset_index()
    )
    fig3 = px.bar(
        remote_summary,
        x="偏遠程度",
        y="平均學生數",
        color="平均學生數",
        title="各偏遠程度平均學生規模（互動圖）",
        text="平均學生數",
    )
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig3.write_html(str(output_dir / "interactive_remote_summary.html"))


def plot_regional_and_regression_analysis(merged: pd.DataFrame, analysis_result: dict[str, object], output_dir: Path) -> None:
    """輸出區域比較與回歸分析圖。"""
    regional_summary = analysis_result["regional_summary"]
    regression_df = analysis_result["regression_df"]
    intercept = analysis_result["intercept"]
    slope = analysis_result["slope"]
    remoteness_levels = analysis_result["remoteness_levels"]

    region_top = regional_summary.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=region_top.reset_index(), x="總學生數", y="鄉鎮市區", order=region_top.index.tolist(), ax=ax, color="#4C72B0")
    ax.set_title("各行政區總學生數比較")
    ax.set_xlabel("總學生數")
    ax.set_ylabel("行政區")
    save_figure(fig, "區域比較_總學生數.png", output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    code_map = {level: idx for idx, level in enumerate(remoteness_levels)}
    regression_df["偏遠程度編碼"] = regression_df["偏遠程度"].map(code_map)
    sns.scatterplot(data=regression_df, x="偏遠程度編碼", y="生師比", ax=ax, color="#4C72B0")
    x_vals = np.array([0, 1, 2, 3, 4], dtype=float)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color="#C44E52", linewidth=2)
    ax.set_xticks(range(len(remoteness_levels)))
    ax.set_xticklabels(remoteness_levels)
    ax.set_title("偏遠程度與生師比回歸分析")
    ax.set_xlabel("偏遠程度")
    ax.set_ylabel("生師比")
    save_figure(fig, "偏遠程度_vs_生師比回歸.png", output_dir)

    fig = px.bar(
        regional_summary.reset_index(),
        x="鄉鎮市區",
        y="平均生師比",
        color="平均生師比",
        title="各行政區平均生師比（互動圖）",
        text="平均生師比",
    )
    fig.write_html(str(output_dir / "interactive_region_bi_ratio.html"))


def plot_multivariable_analysis(multivariable_result: dict[str, object], output_dir: Path) -> None:
    """輸出多變量分析相關圖表。"""
    coeff_df = multivariable_result["coeff_df"].head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=coeff_df, x="係數", y="變數", ax=ax, color="#4C72B0")
    ax.set_title("多變量模型係數大小比較")
    ax.set_xlabel("係數")
    ax.set_ylabel("變數")
    save_figure(fig, "多變量分析_係數.png", output_dir)

    y_true = multivariable_result["model_df"]["生師比"].to_numpy()
    y_pred = multivariable_result["y_pred"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=y_true, ax=ax, color="#4C72B0")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="#C44E52", linestyle="--")
    ax.set_title("多變量模型實際值 vs 預測值")
    ax.set_xlabel("預測生師比")
    ax.set_ylabel("實際生師比")
    save_figure(fig, "多變量分析_實際_vs_預測.png", output_dir)


def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    """將圖表存成 PNG 檔並關閉圖表物件。"""
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_charts(
    merged: pd.DataFrame,
    class_dist: pd.Series,
    student_dist: pd.Series,
    male_total: int,
    female_total: int,
    output_dir: Path,
) -> None:
    """產生摘要型圖表。"""
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
    class_df = pd.DataFrame({"分層": class_dist.index, "學校數": class_dist.values})
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


def plot_investigation_charts(merged: pd.DataFrame, output_dir: Path) -> None:
    """產生延伸分析圖表。"""
    remoteness_order = REMOTE_ORDER

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=merged, x="偏遠程度", y="生師比", order=remoteness_order, ax=ax, color="#8dd3c7")
    ax.set_title("高雄市各偏遠程度國小之生師比分佈")
    ax.set_ylabel("生師比 (總學生/總專任教師)")
    ax.set_xlabel("偏遠程度")
    save_figure(fig, "option1_biobox.png", output_dir)

    cross_tab = pd.crosstab(merged["學生人數分層"], merged["班級數分層"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"size": 14}, ax=ax)
    ax.set_title("高雄市國小學生人數與班級規模交叉分析")
    ax.set_ylabel("學生人數分層")
    ax.set_xlabel("班級規模分層")
    save_figure(fig, "option2_heatmap.png", output_dir)

    gender_agg = merged.groupby("偏遠程度")[["男學生總數", "女學生總數", "男專任教師", "女專任教師"]].sum().loc[remoteness_order]
    gender_agg["學生男%"] = gender_agg["男學生總數"] / (gender_agg["男學生總數"] + gender_agg["女學生總數"]) * 100
    gender_agg["學生女%"] = gender_agg["女學生總數"] / (gender_agg["男學生總數"] + gender_agg["女學生總數"]) * 100
    gender_agg["教師男%"] = gender_agg["男專任教師"] / (gender_agg["男專任教師"] + gender_agg["女專任教師"]) * 100
    gender_agg["教師女%"] = gender_agg["女專任教師"] / (gender_agg["男專任教師"] + gender_agg["女專任教師"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(gender_agg.index, gender_agg["學生男%"], label="男學生", color="#4C72B0")
    axes[0].bar(gender_agg.index, gender_agg["學生女%"], bottom=gender_agg["學生男%"], label="女學生", color="#C44E52")
    axes[0].set_title("各偏遠程度學生性別結構 (100%堆疊)")
    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.7)
    axes[0].legend(loc="lower right")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("百分比 (%)")

    axes[1].bar(gender_agg.index, gender_agg["教師男%"], label="男教師", color="#55A868")
    axes[1].bar(gender_agg.index, gender_agg["教師女%"], bottom=gender_agg["教師男%"], label="女教師", color="#DD8452")
    axes[1].set_title("各偏遠程度專任教師性別結構 (100%堆疊)")
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.7)
    axes[1].legend(loc="lower right")
    axes[1].set_ylim(0, 100)
    save_figure(fig, "option3_gender.png", output_dir)

    dist_rem = pd.crosstab(merged["鄉鎮市區"], merged["偏遠程度"])
    for col in remoteness_order:
        if col not in dist_rem.columns:
            dist_rem[col] = 0
    dist_rem = dist_rem[remoteness_order]
    dist_rem["Total"] = dist_rem.sum(axis=1)
    dist_rem = dist_rem.sort_values("Total", ascending=False).drop("Total", axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))
    dist_rem.plot(kind="bar", stacked=True, cmap="viridis", ax=ax)
    ax.set_title("高雄市各行政區學校偏遠等級分佈")
    ax.set_xlabel("行政區")
    ax.set_ylabel("學校數量")
    ax.legend(title="偏遠程度")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "option4_distribution.png", output_dir)


# ----------------------------------------------------
# 5. 主流程
# ----------------------------------------------------
def main() -> None:
    """執行完整 EDA 流程，輸出摘要檔與圖表。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_chinese_font()

    df_stats, df_list = load_raw_data()
    merged = prepare_analysis_frame(df_stats, df_list)
    merged = add_stratification_columns(merged)

    summary_lines, class_dist, student_dist, male_total, female_total, male_teacher_total, female_teacher_total = build_summary_components(merged)
    print("\n".join(summary_lines))

    with open(OUTPUT_DIR / "高雄市114國小_EDA摘要.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    analysis_result = analyze_regional_and_regression(merged)
    multivariable_result = analyze_multivariable_regression(merged)
    formal_report_lines = build_formal_report(
        merged,
        male_total,
        female_total,
        male_teacher_total,
        female_teacher_total,
        analysis_result,
        multivariable_result,
    )
    with open(OUTPUT_DIR / "高雄市114國小_正式報告.md", "w", encoding="utf-8") as f:
        f.write("\n".join(formal_report_lines))

    with open(OUTPUT_DIR / "高雄市114國小_區域比較.csv", "w", encoding="utf-8-sig", newline="") as f:
        analysis_result["regional_summary"].to_csv(f)

    with open(OUTPUT_DIR / "高雄市114國小_回歸分析摘要.txt", "w", encoding="utf-8") as f:
        f.write(
            f"偏遠程度與生師比回歸分析\n"
            f"截距: {analysis_result['intercept']:.3f}\n"
            f"斜率: {analysis_result['slope']:.3f}\n"
            f"R²: {analysis_result['r_squared']:.3f}\n\n"
            f"多變量分析（偏遠程度 + 行政區 + 學校規模）\n"
            f"R²: {multivariable_result['r_squared']:.3f}\n"
            f"{multivariable_result['coeff_df'].to_string(index=False)}\n"
        )

    plot_summary_charts(merged, class_dist, student_dist, male_total, female_total, OUTPUT_DIR)
    plot_investigation_charts(merged, OUTPUT_DIR)
    plot_regional_and_regression_analysis(merged, analysis_result, OUTPUT_DIR)
    plot_multivariable_analysis(multivariable_result, OUTPUT_DIR)
    create_interactive_charts(merged, OUTPUT_DIR)
    print(f"\n分析已完成，結果與圖表已輸出至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
