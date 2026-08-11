import os
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "edubigdata_HW", "高雄市114學校別_政府公開資料庫(移除新威).csv")
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_FILE = os.path.join(BASE_DIR, "NotoSansCJKtc-Regular.otf")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_font():
    if os.path.exists(FONT_FILE):
        return FONT_FILE
    try:
        print("正在下載中文字型...")
        urllib.request.urlretrieve(FONT_URL, FONT_FILE)
        return FONT_FILE
    except Exception:
        return None


def set_chinese_font():
    font_file = ensure_font()
    font_candidates = ["Noto Sans CJK TC", "Microsoft JhengHei", "微軟正黑體", "Taipei Sans TC", "sans-serif"]
    if font_file and os.path.exists(font_file):
        fm.fontManager.addfont(font_file)
    plt.rcParams["font.family"] = font_candidates
    plt.rcParams["font.sans-serif"] = font_candidates
    plt.rcParams["axes.unicode_minus"] = False


def save_plot(fig, filename, output_dir=OUTPUT_DIR):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # 總數計算
    df = df.copy()
    df["總學生數"] = df[[
        "1年級男學生數", "1年級女學生數", "2年級男學生數", "2年級女學生數",
        "3年級男學生數", "3年級女學生數", "4年級男學生數", "4年級女學生數",
        "5年級男學生數", "5年級女學生數", "6年級男學生數", "6年級女學生數"
    ]].sum(axis=1)
    df["總專任教師數"] = df.get("男專任教師", 0) + df.get("女專任教師", 0)
    df["總班級數"] = df[["1年級班級數", "2年級班級數", "3年級班級數", "4年級班級數", "5年級班級數", "6年級班級數"]].sum(axis=1)
    df["生師比"] = df["總學生數"] / df["總專任教師數"].replace(0, pd.NA)
    df["總男學生"] = df[["1年級男學生數", "2年級男學生數", "3年級男學生數", "4年級男學生數", "5年級男學生數", "6年級男學生數"]].sum(axis=1)
    df["總女學生"] = df[["1年級女學生數", "2年級女學生數", "3年級女學生數", "4年級女學生數", "5年級女學生數", "6年級女學生數"]].sum(axis=1)

    def get_student_layer(n):
        if n <= 40:
            return "1_40人以下"
        elif n <= 100:
            return "2_41-100人"
        elif n <= 300:
            return "3_101-300人"
        elif n <= 500:
            return "4_301-500人"
        elif n <= 1000:
            return "5_501-1000人"
        else:
            return "6_1000人以上"

    def get_class_layer(n):
        if n <= 6:
            return "1_6班及以下"
        elif n <= 12:
            return "2_7-12班"
        elif n <= 24:
            return "3_13-24班"
        elif n <= 48:
            return "4_25-48班"
        else:
            return "5_49班以上"

    df["學生人數分層"] = df["總學生數"].apply(get_student_layer)
    df["班級規模分層"] = df["總班級數"].apply(get_class_layer)

    return df


def make_plots(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    remoteness_order = ["一般", "非山非市", "偏遠", "特偏", "極偏"]

    # 選項一：偏鄉資源配置對比 (生師比箱型圖)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="偏遠程度", y="生師比", order=remoteness_order, palette="Set2", ax=ax)
    ax.set_title("高雄市各偏遠程度國小之生師比分佈", fontsize=16)
    ax.set_ylabel("生師比 (總學生/總專任教師)")
    ax.set_xlabel("偏遠程度")
    save_plot(fig, "option1_biobox.png", output_dir=output_dir)

    # 選項二：學校規模 M 型化 (交叉熱力圖)
    cross_tab = pd.crosstab(df["學生人數分層"], df["班級規模分層"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"size": 14}, ax=ax)
    ax.set_title("高雄市國小學生人數與班級規模交叉分析", fontsize=16)
    ax.set_ylabel("學生人數分層", fontsize=12)
    ax.set_xlabel("班級規模分層", fontsize=12)
    save_plot(fig, "option2_heatmap.png", output_dir=output_dir)

    # 選項三：性別結構不對稱現象 (100% 堆疊長條圖)
    gender_agg = df.groupby("偏遠程度")[["總男學生", "總女學生", "男專任教師", "女專任教師"]].sum().loc[remoteness_order]
    gender_agg["學生男%"] = gender_agg["總男學生"] / (gender_agg["總男學生"] + gender_agg["總女學生"]) * 100
    gender_agg["學生女%"] = gender_agg["總女學生"] / (gender_agg["總男學生"] + gender_agg["總女學生"]) * 100
    gender_agg["教師男%"] = gender_agg["男專任教師"] / (gender_agg["男專任教師"] + gender_agg["女專任教師"]) * 100
    gender_agg["教師女%"] = gender_agg["女專任教師"] / (gender_agg["男專任教師"] + gender_agg["女專任教師"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(gender_agg.index, gender_agg["學生男%"], label="男學生", color="#4C72B0")
    axes[0].bar(gender_agg.index, gender_agg["學生女%"], bottom=gender_agg["學生男%"], label="女學生", color="#C44E52")
    axes[0].set_title("各偏遠程度學生性別結構 (100%堆疊)", fontsize=14)
    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.7)
    axes[0].legend(loc="lower right")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("百分比 (%)")

    axes[1].bar(gender_agg.index, gender_agg["教師男%"], label="男教師", color="#55A868")
    axes[1].bar(gender_agg.index, gender_agg["教師女%"], bottom=gender_agg["教師男%"], label="女教師", color="#DD8452")
    axes[1].set_title("各偏遠程度專任教師性別結構 (100%堆疊)", fontsize=14)
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.7)
    axes[1].legend(loc="lower right")
    axes[1].set_ylim(0, 100)
    save_plot(fig, "option3_gender.png", output_dir=output_dir)

    # 選項四：各行政區偏遠等級分佈 (行政區堆疊長條圖)
    dist_rem = pd.crosstab(df["鄉鎮市區"], df["偏遠程度"])
    for col in remoteness_order:
        if col not in dist_rem.columns:
            dist_rem[col] = 0
    dist_rem = dist_rem[remoteness_order]
    dist_rem["Total"] = dist_rem.sum(axis=1)
    dist_rem = dist_rem.sort_values("Total", ascending=False).drop("Total", axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))
    dist_rem.plot(kind="bar", stacked=True, cmap="viridis", ax=ax)
    ax.set_title("高雄市各行政區學校偏遠等級分佈", fontsize=16)
    ax.set_xlabel("行政區", fontsize=12)
    ax.set_ylabel("學校數量", fontsize=12)
    ax.legend(title="偏遠程度")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "option4_distribution.png", output_dir=output_dir)


def load_data(data_file: str = DATA_FILE) -> pd.DataFrame:
    print("讀取資料檔：", data_file)
    return pd.read_csv(data_file, encoding="utf-8-sig")


def main(data_file: str = DATA_FILE, output_dir: str = OUTPUT_DIR):
    set_chinese_font()
    sns.set_theme(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_file)
    df = compute_metrics(df)
    make_plots(df, output_dir=output_dir)
    print("圖表已輸出到：", output_dir)


if __name__ == "__main__":
    main()