import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# 0. 環境設定 (解決 Matplotlib 繁體中文顯示問題)
# ----------------------------------------------------
# 根據你的作業系統選擇字體 (Windows: Microsoft JhengHei, Mac: PingFang HK)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")

# ----------------------------------------------------
# 1. 讀取資料
# ----------------------------------------------------
# 請確認檔案與程式碼在同一目錄下
df = pd.read_csv("H:\我的雲端硬碟\GitHub and VS\edubigdata_HW\高雄市114學校別_政府公開資料庫(移除新威).csv")

# ----------------------------------------------------
# 2. 特徵工程 (特徵計算與分層)
# ----------------------------------------------------
# 計算各校總班級數 (1~6年級)
class_cols = [f'{i}年級班級數' for i in range(1, 7)]
df['總班級數'] = df[class_cols].sum(axis=1)

# 每校班級數分層設定
bins_class = [0, 6, 12, 24, 48, np.inf]
labels_class = ['6班(含)以下', '7-12班', '13-24班', '25-48班', '49班以上']
df['班級數分層'] = pd.cut(df['總班級數'], bins=bins_class, labels=labels_class)
class_dist = df['班級數分層'].value_counts().sort_index()

# 計算各校男、女學生數及總學生數
df['男學生總數'] = df[[f'{i}年級男學生數' for i in range(1, 7)]].sum(axis=1)
df['女學生總數'] = df[[f'{i}年級女學生數' for i in range(1, 7)]].sum(axis=1)
df['總學生數'] = df['男學生總數'] + df['女學生總數']

# 學生人數分層設定
bins_student = [0, 50, 100, 300, 600, 1000, np.inf]
labels_student = ['50人以下', '51-100人', '101-300人', '301-600人', '601-1000人', '1000人以上']
df['學生人數分層'] = pd.cut(df['總學生數'], bins=bins_student, labels=labels_student)
student_dist = df['學生人數分層'].value_counts().sort_index()

# 性別總數
total_male = df['男學生總數'].sum()
total_female = df['女學生總數'].sum()

# ----------------------------------------------------
# 3. 終端機文字輸出 (Console Output)
# ----------------------------------------------------
print("==== 高雄市 114 學年度國小 EDA 摘要 ====")
print(f"1. 總校數: {len(df)} 所")
print("\n2. 每校班級數分層:")
print(class_dist.to_string())
print("\n3. 學生人數分層:")
print(student_dist.to_string())
print("\n4. 學生男女性別:")
print(f"   男學生: {total_male:,} 人")
print(f"   女學生: {total_female:,} 人")
print(f"   性別比(男/女): {total_male/total_female:.3f}")

# ----------------------------------------------------
# 4. 資料視覺化 (Data Visualization - 建立 2x2 子圖表)
# ----------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('高雄市 114 學年度國小探索性資料分析 (EDA)', fontsize=20, fontweight='bold', y=0.95)

# 圖 1: 校數分佈 (僅用單一長條表示總數)
sns.barplot(x=['總校數'], y=[len(df)], ax=axes[0, 0], palette=['#4C72B0'])
axes[0, 0].set_title('全市國小總校數', fontsize=14)
axes[0, 0].set_ylabel('學校數量', fontsize=12)
for i, v in enumerate([len(df)]):
    axes[0, 0].text(i, v + 2, str(v), ha='center', fontsize=14, fontweight='bold')

# 圖 2: 每校班級數分層
sns.barplot(x=class_dist.index, y=class_dist.values, ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('每校班級數分層', fontsize=14)
axes[0, 1].set_ylabel('學校數量', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=15)
for i, v in enumerate(class_dist.values):
    axes[0, 1].text(i, v + 1, str(v), ha='center', fontsize=12)

# 圖 3: 學生人數分層
sns.barplot(x=student_dist.index, y=student_dist.values, ax=axes[1, 0], palette='magma')
axes[1, 0].set_title('學生人數分層', fontsize=14)
axes[1, 0].set_ylabel('學校數量', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=15)
for i, v in enumerate(student_dist.values):
    axes[1, 0].text(i, v + 1, str(v), ha='center', fontsize=12)

# 圖 4: 學生性別分佈 (圓餅圖)
axes[1, 1].pie([total_male, total_female], labels=['男學生', '女學生'], 
               autopct='%1.1f%%', startangle=90, colors=['#55A868', '#C44E52'], 
               textprops={'fontsize': 14})
axes[1, 1].set_title(f'學童性別比例\n(總人數: {total_male+total_female:,}人)', fontsize=14)

# 調整佈局並顯示
plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # 留出主標題空間
plt.show()