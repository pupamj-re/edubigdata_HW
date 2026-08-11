```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# 0. 環境設定 (解決 Matplotlib 繁體中文顯示問題)
# ----------------------------------------------------
# 若在 Mac 上請使用 'PingFang HK'，Windows 上可使用 'Microsoft JhengHei'
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")

# ----------------------------------------------------
# 1. 讀取資料
# ----------------------------------------------------
file_stats = "H:\我的雲端硬碟\GitHub and VS\edubigdata_HW\高雄市114學校別_政府公開資料庫(移除新威).csv"
file_list = "H:\我的雲端硬碟\GitHub and VS\edubigdata_HW\高雄市114學名錄_政府公開資料庫.csv"

df_stats = pd.read_csv(file_stats)
df_list = pd.read_csv(file_list)

# ----------------------------------------------------
# 2. 資料清理與特徵工程 (Feature Engineering)
# ----------------------------------------------------
# 判斷公私立 (因部分附設國小不在學名錄中，直接從學校名稱判斷更準確)
df_stats['公私立'] = df_stats['學校名稱'].apply(
    lambda x: '私立' if '私立' in x else ('國立' if '國立' in x else '市立')
)

# 計算各校總班級數
class_cols = [f'{i}年級班級數' for i in range(1, 7)]
df_stats['總班級數'] = df_stats[class_cols].sum(axis=1)

# 計算各校男女學生與總學生數
df_stats['男學生總數'] = df_stats[[f'{i}年級男學生數' for i in range(1, 7)]].sum(axis=1)
df_stats['女學生總數'] = df_stats[[f'{i}年級女學生數' for i in range(1, 7)]].sum(axis=1)
df_stats['總學生數'] = df_stats['男學生總數'] + df_stats['女學生總數']

# 計算各校教師數與生師比
df_stats['總教師數'] = df_stats['男專任教師'] + df_stats['女專任教師']
# 避免除以 0 產生錯誤
df_stats['生師比'] = df_stats['總學生數'] / df_stats['總教師數'].replace(0, np.nan)

# ----------------------------------------------------
# 3. 關鍵數據輸出 (EDA Summary)
# ----------------------------------------------------
print("==== 高雄市 114 學年度國小基本數據 ====")
print(f"總學校數: {len(df_stats)}")
print(f"總學生數: {df_stats['總學生數'].sum():,} 人")
print(f"總班級數: {df_stats['總班級數'].sum():,} 班")
print(f"平均生師比: {df_stats['總學生數'].sum() / df_stats['總教師數'].sum():.2f}")

# ----------------------------------------------------
# 4. 資料視覺化 (Data Visualization)
# ----------------------------------------------------

# (A) 各行政區學生總數長條圖 (Top 10)
district_students = df_stats.groupby('鄉鎮市區')['總學生數'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=district_students.values, y=district_students.index, palette="viridis")
plt.title('高雄市國小學生數 Top 10 行政區', fontsize=16)
plt.xlabel('學生總數', fontsize=12)
plt.ylabel('行政區', fontsize=12)
plt.tight_layout()
plt.show()

# (B) 偏遠程度與學生人數分佈 (圓餅圖)
remote_students = df_stats.groupby('偏遠程度')['總學生數'].sum()
remote_labels = ['一般(0)', '偏遠(1)', '特偏(2)', '極偏(3)', '其他(4)'] # 依據資料字典替換

plt.figure(figsize=(8, 8))
plt.pie(remote_students, labels=[f"程度 {i}" for i in remote_students.index], 
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('高雄市國小各偏遠程度學生人數佔比', fontsize=16)
plt.show()

# (C) 人數最多前 5 名學校
top5_schools = df_stats.sort_values(by='總學生數', ascending=False).head(5)
print("\n==== 學生人數最多的前五所國小 ====")
print(top5_schools[['學校名稱', '鄉鎮市區', '總學生數', '總班級數', '生師比']])
