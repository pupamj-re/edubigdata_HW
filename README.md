# 高雄市國小資料分析

此專案會讀取 `edubigdata_HW/` 中的 CSV，產生數個圖表並輸出到 `output/`。

## 圖表預覽

- 生師比箱型圖

![生師比箱型圖](output/option1_biobox.png)

- 學生人數 vs 班級規模 熱力圖

![交叉熱力圖](output/option2_heatmap.png)

- 學生與教師性別結構（100% 堆疊）

![性別結構](output/option3_gender.png)

- 行政區偏遠等級分佈

![行政區分佈](output/option4_distribution.png)

---

## 本機執行

建立虛擬環境並安裝需求：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

執行主程式產生圖檔：

```bash
python gemini-code-1786348741522.py
```

或用 notebook 互動執行（示範如何以檔案路徑匯入函式）：

打開 `gemini-code-1786348741522.ipynb` 並執行 cell。