import dash
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import os
import dash_ag_grid as dag  # ✅ 使用 AG Grid 替代 dash_table
from pathlib import Path 

# === 設定 ===
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = str(SCRIPT_DIR / "data.csv")
ASSETS_DIR = str(Path("assets") / "raw_A") # 圖片資料夾
IMG_COL_INDEX = 0
IMG_WIDTH = 128
PAGE_SIZE = 100 # 每頁顯示的資料筆數

# === 資料載入 ===
df = pd.read_csv(CSV_PATH)
img_col = df.columns[IMG_COL_INDEX]
df_display = df.copy()

# 轉圖片為 markdown 顯示語法
def make_img_markdown(rel_path):
    filename = os.path.basename(rel_path)
    return f"![img]({ASSETS_DIR}/{filename})"

df_display[img_col] = df_display[img_col].apply(make_img_markdown)

# === App ===
app = Dash(__name__)
app.title = "互動式圖片資料表"

# 建立 AG Grid 欄位設定
column_defs = []
for col in df_display.columns:
    col_def = {
        "headerName": col,
        "field": col,
        "filter": "agSetColumnFilter",  # ✅ 使用下拉式選單過濾器
        "sortable": True,
        "resizable": True,
    }
    if col == img_col:
        col_def["cellRenderer"] = "markdown"  # ✅ 顯示 HTML 圖片
        col_def["autoHeight"] = True
    column_defs.append(col_def)

# === Layout ===
app.layout = html.Div([
    html.H2("📷 圖片資料總表"),

    html.Div([
        html.Label("選擇繪圖欄位："),
        dcc.Dropdown(
            id="plot-column",
            options=[{"label": col, "value": col} for col in df.columns[1:]],
            placeholder="請選擇欄位繪圖"
        )
    ], style={"width": "300px", "marginBottom": "20px"}),

    dcc.Graph(id="column-plot"),

    dag.AgGrid(
        id="main-table",
        #columnSize="sizeToFit",
        columnDefs=column_defs,
        rowData=df_display.to_dict("records"),
        defaultColDef={
            "filter": True,
            "editable": True,
            "flex": 1,
            "minWidth": 120,
            #"maxWidth": 'auto',
            "wrapText": True,
            "autoHeight": True,
            "cellStyle": {"textAlign": "center"},
        },
        dashGridOptions={
            "pagination": True,
            "paginationPageSize": PAGE_SIZE,
            "domLayout": "autoHeight",
        },
        style={"height": "600px", "width": "100%"},
    )
])

# === Callback: 動態繪圖 ===
@app.callback(
    Output("column-plot", "figure"),
    Input("plot-column", "value"),
    State("main-table", "rowData")
)
def update_graph(col, table_data):
    if not col or not table_data:
        return px.scatter()
    dff = pd.DataFrame(table_data)
    if pd.api.types.is_numeric_dtype(dff[col]):
        return px.histogram(dff, x=col, nbins=20, title=f"{col} 分布圖")
    else:
        return px.bar(dff[col].value_counts().reset_index(), x="index", y=col, title=f"{col} 類別統計")

if __name__ == "__main__":
    app.run(debug=False)
