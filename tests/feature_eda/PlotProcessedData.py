import pandas as pd
import dash
from dash import html, dcc, Input, Output
import dash_ag_grid as dag
import plotly.graph_objs as go
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# === 讀取資料 ===
MAIN_DIR = Path('~/TBrain_SmartPing')
#CSV_PATH = MAIN_DIR / 'tests' / 'feature_eda' / 'TimeSeriesFeatures' / 'Results6' / 'train_all.csv'
CSV_PATH = MAIN_DIR / 'data' / 'processed' / 'TimeSeriesFeaturesV5' / 'TrainingData' / 'train_all.csv'
df = pd.read_csv(str(CSV_PATH))
categorical_targets = ["player_id", "mode", "gender", "hold_racket_handed", "play_years", "level"]
ordered_targets = ["play_years", "level"]
feature_cols = [col for col in df.columns if col not in ["unique_id"] + categorical_targets]

# === 編碼類別型變數 ===
le_dict = {}
df_encoded = df.copy()
for col in categorical_targets:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# === 計算每個預測目標與 feature 的 MI 分數 ===
importance_df = []
for target in categorical_targets:
    X = df_encoded[feature_cols]
    y = df_encoded[target]
    scores = mutual_info_classif(X, y, discrete_features=False)
    for f, s in zip(feature_cols, scores):
        importance_df.append({"target": target, "feature": f, "MI_score": s})
importance_df = pd.DataFrame(importance_df)
importance_df = importance_df.sort_values(by=["target", "MI_score"], ascending=[True, False])

# === Dash App ===
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("特徵重要性分析報告"),
    dag.AgGrid(
        id="importance-table",
        columnDefs=[
            {"headerName": "目標變數", "field": "target", "checkboxSelection": True},
            {"headerName": "特徵欄位", "field": "feature"},
            {"headerName": "MI 分數", "field": "MI_score", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}}
        ],
        rowData=importance_df.to_dict("records"),
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        columnSize="sizeToFit",
        style={"height": "400px", "width": "100%"},
    ),
    html.Hr(),

    html.Div([
        html.Div([
            html.Label("選擇預測目標變數"),
            dcc.Dropdown(options=[{"label": col, "value": col} for col in categorical_targets], value="gender", id="target-dropdown")
        ], style={"width": "30%", "display": "inline-block"}),

        html.Div([
            html.Label("選擇特徵欄位"),
            dcc.Dropdown(options=[{"label": col, "value": col} for col in feature_cols], value=feature_cols[0], id="feature-dropdown")
        ], style={"width": "30%", "display": "inline-block", "marginLeft": "20px"}),
    ]),

    dcc.Graph(id="prob-plot")
])

@app.callback(
    Output("prob-plot", "figure"),
    Input("target-dropdown", "value"),
    Input("feature-dropdown", "value")
)
def update_plot(target_col, feature_col):
    grouped = df[[target_col, feature_col]].groupby(target_col)
    traces = []
    for name, group in grouped:
        trace = go.Histogram(
            x=group[feature_col],
            histnorm="probability",
            name=str(name),
            opacity=0.6
        )
        traces.append(trace)
    layout = go.Layout(
        title=f"{feature_col} 的各類別 {target_col} 機率分布圖",
        xaxis_title=feature_col,
        yaxis_title="Probability",
        barmode="overlay"
    )
    return {"data": traces, "layout": layout}

if __name__ == "__main__":
    app.run(debug=True)
