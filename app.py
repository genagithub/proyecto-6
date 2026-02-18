import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Output, Input


df = pd.read_csv("data/stock_market.csv")
df["Date"] = pd.to_datetime(df["Date"])

df["RSI"] = ta.rsi(df["Close"], length=14)

macd = df.ta.macd(close="Close", fast=12, slow=26, signal=9)
df = pd.concat([df, macd], axis=1)

df["Close_yesterday"] = df["Close"].shift(1)
df_ml = df[["Close","Close_yesterday","RSI","MACD","Sentiment","Volume"]].dropna()

train_size = int(len(df_ml) * 0.75)
train, test = df_ml.iloc[:train_size], df_ml.iloc[train_size:]

X_train = train[["Close_yesterday","RSI","MACD","Sentiment","Volume"]]
y_train = train["Close"]
X_test = test[["Close_yesterday","RSI","MACD","Sentiment","Volume"]]
y_test = test["Close"]

# --- Modelo de validación ---
val_forecast_model = RandomForestRegressor(n_estimators=100, random_state=42)
val_forecast_model.fit(X_train, y_train)

test_predicts = val_forecast_model.predict(X_test)

MAE_raw = mean_absolute_error(y_test, test_predicts)
MAE_round = round(MAE_raw, 2)
MAE_ = "$"+str(MAE_round)

X = df_ml[["Close_yesterday", "RSI", "MACD", "Sentiment"]]
y = df_ml["Close"]

# --- Modelo de pronóstico a implementar ---
forecast_model = RandomForestRegressor(n_estimators=100, random_state=42)
forecast_model.fit(X, y)

max_month_mean = html.B(children=[], id="max")
min_month_mean = html.B(children=[], id="min")

dates = [
    {"label":"Primer Mes","value":30},
    {"label":"Primer Trimestre","value":90},
    {"label":"Primer Cuatrimestre","value":120},
    {"label":"Primer Semestre","value":180}
]

html_b = html.B("Acciones ($)", style={"color":"yellow"})

MAE = html.B(MAE_, id="MAE")
ROI = html.B(children=[], id="ROI")
STD = html.B(children=[], id="STD")

app = dash.Dash(__name__)

app.layout =  html.Div(id="body", className="e7_body", children=[
    html.H1("Análisis de Tendencia Mensual", id="H1", className="e7_title"),
    html.Div(id="KPI_div_1", className="e7_KPI_div_1", children=[
        html.P(max_month_mean, className="e7_KPI_1"),
        html.P(min_month_mean, className="e7_KPI_1")
    ]),
    html.Div(id="graph_div_1", className="e7_graph_div_1", children=[
       dcc.Dropdown(id="dropdown_1", className="e7_dropdown_1",
                        options=[
                            {"label":"Volumen","value":"Volume"},
                            {"label":"Precio ($)","value":"Close"}
                        ],
                        value="Volume",
                        multi=False,
                        clearable=False),
        dcc.Graph(id="trend_analysis", figure={}, className="e7_graph_1")
    ]),
    html.H2(["Pronóstico de ", html_b], id="H2", className="e7_title"),
    html.Div(id="forecast_div", className="e7_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e7_KPI_div_2", children=[
            html.Div(className="e7_KPI_2", children=[html.P("MAE", className="e7_KPI_title"), html.P(MAE, className="e7_KPI_p")]),
            html.Div(className="e7_KPI_2", children=[html.P("ROI", className="e7_KPI_title"), html.P(ROI, className="e7_KPI_p")]),
            html.Div(className="e7_KPI_2", children=[html.P("STD", className="e7_KPI_title"), html.P(STD, className="e7_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e7_graph_div_2", children=[
            html.Div(id="dropdown_div", className="e7_dropdown_div", children=[
                dcc.Dropdown(id="dropdown_2", className="e7_dropdown_2",
                        options=dates,
                        value=30,
                        multi=False,
                        clearable=False)]),
            dcc.Graph(id="forecasting", figure={}, className="e7_graph_2")    
        ])
    ])
])


@app.callback(
    [Output(component_id="max",component_property="children"),
    Output(component_id="min",component_property="children"),
    Output(component_id="trend_analysis",component_property="figure"),
    Output(component_id="forecasting",component_property="figure"),
    Output(component_id="ROI",component_property="children"),
    Output(component_id="STD",component_property="children")],
    [Input(component_id="dropdown_1",component_property="value"),
    Input(component_id="dropdown_2",component_property="value")]
)

def update_graph(slct_var, slct_days):
    
    df_seg = df.groupby(pd.Grouper(key="Date", freq="ME"))[slct_var].mean().reset_index()
    df_seg["month"] = df_seg["Date"].dt.strftime("%b")
    df_seg["month_year"] = df_seg["Date"].dt.strftime("%b %Y")

    month_mean = df_seg.groupby("month")[slct_var].mean().reset_index()

    max_mean = str(round(month_mean.max()[1], 1))
    max_month = month_mean.max()[0]
    max_month_mean = f"Máx: {max_month} ({max_mean}$)"

    min_mean = str(round(month_mean.min()[1], 1))
    min_month = month_mean.min()[0]
    min_month_mean = f"Mín: {min_month} ({min_mean}$)"
    
    months_means = go.Figure()
    months_means.add_trace(go.Scatter(x=df_seg["month_year"], y=df_seg[slct_var], mode="lines", fill="tozeroy", fillcolor="rgba(0, 255, 0, 0.2)"))
    months_means.update_layout(title_text="Promedios mensuales",  xaxis_title=" ", yaxis_title=" ")
    
    label_slct = next((option["label"] for option in dates if option["value"] == slct_days), slct_days)
    
    last_row = df_ml.iloc[-1]
    
    actual_close = last_row["Close"]
    actual_rsi = last_row["RSI"]
    actual_macd = last_row["MACD"]
    actual_sentiment = last_row["Sentiment"]
    actual_volume = last_row["Volume"]
    actual_date = pd.to_datetime(df["Date"]).max()
    
    predicts = []
    future_dates = []

    volatility = df["Close"].pct_change().std()
    mean_volume = df_ml["Volume"].mean()
    std_volume = df_ml["Volume"].std()
    min_volume = df_ml["Volume"].min()

    for _ in range(slct_days):
        input_data = [[actual_close, actual_rsi, actual_macd, actual_sentiment, actual_volume]]
        
        price_predict = forecast_model.predict(input_data)[0]
        variability = np.random.normal(0, volatility) 
        price_predict = price_predict * (1 + variability)
        
        daily_return = (price_predict - actual_close) / actual_close
        
        predicts.append(price_predict)
        actual_date += pd.Timedelta(days=1)
        future_dates.append(actual_date)
        
        actual_rsi = np.clip(actual_rsi + (daily_return * 100 * 0.1), 0, 100)
        actual_macd = actual_macd + (daily_return * actual_close * 0.05)
        actual_sentiment = np.clip(actual_sentiment + np.random.normal(0, 0.01), -1, 1)
        actual_volume = (actual_volume * 0.7) + (mean_volume * 0.3) + np.random.normal(0, std_volume * 0.05)
        actual_volume = max(min_volume, actual_volume)
        actual_close = price_predict
        
    df_forecast = pd.DataFrame({
        "Date":future_dates,
        "Forecast":predicts
    })
    
    last_two_years = datetime.now() - timedelta(days=2*365) 
    df_fil = df[df["Date"] >= last_two_years].copy()
    
    forecasting = go.Figure()
    forecasting.add_trace(go.Scatter(x=df_fil["Date"], y=df_fil["Close"], mode="lines", fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.2)", name="Período 2024-2025"))
    forecasting.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast"], name="Pronóstico 2026", fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.2)"))
    forecasting.update_layout(title_text=f"Pronóstico del {label_slct}",  xaxis_title=" ", yaxis_title=" ")
    
    future_prices = np.array(predicts).flatten()
    initial_price = df["Close"].iloc[-1]
    final_price = future_prices[-1]
    ROI = round(((final_price - initial_price) / initial_price) * 100, 2)
    if ROI > 0:
        ROI = "+"+str(ROI)+"%"
    else:
        ROI = str(ROI)+"%"

    STDs = np.diff(np.insert(future_prices, 0, initial_price)) / np.insert(future_prices, 0, initial_price)[:-1]
    STD = round(np.std(STDs) * 100, 2)
    STD = "$"+str(STD)

    return max_month_mean, min_month_mean, months_means, forecasting, ROI, STD

if __name__ == "__main__":
    max_month_mean = html.B(children=[], id="max")
min_month_mean = html.B(children=[], id="min")

dates = [
    {"label":"Primer Mes","value":30},
    {"label":"Primer Trimestre","value":90},
    {"label":"Primer Cuatrimestre","value":120},
    {"label":"Primer Semestre","value":180}
]

html_b = html.B("Acciones ($)", style={"color":"yellow"})

MAE = html.B(MAE_, id="MAE")
ROI = html.B(children=[], id="ROI")
STD = html.B(children=[], id="STD")

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body", className="e7_body", children=[
    html.H1("Análisis de Tendencia Mensual", id="H1", className="e7_title"),
    html.Div(id="KPI_div_1", className="e7_KPI_div_1", children=[
        html.P(max_month_mean, className="e7_KPI_1"),
        html.P(min_month_mean, className="e7_KPI_1")
    ]),
    html.Div(id="graph_div_1", className="e7_graph_div_1", children=[
       dcc.Dropdown(id="dropdown_1", className="e7_dropdown_1",
                        options=[
                            {"label":"Volumen","value":"Volume"},
                            {"label":"Precio ($)","value":"Close"}
                        ],
                        value="Volume",
                        multi=False,
                        clearable=False),
        dcc.Graph(id="trend_analysis", figure={}, className="e7_graph_1")
    ]),
    html.H2(["Pronóstico de ", html_b], id="H2", className="e7_title"),
    html.Div(id="forecast_div", className="e7_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e7_KPI_div_2", children=[
            html.Div(className="e7_KPI_2", children=[html.P("MAE", className="e7_KPI_title"), html.P(MAE, className="e7_KPI_p")]),
            html.Div(className="e7_KPI_2", children=[html.P("ROI", className="e7_KPI_title"), html.P(ROI, className="e7_KPI_p")]),
            html.Div(className="e7_KPI_2", children=[html.P("STD", className="e7_KPI_title"), html.P(STD, className="e7_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e7_graph_div_2", children=[
            html.Div(id="dropdown_div", className="e7_dropdown_div", children=[
                dcc.Dropdown(id="dropdown_2", className="e7_dropdown_2",
                        options=dates,
                        value=30,
                        multi=False,
                        clearable=False)]),
            dcc.Graph(id="forecasting", figure={}, className="e7_graph_2")    
        ])
    ])
])


@app.callback(
    [Output(component_id="max",component_property="children"),
    Output(component_id="min",component_property="children"),
    Output(component_id="trend_analysis",component_property="figure"),
    Output(component_id="forecasting",component_property="figure"),
    Output(component_id="ROI",component_property="children"),
    Output(component_id="STD",component_property="children")],
    [Input(component_id="dropdown_1",component_property="value"),
    Input(component_id="dropdown_2",component_property="value")]
)

def update_graph(slct_var, slct_days):
    
    df_seg = df.groupby(pd.Grouper(key="Date", freq="ME"))[slct_var].mean().reset_index()
    df_seg["month"] = df_seg["Date"].dt.strftime("%b")
    df_seg["month_year"] = df_seg["Date"].dt.strftime("%b %Y")

    month_mean = df_seg.groupby("month")[slct_var].mean().reset_index()

    max_mean = str(round(month_mean.max()[1], 1))
    max_month = month_mean.max()[0]
    max_month_mean = f"Máx: {max_month} ({max_mean}$)"

    min_mean = str(round(month_mean.min()[1], 1))
    min_month = month_mean.min()[0]
    min_month_mean = f"Mín: {min_month} ({min_mean}$)"
    
    months_means = go.Figure()
    months_means.add_trace(go.Scatter(x=df_seg["month_year"], y=df_seg[slct_var], mode="lines", fill="tozeroy", fillcolor="rgba(0, 255, 0, 0.2)"))
    months_means.update_layout(title_text="Promedios mensuales",  xaxis_title=" ", yaxis_title=" ")
    
    label_slct = next((option["label"] for option in dates if option["value"] == slct_days), slct_days)
    
    last_row = df_ml.iloc[-1]
    
    actual_close = last_row["Close"]
    actual_rsi = last_row["RSI"]
    actual_macd = last_row["MACD"]
    actual_sentiment = last_row["Sentiment"]
    actual_volume = last_row["Volume"]
    actual_date = pd.to_datetime(df["Date"]).max()
    
    predicts = []
    future_dates = []

    volatility = df["Close"].pct_change().std()
    mean_volume = df_ml["Volume"].mean()
    std_volume = df_ml["Volume"].std()
    min_volume = df_ml["Volume"].min()

    for _ in range(slct_days):
        input_data = [[actual_close, actual_rsi, actual_macd, actual_sentiment, actual_volume]]
        
        price_predict = forecast_model.predict(input_data)[0]
        variability = np.random.normal(0, volatility) 
        price_predict = price_predict * (1 + variability)
        
        daily_return = (price_predict - actual_close) / actual_close
        
        predicts.append(price_predict)
        actual_date += pd.Timedelta(days=1)
        future_dates.append(actual_date)
        
        actual_rsi = np.clip(actual_rsi + (daily_return * 100 * 0.1), 0, 100)
        actual_macd = actual_macd + (daily_return * actual_close * 0.05)
        actual_sentiment = np.clip(actual_sentiment + np.random.normal(0, 0.01), -1, 1)
        actual_volume = (actual_volume * 0.7) + (mean_volume * 0.3) + np.random.normal(0, std_volume * 0.05)
        actual_volume = max(min_volume, actual_volume)
        actual_close = price_predict
        
    df_forecast = pd.DataFrame({
        "Date":future_dates,
        "Forecast":predicts
    })
    
    last_two_years = datetime.now() - timedelta(days=2*365) 
    df_fil = df[df["Date"] >= last_two_years].copy()
    
    forecasting = go.Figure()
    forecasting.add_trace(go.Scatter(x=df_fil["Date"], y=df_fil["Close"], mode="lines", fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.2)", name="Período 2024-2025"))
    forecasting.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast"], name="Pronóstico 2026", fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.2)"))
    forecasting.update_layout(title_text=f"Pronóstico del {label_slct}",  xaxis_title=" ", yaxis_title=" ")
    
    future_prices = np.array(predicts).flatten()
    initial_price = df["Close"].iloc[-1]
    final_price = future_prices[-1]
    ROI = round(((final_price - initial_price) / initial_price) * 100, 2)
    if ROI > 0:
        ROI = "+"+str(ROI)+"%"
    else:
        ROI = str(ROI)+"%"

    STDs = np.diff(np.insert(future_prices, 0, initial_price)) / np.insert(future_prices, 0, initial_price)[:-1]
    STD = round(np.std(STDs) * 100, 2)
    STD = "$"+str(STD)

    return max_month_mean, min_month_mean, months_means, forecasting, ROI, STD

if __name__ == "__main__":
    app.run_server(debug=False)max_month_mean = html.B(children=[], id="max")
min_month_mean = html.B(children=[], id="min")

dates = [
    {"label":"Primer Mes","value":30},
    {"label":"Primer Trimestre","value":90},
    {"label":"Primer Cuatrimestre","value":120},
    {"label":"Primer Semestre","value":180}
]

html_b = html.B("Acciones ($)", style={"color":"yellow"})

MAE = html.B(MAE_, id="MAE")
ROI = html.B(children=[], id="ROI")
STD = html.B(children=[], id="STD")

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body", className="e7_body", children=[
    html.H1("Análisis de Tendencia Mensual", id="H1", className="e7_title"),
    html.Div(id="KPI_div_1", className="e7_KPI_div_1", children=[
        html.P(max_month_mean, className="e7_KPI_1"),
        html.P(min_month_mean, className="e7_KPI_1")
    ]),
    html.Div(id="graph_div_1", className="e7_graph_div_1", children=[
       dcc.Dropdown(id="dropdown_1", className="e7_dropdown_1",
                        options=[
                            {"label":"Volumen","value":"Volume"},
                            {"label":"Precio ($)","value":"Close"}
                        ],
                        value="Volume",
                        multi=False,
                        clearable=False),
        dcc.Graph(id="trend_analysis", figure={}, className="e7_graph_1")
    ]),
    html.H2(["Pronóstico de ", html_b], id="H2", className="e7_title"),
    html.Div(id="forecast_div", className="e7_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e7_KPI_div_2", children=[
            html.Div(className="e7_KPI_2", children=[html.P("MAE", className="e7_KPI_title"), html.P(MAE, className="e7_KPI_p")]),
            html.Div(className="e7_KPI_2", children=[html.P("ROI", className="e7_KPI_title"), html.P(ROI, className="e7_KPI_p")]),
            html.Div(className="e7_KPI_2", children=[html.P("STD", className="e7_KPI_title"), html.P(STD, className="e7_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e7_graph_div_2", children=[
            html.Div(id="dropdown_div", className="e7_dropdown_div", children=[
                dcc.Dropdown(id="dropdown_2", className="e7_dropdown_2",
                        options=dates,
                        value=30,
                        multi=False,
                        clearable=False)]),
            dcc.Graph(id="forecasting", figure={}, className="e7_graph_2")    
        ])
    ])
])


@app.callback(
    [Output(component_id="max",component_property="children"),
    Output(component_id="min",component_property="children"),
    Output(component_id="trend_analysis",component_property="figure"),
    Output(component_id="forecasting",component_property="figure"),
    Output(component_id="ROI",component_property="children"),
    Output(component_id="STD",component_property="children")],
    [Input(component_id="dropdown_1",component_property="value"),
    Input(component_id="dropdown_2",component_property="value")]
)

def update_graph(slct_var, slct_days):
    
    df_seg = df.groupby(pd.Grouper(key="Date", freq="ME"))[slct_var].mean().reset_index()
    df_seg["month"] = df_seg["Date"].dt.strftime("%b")
    df_seg["month_year"] = df_seg["Date"].dt.strftime("%b %Y")

    month_mean = df_seg.groupby("month")[slct_var].mean().reset_index()

    max_mean = str(round(month_mean.max()[1], 1))
    max_month = month_mean.max()[0]
    max_month_mean = f"Máx: {max_month} ({max_mean}$)"

    min_mean = str(round(month_mean.min()[1], 1))
    min_month = month_mean.min()[0]
    min_month_mean = f"Mín: {min_month} ({min_mean}$)"
    
    months_means = go.Figure()
    months_means.add_trace(go.Scatter(x=df_seg["month_year"], y=df_seg[slct_var], mode="lines", fill="tozeroy", fillcolor="rgba(0, 255, 0, 0.2)"))
    months_means.update_layout(title_text="Promedios mensuales",  xaxis_title=" ", yaxis_title=" ")
    
    label_slct = next((option["label"] for option in dates if option["value"] == slct_days), slct_days)
    
    last_row = df_ml.iloc[-1]
    
    actual_close = last_row["Close"]
    actual_rsi = last_row["RSI"]
    actual_macd = last_row["MACD"]
    actual_sentiment = last_row["Sentiment"]
    actual_volume = last_row["Volume"]
    actual_date = pd.to_datetime(df["Date"]).max()
    
    predicts = []
    future_dates = []

    volatility = df["Close"].pct_change().std()
    mean_volume = df_ml["Volume"].mean()
    std_volume = df_ml["Volume"].std()
    min_volume = df_ml["Volume"].min()

    for _ in range(slct_days):
        input_data = [[actual_close, actual_rsi, actual_macd, actual_sentiment, actual_volume]]
        
        price_predict = forecast_model.predict(input_data)[0]
        variability = np.random.normal(0, volatility) 
        price_predict = price_predict * (1 + variability)
        
        daily_return = (price_predict - actual_close) / actual_close
        
        predicts.append(price_predict)
        actual_date += pd.Timedelta(days=1)
        future_dates.append(actual_date)
        
        actual_rsi = np.clip(actual_rsi + (daily_return * 100 * 0.1), 0, 100)
        actual_macd = actual_macd + (daily_return * actual_close * 0.05)
        actual_sentiment = np.clip(actual_sentiment + np.random.normal(0, 0.01), -1, 1)
        actual_volume = (actual_volume * 0.7) + (mean_volume * 0.3) + np.random.normal(0, std_volume * 0.05)
        actual_volume = max(min_volume, actual_volume)
        actual_close = price_predict
        
    df_forecast = pd.DataFrame({
        "Date":future_dates,
        "Forecast":predicts
    })
    
    last_two_years = datetime.now() - timedelta(days=2*365) 
    df_fil = df[df["Date"] >= last_two_years].copy()
    
    forecasting = go.Figure()
    forecasting.add_trace(go.Scatter(x=df_fil["Date"], y=df_fil["Close"], mode="lines", fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.2)", name="Período 2024-2025"))
    forecasting.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast"], name="Pronóstico 2026", fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.2)"))
    forecasting.update_layout(title_text=f"Pronóstico del {label_slct}",  xaxis_title=" ", yaxis_title=" ")
    
    future_prices = np.array(predicts).flatten()
    initial_price = df["Close"].iloc[-1]
    final_price = future_prices[-1]
    ROI = round(((final_price - initial_price) / initial_price) * 100, 2)
    if ROI > 0:
        ROI = "+"+str(ROI)+"%"
    else:
        ROI = str(ROI)+"%"

    STDs = np.diff(np.insert(future_prices, 0, initial_price)) / np.insert(future_prices, 0, initial_price)[:-1]
    STD = round(np.std(STDs) * 100, 2)
    STD = "$"+str(STD)

    return max_month_mean, min_month_mean, months_means, forecasting, ROI, STD

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
