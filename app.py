import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Output, Input


df = pd.read_csv("data/marketing_campaign.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Acquisition_Cost"] = df["Acquisition_Cost"].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"])
df["Conversions"] = round(df["Clicks"] * (df["Conversion_Rate"] / 100))

all_categorical_vars = ["Campaign_Type", "Company", "Channel_Used", "Location"]

df_global_ts = df.groupby(["Date"] + all_categorical_vars).agg({
    "Conversions": "sum", 
    "Acquisition_Cost": "sum", 
    "Clicks": "sum",
    "Impressions": "sum", 
    "ROI": "mean", 
    "Conversion_Rate": "mean"
}).reset_index().sort_values("Date")

df_global_ts["CTR"] = (df_global_ts["Clicks"] / df_global_ts["Impressions"]).fillna(0).replace([np.inf, -np.inf], 0)
df_global_ts["CPC"] = (df_global_ts["Acquisition_Cost"] / df_global_ts["Clicks"]).fillna(0).replace([np.inf, -np.inf], 0)
df_global_ts["CPM"] = ((df_global_ts["Acquisition_Cost"] / df_global_ts["Impressions"]) * 1000).fillna(0).replace([np.inf, -np.inf], 0)

df_global_ts["Conv_Lag1"] = df_global_ts["Conversions"].shift(1)
df_global_ts["ROI_Lag1"] = df_global_ts["ROI"].shift(1)
df_global_ts["Clicks_Lag1"] = df_global_ts["Clicks"].shift(1)
df_global_ts["day_of_week"] = df_global_ts["Date"].dt.dayofweek

df_global_model = df_global_ts.dropna()

numerical_features = ["Conv_Lag1", "ROI_Lag1", "Clicks_Lag1", "day_of_week", "CTR", "CPC", "CPM"]

X_global_raw = df_global_model[numerical_features + all_categorical_vars]
X_global = pd.get_dummies(X_global_raw, columns=all_categorical_vars)

final_features = list(X_global.columns)

rf_conv = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42)
rf_roi = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42)
rf_cvr = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42)

rf_conv.fit(X_global, df_global_model["Conversions"])
rf_roi.fit(X_global, df_global_model["ROI"])
rf_cvr.fit(X_global, df_global_model["Conversion_Rate"])

factor_top_b = html.B(children=[], id="factor")
value_top_b = html.B(children=[], id="value")

ROI = html.B(children=[], id="ROI")
CVR = html.B(children=[], id="CVR")
CPC = html.B(children=[], id="CPC")

vars = [
    {"label":"Tipo de campaña","value":"Campaign_Type"},
    {"label":"Compañía","value":"Company"},
    {"label":"Canal usado","value":"Channel_Used"},
    {"label":"Locación","value":"Location"}
]

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body", className="e6_body", children=[
    html.H1("Análisis por dimensiones personalizadas", id="H1", className="e6_title"),
    html.Div(id="dropdown_div_1", className="e6_dropdown_div_1", children=[
        dcc.Dropdown(id="dropdown_vars", className="e6_dropdown_1",
                    options=vars,
                    value="Campaign_Type",
                    multi=False,
                    clearable=False)
    ]),
    html.Div(id="graph_div_1", className="e6_graph_div_1", children=[
        html.Div(id="KPI_div_1", className="e6_KPI_div_1", children=[
            html.P(factor_top_b, className="e6_KPI_1", style={"margin-right":"25px","width":"310px"}),
            html.P(value_top_b, className="e6_KPI_1", style={"margin-left":"25px","width":"310px"})
        ]),
        dcc.Graph(id="conversions_analysis", figure={}, className="e6_graph_1", style={"width":"74%"})
    ]),
    html.A(href="https://github.com/genagithub/proyecto-6/blob/main/README.md", children=[html.H2("Proyección de conversiones (modelos promocionales)", id="H2", className="e6_title")]),
    html.Div(id="forecast_div", className="e6_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e6_KPI_div_2", children=[
            html.Div(className="e6_KPI_2", children=[html.P("ROI", className="e6_KPI_title"), html.P([ROI,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CVR", className="e6_KPI_title"), html.P([CVR,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CPC", className="e6_KPI_title"), html.P(["$",CPC], className="e6_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e6_graph_div_2", children=[
        html.Div(id="dropdown_div_2", className="e6_dropdown_div_2", children=[
            dcc.Dropdown(id="dropdown_var1", className="e6_dropdown_2",
                        options=df["Campaign_Type"].unique(),
                        value=df["Campaign_Type"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_var2", className="e6_dropdown_2",
                        options=df["Company"].unique(),
                        value=df["Company"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_var3", className="e6_dropdown_2",
                        options=df["Channel_Used"].unique(),
                        value=df["Channel_Used"].unique()[0],
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_var4", className="e6_dropdown_2",
                        options=df["Location"].unique(),
                        value=df["Location"].unique()[0],
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="forecasting", figure={}, className="e6_graph_2") 
        ])   
    ])   
])


@app.callback(
    [Output(component_id="factor",component_property="children"),
    Output(component_id="value",component_property="children"),
    Output(component_id="conversions_analysis",component_property="figure"),
    Output(component_id="dropdown_var1",component_property="style"),
    Output(component_id="dropdown_var2",component_property="style"),
    Output(component_id="dropdown_var3",component_property="style"),
    Output(component_id="dropdown_var4",component_property="style"),
    Output(component_id="forecasting",component_property="figure"),
    Output(component_id="ROI",component_property="children"),
    Output(component_id="CVR",component_property="children"),
    Output(component_id="CPC",component_property="children")],
    [Input(component_id="dropdown_vars",component_property="value"),
    Input(component_id="dropdown_var1",component_property="value"),
    Input(component_id="dropdown_var2",component_property="value"),
    Input(component_id="dropdown_var3",component_property="value"),
    Input(component_id="dropdown_var4",component_property="value")]
)

def update_forecast(slct_var, slct_campaign, slct_company, slct_channel, slct_location):
    
    campaign_style = {"position":"absolute","top":"0","left":"0", "zIndex": 1}
    company_style = {"position":"absolute","top":"0","left":"0", "zIndex": 1}
    channel_style = {"position":"absolute","top":"0","left":"0", "zIndex": 1}
    location_style = {"position":"absolute","top":"0","left":"0", "zIndex": 1}

    if slct_var == "Campaign_Type":
        campaign_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_campaign].copy()
    elif slct_var == "Company":
        company_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_company].copy()
    elif slct_var == "Channel_Used":
        channel_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_channel].copy()
    else:
        location_style["zIndex"] = 5
        df_segment = df[df[slct_var] == slct_location].copy()

    if df_segment.empty or len(df_segment) < 3:
        return "Sin datos", "0%", go.Figure(), campaign_style, company_style, channel_style, location_style, go.Figure(), 0, 0, 0

    df_ts = df_segment.groupby(["Date"] + all_categorical_vars).agg({
        "Conversions": "sum", 
        "Acquisition_Cost": "sum", 
        "Clicks": "sum",
        "Impressions": "sum", 
        "ROI": "mean", 
        "Conversion_Rate": "mean"
    }).reset_index().sort_values("Date")

    df_ts["CTR"] = (df_ts["Clicks"] / df_ts["Impressions"]).replace([np.inf, -np.inf], 0).fillna(0)
    df_ts["CPC"] = (df_ts["Acquisition_Cost"] / df_ts["Clicks"]).replace([np.inf, -np.inf], 0).fillna(0)
    df_ts["CPM"] = ((df_ts["Acquisition_Cost"] / df_ts["Impressions"]) * 1000).replace([np.inf, -np.inf], 0).fillna(0)
    df_ts["Conv_Lag1"] = df_ts["Conversions"].shift(1)
    df_ts["ROI_Lag1"] = df_ts["ROI"].shift(1)
    df_ts["Clicks_Lag1"] = df_ts["Clicks"].shift(1)
    df_ts["day_of_week"] = df_ts["Date"].dt.dayofweek

    df_model = df_ts.dropna()
    
    if df_model.empty:
        return "Datos insuficientes (lags)", "0%", go.Figure(), campaign_style, company_style, channel_style, location_style, go.Figure(), 0, 0, 0

    last_row_raw = df_model.iloc[[-1]][numerical_features + all_categorical_vars]
    X_segment_dummies = pd.get_dummies(last_row_raw, columns=all_categorical_vars)
    dummy_context_row = X_segment_dummies.reindex(columns=final_features, fill_value=0)
    
    importance = rf_conv.feature_importances_  
    
    df_imp = pd.DataFrame({
        "factor": final_features,
        "importance": importance
    }).sort_values(by="importance", ascending=True)

    top_row = df_imp.iloc[-1]
    factor_top_text = f"Impulsor principal: {top_row['factor']}"
    value_top_text = f"Influencia en el éxito: {round(top_row['importance'] * 100, 1)}%"
    
    slct_label = next((opt["label"] for opt in vars if opt["value"] == slct_var), "Variable")

    bar_chart = px.bar(
        df_imp.tail(3), 
        x="importance", 
        y="factor", 
        orientation="h",
        title=f"Impacto de los Factores en el éxito por {slct_label}",
        color="importance",
        color_continuous_scale="Blues"
    )
    
    bar_chart.update_layout(
        height=350,
        autosize=False,  
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )

    future_dates = pd.date_range(df_model["Date"].max() + pd.Timedelta(days=1), periods=14)
    last_row = df_model.iloc[-1]
    mean_cost, mean_clicks = df_ts["Acquisition_Cost"].mean(), df_ts["Clicks"].mean()
    last_cpc = mean_cost / mean_clicks if mean_clicks > 0 else 0

    dates, conversions, ROIs, CVRs, CPCs = [last_row["Date"]], [last_row["Conversions"]], [last_row["ROI"]], [last_row["Conversion_Rate"]], [last_cpc]
    curr_conv, curr_roi, curr_clicks_lag, curr_ctr, curr_cpc, curr_cpm = last_row["Conversions"], last_row["ROI"], mean_clicks, last_row["CTR"], last_row["CPC"], last_row["CPM"]
    
    X_raw_segment = df_model[numerical_features + all_categorical_vars]
    X_segment_full = pd.get_dummies(X_raw_segment, columns=all_categorical_vars)
    X_segment_full = X_segment_full.reindex(columns=final_features, fill_value=0)
    dummy_pred_row = X_segment_full.iloc[[-1]].copy()

    for date in future_dates:
        dummy_pred_row["Conv_Lag1"] = curr_conv
        dummy_pred_row["ROI_Lag1"] = curr_roi
        dummy_pred_row["Clicks_Lag1"] = curr_clicks_lag
        dummy_pred_row["day_of_week"] = date.dayofweek
        dummy_pred_row["CTR"] = curr_ctr
        dummy_pred_row["CPC"] = curr_cpc
        dummy_pred_row["CPM"] = curr_cpm
        
        dummy_pred_row = dummy_pred_row[final_features]
        
        pred_conv = max(0, rf_conv.predict(dummy_pred_row)[0]) 
        pred_roi = rf_roi.predict(dummy_pred_row)[0]
        pred_cvr = rf_cvr.predict(dummy_pred_row)[0]
        cpc_pred = mean_cost / mean_clicks if mean_clicks > 0 else 0
        
        dates.append(date)
        conversions.append(pred_conv)
        ROIs.append(pred_roi)
        CVRs.append(pred_cvr)
        CPCs.append(cpc_pred)
    
        curr_conv, curr_roi = pred_conv, pred_roi
        
    df_forecast = pd.DataFrame({
        "Date": dates,
        "Conversions": conversions,
        "ROI": ROIs,
        "CVR": CVRs,
        "CPC": CPCs
    })

    df_ts_recent = df_ts[df_ts["Date"] >= "2021-09-01"]
    
    forecasting = go.Figure()
    forecasting.add_trace(go.Scatter(x=df_ts_recent["Date"], y=df_ts_recent["Conversions"], mode="lines", fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.15)", name="Conversiones Históricas"))
    forecasting.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Conversions"], mode="lines+markers", fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.15)", name="Pronóstico de 14 días"))
    forecasting.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Fecha", yaxis_title="Conversiones")
    
    ROI_val = str(round(df_forecast["ROI"].mean(), 2))
    CVR_val = str(round(df_forecast["CVR"].mean(), 2))
    CPC_val = str(round(df_forecast["CPC"].mean(), 2))

    return factor_top_text, value_top_text, bar_chart, campaign_style, company_style, channel_style, location_style, forecasting, ROI_val, CVR_val, CPC_val


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
