import pandas as pd
import numpy as np
import statsmodels.api as sm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Output, Input


df = pd.read_csv("data/digital_advertising.zip")

df["Date"] = pd.to_datetime(df["Date"])
df["Acquisition_Cost"] = df["Acquisition_Cost"].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"])
df["Conversions"] = round(df["Clicks"] * df["Conversion_Rate"])

df_clean = df.copy()

for col in ["Campaign_Type","Company","Location","Channel_Used"]:
    df_clean[col] = df_clean[col].astype(str).str.strip()

cat_vars = ["Campaign_Type","Company","Location","Channel_Used"]
df_dummies = pd.get_dummies(df_clean[cat_vars], dtype=float)

for col in df_dummies.columns:
    df_clean[col] = df_dummies[col] * df_clean["Clicks"]

exogenous_cols = list(df_dummies.columns)

df_daily = df_clean.groupby("Date").agg({
    "Conversions": "sum",
    "Clicks": "sum",
    "Impressions": "sum",
    "Acquisition_Cost": "sum",
    **{col: "sum" for col in exogenous_cols}
}).asfreq("D")

y = df_daily["Conversions"]
X = df_daily[exogenous_cols]

sarimax = sm.tsa.statespace.SARIMAX(
    y,
    exog=X,
    order=(1, 0, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_model = sarimax.fit(disp=False)

factor_top_b = html.B(children=[], id="factor")
value_top_b = html.B(children=[], id="value")

CPA = html.B(children=[], id="CPA")
CVR = html.B(children=[], id="CVR")
CPC = html.B(children=[], id="CPC")

vars = [
    {"label":"Tipo de campaña","value":"Campaign_Type"},
    {"label":"Compañía","value":"Company"},
    {"label":"Canal usado","value":"Channel_Used"},
    {"label":"Locación","value":"Location"}
]

periods = [
       {"label":"1 semana","value":7},
       {"label":"2 semanas","value":14},
       {"label":"3 semanas","value":21},
       {"label":"4 semanas","value":28}
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
                clearable=False),
    ]),
    html.Div(id="graph_div_1", className="e6_graph_div_1", children=[
        html.Div(id="KPI_div_1", className="e6_KPI_div_1", children=[
            html.P(factor_top_b, className="e6_KPI_1", style={"margin-right":"20px"}),
            html.P(value_top_b, className="e6_KPI_1", style={"margin-left":"20px"})
        ]),
        dcc.Graph(id="conversions_analysis", figure={}, className="e6_graph_1")
    ]),
    html.H2("Proyección variable de conversiones", id="H2", className="e6_title"),
    html.Div(id="forecast_div", className="e6_forecast_div", children=[
        html.Div(id="KPI_div_2", className="e6_KPI_div_2", children=[
            html.Div(className="e6_KPI_2", children=[html.P("CPA", className="e6_KPI_title"), html.P(["$",CPA], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CVR", className="e6_KPI_title"), html.P([CVR,"%"], className="e6_KPI_p")]),
            html.Div(className="e6_KPI_2", children=[html.P("CPC", className="e6_KPI_title"), html.P(["$",CPC], className="e6_KPI_p")])
        ]),
        html.Div(id="graph_div_2", className="e6_graph_div_2", children=[
        html.Div(id="dropdown_div_2", className="e6_dropdown_div_2", children=[
            dcc.Dropdown(id="dropdown_var4", className="e6_dropdown_2",
                        options={"1 semana":7,
                                 "2 semanas":14,
                                 "3 semanas":21,
                                 "4 semanas":28},
                        value=14,
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="forecasting", figure={}, className="e6_graph_2")
        ])
    ])
])


@app.callback(
    [Output(component_id="factor", component_property="children"),
    Output(component_id="value", component_property="children"),
    Output(component_id="conversions_analysis", component_property="figure"),
    Output(component_id="forecasting", component_property="figure"),
    Output(component_id="CPA", component_property="children"),
    Output(component_id="CVR", component_property="children"),
    Output(component_id="CPC", component_property="children")],
    [Input(component_id="dropdown_vars", component_property="value"),
    Input(component_id="dropdown_period", component_property="value")]
)

def update_forecast(slct_var, slct_period):

    df_coef = pd.DataFrame({
        "Variable": sarimax_model.params.index,
        "Impact": sarimax_model.params.values,
        "P_value": sarimax_model.pvalues.values
    })

    df_filtered = df_coef[df_coef["Variable"].str.startswith(f"{slct_var}_")].copy()
    df_filtered["Category"] = df_filtered["Variable"].str.replace(f"{slct_var}_", "")
    df_filtered = df_filtered.sort_values(by="Impact", ascending=False)

    df_positives = df_filtered[df_filtered["Impact"] > 0]
    if not df_positives.empty:
        top_row = df_positives.sort_values(by="Impact", ascending=False).iloc[0]
        factor_top_text = f"Impulsor principal: {top_row["Category"]}"
        value_top_text = f"Impacto marginal neto: +{round(top_row["Impact"], 2)} leads/clic"
    else:
        factor_top_text = "Impulsor principal: Ningun impacto positivo"
        value_top_text = "Impacto marginal neto: N/A"

    barchart = px.bar(
        df_filtered,
        x="Impact",
        y="Category",
        orientation="h",
        title=f"Impacto Marginal Neto de {slct_var}",
        labels={"Impact": "Conversiones Adicionales por Clic", "Categoria": "Segmento"},
        color="Impacto",
        color_continuous_scale="RdYlGn"
    )

    barchart.update_layout(
        height=350,
        autosize=False,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )

    future_scenario = df_daily[exogenous_cols].tail(14).mean()

    future_dates = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=slct_period, freq='D')
    X_future = pd.DataFrame([future_scenario] * slct_period, index=future_dates)

    forecast = sarimax_model.forecast(steps=slct_period, exog=X_future)
    df_forecast = pd.DataFrame({"Conversiones Proyectadas": forecast}, index=future_dates)

    forecasting = px.line(
        df_forecast,
        y="Conversiones Proyectadas",
        title=f"Pronóstico de Conversiones Totales del Ecosistema - Próximos {slct_period} Días",
        labels={"index": "Fecha", "Conversiones Proyectadas": "Leads"}
    )

    forecasting.update_traces(line_color="red", line_dash="dash")
    forecasting.update_layout(template="plotly_white", height=350)

    mean_total_clics = df_daily["Clicks"].tail(14).mean()
    mean_total_cost = df_daily["Acquisition_Cost"].tail(14).mean()
    mean_total_impressions = df_daily["Impressions"].tail(14).mean()

    mean_projected_conversions = forecast.mean()

    cpa_projected = mean_total_cost / mean_projected_conversions if mean_projected_conversions > 0 else 0
    cvr_projected = (mean_projected_conversions / mean_total_clics) * 100 if mean_total_clics > 0 else 0
    cpc_projected = mean_total_cost  / mean_total_clics if mean_total_clics > 0 else 0

    CPA_val = f"${round(cpa_projected, 2)}"
    CVR_val = round(cvr_projected, 2)
    CPC_val = f"${round(cpc_projected, 2)}"

    return factor_top_text, value_top_text, barchart, forecasting, CPA_val, CVR_val, CPC_val


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
