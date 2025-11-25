#!/usr/bin/env python3
"""
Unmeasured Per Household Consumption – SARIMAX Forecast Studio (clean final version)

This is a minimal and validated Streamlit app. Use `streamlit run app.py`.
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import date
import io
import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def set_style():
    st.set_page_config(page_title=' Unmeasured Per Household Consumption – SARIMAX Forecast Studio', page_icon='📈', layout='wide')
    st.markdown(
        '''
        <style>
        :root{--blue:#0ea5e9;--orange:#fb923c}
        .stButton>button{background:linear-gradient(135deg,var(--blue),var(--orange));color:#fff}
        .section-card{background:rgba(255,255,255,0.03);border-radius:10px;padding:12px}
        .metric-card{background:rgba(255,255,255,0.02);padding:10px;border-radius:10px}
        .section-title{color:var(--blue);text-transform:uppercase}
        </style>
        ''', unsafe_allow_html=True)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def build_template(start_month: pd.Timestamp, months: int = 6) -> pd.DataFrame:
    dates = pd.date_range(start=start_month, periods=months, freq='MS')
    return pd.DataFrame({
        'Date': dates,
        'Mean_Air_Temperature': [np.nan] * months,
        'Annual_sunshine': [np.nan] * months,
        'Maximum_Temperature': [np.nan] * months,
        'Annual_Precipitation': [np.nan] * months,
        'minimum_air_temperature': [np.nan] * months,
    })


def normalize_conf_int(conf_int: pd.DataFrame):
    cols = list(conf_int.columns)
    flat = ['_'.join(c) if isinstance(c, tuple) else str(c) for c in cols]
    lower_i = next((i for i, c in enumerate(flat) if c.lower().startswith('lower')), None)
    upper_i = next((i for i, c in enumerate(flat) if c.lower().startswith('upper')), None)
    if lower_i is None or upper_i is None:
        raise ValueError('CI columns could not be parsed')
    return cols[lower_i], cols[upper_i]


def plot_results(results_df: pd.DataFrame):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='#0ea5e9')))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Upper'], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Lower'], mode='lines', fill='tonexty', showlegend=False))
        fig.update_layout(template='plotly_dark', height=420)
        st.plotly_chart(fig, use_container_width=True)
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(results_df.index, results_df['Forecast'], marker='o', color='#0ea5e9')
        ax.fill_between(results_df.index, results_df['Lower'], results_df['Upper'], color='#fb923c', alpha=0.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Uphc')
        st.pyplot(fig)
    else:
        st.info('Install plotly or matplotlib to view charts')


def main():
    set_style()
    st.title('📈 Unmeasured Per Household Consumption — SARIMAX Forecast Studio')
    st.write('Configure a 6-month scenario and forecast Unmeasured Per Household Consumption using a SARIMAX model.')

    with st.sidebar:
        st.markdown('## Model & settings')
        model_path = st.text_input('Model path (joblib)', value='uPHC_sarimax_forecast_model.pkl')
        if st.button('Load model'):
            try:
                _ = load_model(model_path)
                st.success('Model loaded')
            except Exception as e:
                st.error('Failed to load model')
                st.exception(e)

    sarimax = None
    if os.path.exists('uPHC_sarimax_forecast_model.pkl'):
        try:
            sarimax = load_model('uPHC_sarimax_forecast_model.pkl')
        except Exception:
            sarimax = None

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Step 1 — Scenario setup</div>', unsafe_allow_html=True)
    start_month = st.date_input('Start of forecast period', value=date.today().replace(day=1))
    start_month = start_month.replace(day=1)
    scenario_name = st.text_input('Scenario name', value='Base Case')

    template = build_template(start_month, months=6)
    edited = st.data_editor(template, key='exog', num_rows='fixed', use_container_width=True,
                            column_config={
                                'Date': st.column_config.DateColumn('Date', format='YYYY-MM-DD'),
                                'Mean_Air_Temperature': st.column_config.NumberColumn('Mean_Air_Temperature'),
                                'Annual_sunshine': st.column_config.NumberColumn('Annual_sunshine'),
                                'Maximum_Temperature': st.column_config.NumberColumn('Maximum_Temperature'),
                                'Annual_Precipitation': st.column_config.NumberColumn('Annual_Precipitation'),
                                'minimum_air_temperature': st.column_config.NumberColumn('minimum_air_temperature'),
                            })
    st.markdown('</div>', unsafe_allow_html=True)

    features = ['Mean_Air_Temperature', 'Annual_sunshine', 'Maximum_Temperature', 'Annual_Precipitation', 'minimum_air_temperature']
    missing = edited[features].isna().any().any()
    if missing:
        st.warning('Please fill all numeric inputs in the grid')

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Step 2 — Forecast</div>', unsafe_allow_html=True)

    if st.button('🚀 Run 6-month forecast'):
        if sarimax is None:
            st.error('No model loaded')
        elif missing:
            st.error('Missing inputs')
        else:
            exog = edited.copy()
            exog['Date'] = pd.to_datetime(exog['Date'])
            exog = exog.set_index('Date').asfreq('MS')[features].astype(float)
            try:
                res = sarimax.get_forecast(steps=len(exog), exog=exog)
                pred = np.asarray(res.predicted_mean).ravel()
                conf = res.conf_int(alpha=0.05)
            except Exception as e:
                st.error('Forecast failed')
                st.exception(e)
                return

            cols = list(conf.columns)
            if isinstance(cols[0], tuple):
                flat = ['_'.join(c) for c in cols]
            else:
                flat = [str(c) for c in cols]
            lower_i = next(i for i, c in enumerate(flat) if c.lower().startswith('lower'))
            upper_i = next(i for i, c in enumerate(flat) if c.lower().startswith('upper'))
            lower = np.asarray(conf[cols[lower_i]]).ravel()
            upper = np.asarray(conf[cols[upper_i]]).ravel()

            out = pd.DataFrame({'Forecast': pred, 'Lower': lower, 'Upper': upper}, index=exog.index)
            st.dataframe(out.round(3))
            plot_results(out)
            buf = io.StringIO(); out.to_csv(buf); st.download_button('Download CSV', data=buf.getvalue(), file_name='forecast.csv')

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
