import streamlit as st
import pandas as pd
import sys
import os

# Add project root to sys.path to resolve 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import plotly.express as px
from src.market_data import MarketEnvironment
from src.backtesting import run_backtest_experiment, plot_confusion_heatmap


@st.cache_data
def load_data():
    return pd.read_parquet('data/processed/lending_club_panel.parquet')

# --- Add this Function to your Dashboard ---
def render_analysis_tab():
    st.header("🧪 Quantitative Research Lab")
    
    tab_market, tab_backtest = st.tabs(["📈 Market Calibration", "⚙️ Backtesting Engine"])
    
    # --- TAB A: MARKET DATA ---
    with tab_market:
        st.subheader("Vasicek Model Calibration (FRED)")
        st.markdown(r"""
        We fit the parameters to historical **30-Year Mortgage Rates** retrieved from the Federal Reserve.
        $$ dr_t = \kappa(\theta - r_t)dt + \sigma dW_t $$
        """)
        
        if st.button("Download & Calibrate Rates"):
            with st.spinner("Connecting to Federal Reserve API..."):
                env = MarketEnvironment()
                df_rates = env.fetch_data()
                params = env.calibrate_vasicek()
                
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean Reversion (κ)", f"{params['kappa']:.3f}")
                c2.metric("Long Term Mean (θ)", f"{params['theta']:.2%}")
                c3.metric("Volatility (σ)", f"{params['sigma']:.3f}")
                c4.metric("Fit ($R^2$)", f"{params['r_squared']:.3f}")
                
                # Visualization
                st.line_chart(df_rates)
                
                # Save to session state for Simulation Tab
                st.session_state['market_params'] = params
                st.success(f"Calibration Complete. Current Rate: {params['current_rate']:.2%}")

    # --- TAB B: BACKTESTING ---
    with tab_backtest:
        st.subheader("Walk-Forward Validation")
        st.markdown("Test the model's stability by training on the past and predicting the future.")
        
        df = load_data()
        min_date = pd.to_datetime(df['date']).min().date()
        max_date = pd.to_datetime(df['date']).max().date()
        print('min_date, max_date')
        print(min_date, max_date)
        print(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.info("Training Window (History)")
            train_start = st.date_input("Train Start", min_date)
            train_end = st.date_input("Train End", pd.to_datetime("2015-12-31"))
            
        with col2:
            st.warning("Testing Window (Future)")
            test_start = st.date_input("Test Start", pd.to_datetime("2016-01-01"))
            test_end = st.date_input("Test End", max_date)
            
        if st.button("Run Experiment"):
            with st.spinner("Training Model (This may take a moment)..."):
                results, error = run_backtest_experiment(
                    df, str(train_start), str(train_end), str(test_start), str(test_end)
                )
                
            if error:
                st.error(error)
            else:
                st.success(f"Trained on {results['train_n']} obs | Tested on {results['test_n']} obs")
                
                # 1. Confusion Matrix
                st.subheader("1. Transition Probability Matrix")
                fig = plot_confusion_heatmap(results['cm_norm'], results['label_names'])
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Detailed Metrics
                st.subheader("2. Class-Level Statistics")
                report_df = pd.DataFrame(results['report']).transpose()
                st.dataframe(report_df.style.format("{:.2%}").background_gradient(cmap='Blues'))
                
                # PhD Commentary
                with st.expander("🎓 PhD Interpretation Guide"):
                    st.write("""
                    *   **Diagonal (0,0):** Retention Rate.
                    *   **Cell (Late, Default):** This is the **Hazard Rate**. It answers: "If a loan is Late, what is the probability it moves to Default next month?" If your model predicts 0 here, it is failing to capture risk.
                    *   **Cell (Current, Prepaid):** This captures **Refinancing Efficiency**.
                    """)
