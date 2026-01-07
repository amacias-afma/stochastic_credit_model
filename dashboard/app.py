import streamlit as st
# import requests
# import pandas as pd
# import plotly.graph_objects as go

from intro import render_introduction
from math_explanation import render_math_explanation
from analysis_tab import render_analysis_tab
from stochastic_flux import stochastic_flux
from simulation_page import render_simulation_page
from lab_page import render_lab_page

# --- Global Resource Loading ---
@st.cache_resource
def load_resources():
    try:
        # Load the ZOIB Model
        model = StochasticHybridModel.load('hybrid_model.pkl')
        # Load/Calibrate Market Data (Vasicek)
        env = MarketEnvironment()
        market_params = env.calibrate_vasicek()
        return model, market_params
    except Exception as e:
        return None, {'current_rate': 0.045}
    
# --- Main App Controller ---
def main():
    st.set_page_config(layout="wide", page_title="Stochastic Credit Engine")
    
    # Load Resources
    model, market_params = load_resources()
    
    # --- GLOBAL NAVIGATION (The Router) ---
    with st.sidebar:
        st.title("🛡️ Credit Engine")
        # Radio button acts as the 'Page State'
        page = st.radio("Navigation", [
            "📖 Strategic Context", 
            "⚡ Stochastic Flux", 
            "🧪 Research Lab",
            "🎓 Mathematical Model"
        ])
        st.divider()

    # --- ROUTING LOGIC ---
    if page == "📖 Strategic Context":
        render_introduction()
        
    elif page == "⚡ Stochastic Flux":
        render_simulation_page(model, market_params) # Sidebar inputs are defined INSIDE here
        
    elif page == "🧪 Research Lab":
        render_analysis_tab()
        
    elif page == "🎓 Mathematical Model":
        render_math_explanation()

# import sys
# import os

# # Add project root to sys.path to resolve 'src' module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # import plotly.express as px
# from src.market_data import MarketEnvironment
# from src.backtesting import run_backtest_experiment, plot_confusion_heatmap

# def main():
#     st.set_page_config(layout="wide", page_title="Stochastic Credit ALM")
    
#     st.title("🛡️ Stochastic Credit Risk & ALM Engine")
#     st.markdown("*PhD Applied Math Project | Stochastic Control & Machine Learning*")

#     # Create Tabs - "Strategic Context" comes first
#     tab_intro, tab_math, tab_sim, tab_lab = st.tabs([
#         "📖 Strategic Context", 
#         "🎓 Mathematical Model",
#         "⚡ Hybrid Simulation (ZOIB)", 
#         "🧪 Research Lab"
#     ])
    
#     with tab_intro:
#         render_introduction()
        
#     with tab_sim:
#         stochastic_flux()
#         # Your existing simulation logic
#         # pass
        
#     # with tab_lab:
#     #     # Your existing backtesting logic
#     #     pass

#     with tab_math:
#         render_math_explanation()

# # --- Main Execution ---
# def main():
#     st.set_page_config(layout="wide", page_title="Stochastic Credit Model")
#     st.title("🛡️ Stochastic Credit Risk Model")
    
#     # Create Tabs to separate "Engine" from "Theory"
#     tab1, tab2, tab3, tab4 = st.tabs(["🚀 Trajectory Simulation", "📉 Flux Analysis", "🎓 Mathematical Model", "🧪 Quantitative Research Lab"])
    
#     with tab1:
#         st.write("Place your Single Trajectory code here...")
    
#     with tab2:
#         stochastic_flux()
    
#     with tab3:
#         render_math_explanation()

#     with tab4:
#         render_analysis_tab()

if __name__ == "__main__":
    main()