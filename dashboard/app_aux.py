import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Import Modules ---
from intro import render_introduction
from math_doc import render_math_explanation
from src.hybrid_model import StochasticHybridModel
from src.financial_engine import HybridFinancialEngine
from src.market_data import MarketEnvironment

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

    # You can import and call render_backtest() here

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
        render_lab_page()
        
    elif page == "🎓 Mathematical Model":
        render_math_explanation()

if __name__ == "__main__":
    main()