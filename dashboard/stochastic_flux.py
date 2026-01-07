import streamlit as st

def stochastic_flux():
    st.subheader("📊 Stochastic Flux & Confidence Intervals")

    with st.sidebar:
        st.markdown("---")
        st.markdown("## Flux Analysis Controls")
        principal = st.number_input("Principal ($)", 100000, 1000000, 300000)
        fico = st.slider("FICO", 300, 850, 720)
        dti = st.slider("DTI", 0.1, 0.6, 0.35)
        orig_rate = st.slider("Contract Rate", 0.02, 0.10, 0.065)
        market_rate = st.slider("Market Rate", 0.02, 0.10, 0.045)
        months = st.slider("Horizon", 12, 120, 60)

    if st.button("Simulate Flux"):
        payload = {
            "principal": principal, "fico": fico, "dti": dti,
            "orig_rate": orig_rate, "market_rate": market_rate, "months": months,
            "n_sims": 2000
        }

        with st.spinner("Running Monte Carlo (N=2000)..."):
            try:
                response = requests.post("http://localhost:8000/calculate_stochastic_flux", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    fig = go.Figure()
                    x = data['months']

                    # 1. The Fan (Confidence Interval)
                    fig.add_trace(go.Scatter(
                        x=x + x[::-1],
                        y=data['ci_upper'] + data['ci_lower'][::-1],
                        fill='toself',
                        fillcolor='rgba(0,176,246,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='90% Confidence Interval',
                        showlegend=True
                    ))

                    # 2. Expected Flux (Stochastic Mean)
                    fig.add_trace(go.Scatter(
                        x=x, y=data['expected_flux'],
                        line=dict(color='rgb(0,176,246)', width=3),
                        name='Expected Flux (Risk Adjusted)'
                    ))

                    # 3. Input Flux (Contractual)
                    fig.add_trace(go.Scatter(
                        x=x, y=data['input_flux'],
                        line=dict(color='gray', dash='dash'),
                        name='Input Flux (Contractual)'
                    ))

                    fig.update_layout(
                        title="Flux Analysis: Contractual vs. Expected vs. Risk",
                        xaxis_title="Month",
                        yaxis_title="Cash Flow ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, width=1200)

                    # Metric Analysis
                    total_input = sum(data['input_flux'])
                    total_expected = sum(data['expected_flux'])
                    diff = total_expected - total_input

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Contractual Value", f"${total_input:,.0f}")
                    c2.metric("Risk-Adjusted Value", f"${total_expected:,.0f}", delta=f"{diff:,.0f}")
                    c3.metric("Valuation Adjustment", f"{(diff/total_input):.2%}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
