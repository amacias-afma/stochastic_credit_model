import streamlit as st

def render_introduction():
    st.markdown("## 🎯 Strategic Context & Mathematical Framework")
    
    # Executive Summary
    st.info("""
    **Executive Summary:** This project implements a **Stochastic Impulse Control Model** for Retail Credit Risk (Mortgages). 
    Unlike traditional "Siloed" approaches, this model unifies **Default (Credit Risk)** and **Prepayment (Market Risk)** 
    into a single **Competing Risks Framework**. This ensures consistent valuation and risk measurement for 
    **Asset-Liability Management (ALM)** and Regulatory Compliance.
    """)

    st.divider()

    # --- Pillar 1: The ALM Imperative ---
    st.subheader("1. The ALM Imperative: Managing Negative Convexity")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**The Challenge: Embedded Optionality**")
        st.write("""
        From an ALM perspective, a mortgage is not a static bond. It is a complex derivative where the borrower holds two American options against the bank:
        1.  **Put Option (Default):** The borrower forces the bank to buy the asset at the recovery price (exercise when House Price < Loan Value).
        2.  **Call Option (Prepayment):** The borrower forces the bank to sell the asset at par value (exercise when Market Rate < Loan Rate).
        """)
    
    with col2:
        st.markdown("**The Risk: Duration Mismatch**")
        st.warning("""
        *   **Scenario A (Rates Fall):** Borrowers exercise the Call (Prepay). The asset duration shortens exactly when the bank needs long-duration assets to match liabilities. (**Reinvestment Risk**).
        *   **Scenario B (Rates Rise):** Prepayments slow down. The asset duration extends exactly when the bank needs liquidity. (**Extension Risk**).
        """)

    st.markdown("> **Conclusion:** Without a joint stochastic model, the bank miscalculates the **Effective Duration** and **Economic Value of Equity (EVE)**.")

    st.divider()

    # --- Pillar 2: The Unified Theory ---
    st.subheader("2. Beyond Siloed Modeling: The PhD Approach")
    
    st.markdown("""
    **The Industrial Fallacy:**
    Most banks run separate models: a *Logistic Regression* for PD (Probability of Default) and a separate *S-Curve* for CPR (Prepayment).
    This assumes orthogonality: $P(\text{Default} \cap \text{Prepay}) = P(\text{Default}) \times P(\text{Prepay})$.
    """)

    # Comparison Table for Impact
    st.markdown("""
    | Feature | Traditional (Siloed) | PhD Approach (Unified/ZOIB) |
    | :--- | :--- | :--- |
    | **Probability Space** | Independent ($\sum P > 1$) | **Mutually Exclusive** ($\sum P = 1$) |
    | **Interaction** | None (Static) | **Burnout & Lock-in Effects** |
    | **Mathematical Flaw** | Orthogonality Error | None (Law of Total Probability) |
    """)

    st.caption("*Lock-in Effect: If Credit Quality drops (High PD), Prepayment naturally drops to zero (cannot refinance). Siloed models miss this natural hedge.*")

    st.divider()

    # --- Pillar 3: Regulatory Compliance ---
    st.subheader("3. Regulatory Compliance (Basel III & IRRBB)")
    st.write("This architecture is designed to satisfy strict regulatory pillars:")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### 🏛️ IRRBB (Std 368)")
        st.caption("**Behavioral Option Risk**")
        st.write("""
        "Banks must have rigorous procedures to quantify... embedded options in assets. Assumptions must be conceptually sound and empirically supported."
        *(BCBS Standards, April 2016)*
        """)
        
    with c2:
        st.markdown("#### 📉 Stress Testing")
        st.caption("**Liquidity Coverage Ratio (LCR)**")
        st.write("""
        To calculate the LCR denominator (Net Cash Outflows), the model must simulate inflows under stress. A massive Prepayment wave significantly alters the liquidity buffer.
        """)
        
    with c3:
        st.markdown("#### ⚖️ IFRS 9 / CECL")
        st.caption("**Forward-Looking ECL**")
        st.write("""
        Provisions are based on **Lifetime Expected Credit Loss**. This requires integrating the PD over the full stochastic life of the loan, which is heavily dependent on the Prepayment assumptions.
        """)