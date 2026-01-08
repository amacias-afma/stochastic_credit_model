# 🛡️ Stochastic Credit Risk & ALM Engine
### A Hybrid Impulse Control Framework for Mortgage Dynamics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Math](https://img.shields.io/badge/Math-Stochastic%20Control-purple)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Executive Summary

This project implements a **PhD-level Quantitative Finance framework** for modeling Retail Credit Risk (Mortgages/Consumer Loans) within an **Asset-Liability Management (ALM)** context.

Unlike traditional "Siloed" models that estimate *Probability of Default* (PD) and *Prepayment* (CPR) independently, this engine uses a **Unified Competing Risks Framework**. It models the loan lifecycle as a **Stochastic Impulse Control Problem**, where the borrower exercises embedded options (Default Put / Prepayment Call) based on the state of the economy (Interest Rates) and their creditworthiness.

The system is designed to capture **Negative Convexity** and comply with **Basel III IRRBB** (Interest Rate Risk in the Banking Book) standards.

---

## 📚 Mathematical Framework

### 1. The Measure Space (ZOIB)
We model the payment fraction $Y_t \in [0, 1]$ as a random variable on a **Singular Mixed Measure** (Zero-One Inflated). The conditional density is defined as:

$$
f(y \mid \mathcal{F}_{t}) = \underbrace{\pi_d(\mathbf{x}_t) \cdot \delta_0(y)}_{\text{Default (Absorbing)}} + \underbrace{\pi_p(\mathbf{x}_t) \cdot \delta_1(y)}_{\text{Prepayment (Absorbing)}} + \underbrace{\pi_c(\mathbf{x}_t) \cdot g(y; \theta)}_{\text{Continuing (Transient)}}
$$

Where:
*   **$\delta(\cdot)$**: The Dirac Delta mass (Absorbing Boundaries).
*   **$\pi(\cdot)$**: Regime probabilities estimated via **XGBoost Classifier** (Softmax).
*   **$g(y)$**: Continuous density of partial payments estimated via **Logit-Link Regressor**.

### 2. System Dynamics
The outstanding balance $B_t$ evolves according to the stochastic difference equation:

$$ B_{t+1} = B_t \cdot (1 + r \Delta t) - B_t \cdot Y_t $$

This formulation rigorously captures the path-dependency required for **Lifetime Expected Credit Loss (ECL)** calculations under IFRS 9.

---

## 🏗️ Project Architecture

```text
stochastic_credit_alm/
├── data/
│   ├── raw/                   # Landing zone for LendingClub CSVs
│   └── processed/             # Optimized Parquet panel data
├── src/
│   ├── etl_lendingclub.py     # Panel Reconstruction & Imputation Logic
│   ├── hybrid_model.py        # ZOIB Estimator (Classifier + Regressor)
│   ├── financial_engine.py    # Vectorized Monte Carlo Particle Filter
│   ├── market_data.py         # Vasicek Calibration via St. Louis Fed (FRED)
│   └── backtesting.py         # Out-of-Time (OOT) Stability Analysis
├── dashboard/
│   ├── app.py                 # Main Streamlit Router
│   ├── intro.py               # Strategic Context & ALM Documentation
│   └── math_doc.py            # LaTeX Mathematical Documentation
├── requirements.txt           # Python dependencies
└── README.md                  # Project Documentation