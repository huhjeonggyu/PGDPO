# 🐍 PG-DPO Portfolio Optimization — Experiment Scripts

Welcome! 🎉  
This repository is a **tutorial-style** collection of Python scripts that implement and test **Pontryagin-Guided Direct Policy Optimization (PG-DPO)** and its variants — step-by-step, from the most basic to more advanced forms.  
It’s designed not just for research, but also as a **hands-on learning resource** 📚 so you can follow along, modify, and experiment.

---

## 🧪 Test Environment

All experiments here are run on the **Kim and Omberg (1996)** continuous-time portfolio model 🏦.  
This model is ideal for tutorials because it has a **known closed-form optimal policy**, letting us directly measure:

- 📏 **RMSE** between learned and analytical policies
- 📈 **Expected utility difference** from the optimum

---

## 📂 File Descriptions

### `pgdpo_base_single.py` — **Baseline PG-DPO**
🚀 The starting point.  
Trains a Stage-1 PG-DPO policy **without** variance reduction or other enhancements. Reports:
- Stage-1 RMSE vs closed-form policy
- Expected utility difference vs closed-form

---

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Direct Costates)**
🎯 Takes the baseline policy and applies **PMP-based projection** using costates from BPTT.  
Shows how projection alone boosts accuracy.

---

### `pgdpo_antithetic_single.py` — **Antithetic PG-DPO**
🔄 Adds **antithetic sampling** to cut Monte Carlo variance in training and costate estimation.  
Improves both RMSE and utility.

---

### `pgdpo_residual_single.py` — **Residual PG-DPO**
🛠 Builds a residual network on top of the closed-form baseline, learning only **hedging demand corrections**.  
Training becomes more stable and RMSE drops dramatically.

---

### `pgdpo_cv_single.py` — **Residual + Control Variate**
🎚 Adds a **control variate** based on closed-form utility to residual PG-DPO.  
Further reduces variance and improves robustness under noisy conditions.

---

### `pgdpo_richardson_single.py` — **Residual + CV + Richardson Extrapolation**
⏩ Adds **Richardson extrapolation** to the CV setup to reduce timestep bias in utility estimates.  
⚠️ In low-dimensional Kim–Omberg, effect is minimal (and can be slightly negative) due to already tiny bias, but **in high dimensions it can shine** ✨.

---

## 📊 Stage-by-Stage RMSE Results

| Variant                                   | Stage 1 RMSE | Stage 2 RMSE |
|-------------------------------------------|--------------|--------------|
| Baseline PG-DPO                           | 0.233300     | –            |
| Projected PG-DPO (P-PGDPO)                 | 0.233300     | 0.005522     |
| P-PGDPO + Antithetic                       | 0.164274     | 0.004254     |
| P-PGDPO + Residual                         | 0.005663     | 0.003651     |
| P-PGDPO + Residual + Control Variate (CV)  | 0.036626     | 0.003179     |
| P-PGDPO + Residual + CV + Richardson       | 0.037177     | 0.003411     |

---

### 📝 Interpretation
- 📉 **Stage-1 RMSE** drops sharply from BASE → Residual.  
- 🏆 **P-PGDPO projection** consistently gives RMSE < 0.006 across all variants.  
- 💡 Antithetic, Residual, and CV improve Stage-1 accuracy well beyond BASE.  
- ⏳ Richardson is more valuable in **high-dimensional problems**.

---

💡 **Note:** While this repo is research-grade, it’s deliberately structured as a **learning-friendly tutorial** — you can run each script independently, compare results, and see exactly how each enhancement changes the outcome.

