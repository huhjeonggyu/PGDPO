# PG-DPO Portfolio Optimization — Experiment Scripts

This repository contains a series of Python scripts that implement and test **Pontryagin-Guided Direct Policy Optimization (PG-DPO)** and its variants, step-by-step. The scripts progress from a baseline policy optimization to more advanced variants, demonstrating the effect of variance reduction, residual learning, control variates, and Richardson extrapolation.

**Test Environment:**  
All experiments here are run on the **Kim and Omberg (1996)** continuous-time portfolio model. This model is particularly convenient for testing because it has a **known closed-form solution** for the optimal policy, allowing us to directly compute RMSE and expected utility differences between learned policies and the analytical optimum.

## File Descriptions

### `pgdpo_base_single.py` — **Baseline PG-DPO**
Trains a Stage-1 PG-DPO policy without variance reduction or other enhancements.  
Serves as the starting point for all subsequent experiments. Reports:
- Stage-1 RMSE vs closed-form policy
- Expected utility difference vs closed-form

---

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Direct Costates)**
Takes the baseline Stage-1 policy and applies PMP-based projection using **directly computed costates** from BPTT.  
Demonstrates the accuracy boost of projection without any variance reduction.

---

### `pgdpo_antithetic_single.py` — **Antithetic PG-DPO**
Adds **antithetic sampling** to reduce Monte Carlo variance in Stage-1 training and costate estimation.  
Improves RMSE and expected utility before projection, and further sharpens P-PGDPO projection accuracy.

---

### `pgdpo_residual_single.py` — **Residual PG-DPO**
Trains a residual network on top of the closed-form policy baseline, learning only the **hedging demand correction**.  
This architecture stabilizes training and dramatically reduces RMSE in Stage-1.

---

### `pgdpo_cv_single.py` — **Residual + Control Variate**
Extends residual PG-DPO with a **control variate** based on closed-form utility to further reduce variance in Stage-1 training loss.  
Maintains low RMSE while improving training stability in noisy conditions.

---

### `pgdpo_richardson_single.py` — **Residual + CV + Richardson Extrapolation**
Same Stage-1 training as CV, but adds a simulation **Richardson extrapolation** module to demonstrate timestep bias reduction in expected utility estimation.  
**Note:** Richardson extrapolation is often highly effective in high-dimensional problems (where timestep bias is large), but in this low-dimensional Kim–Omberg setup, it shows little or even slightly negative impact due to already small bias and increased variance.

---

## Stage-by-Stage RMSE Results

| Stage             | RMSE(Stage1 vs CF) | RMSE(PPGDPO vs CF) |
|-------------------|--------------------|--------------------|
| BASE              | 0.233300           | –                  |
| P-PGDPO(direct)   | 0.233300           | 0.005522           |
| Antithetic        | 0.164274           | 0.004254           |
| Residual          | 0.005663           | 0.003651           |
| CV                | 0.036626           | 0.003179           |
| Richardson        | 0.037177           | 0.003411           |

**Interpretation:**
- Stage-1 RMSE drops dramatically from BASE to Residual.
- P-PGDPO projection consistently produces RMSE < 0.006 across all variants.
- Antithetic, Residual, and CV variants show substantial Stage-1 improvement over BASE.
- Richardson improves high-dimensional cases significantly, but in this low-dimensional Kim–Omberg setup, its effect is minimal and can be slightly negative due to variance amplification.
