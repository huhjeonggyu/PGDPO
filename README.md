# PG-DPO: Pontryagin-Guided Direct Policy Optimization

This repository is the **official implementation** of the following paper:

[**Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization for Continuous-Time Multi-Asset Portfolio**](https://arxiv.org/abs/2504.11116) <!-- arXiv link 고정 -->

📄 Detailed code-level instructions are provided in the associated PDF guide:  
<https://drive.google.com/file/d/1b-HoFFeEu1tmXH0LWuaD-wszEic0GKfu/view?usp=sharing>

---

## Citation

```bibtex
@article{huh2025breaking,
  title  = {Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization for Continuous-Time Multi-Asset Portfolio},
  author = {Huh, Jeonggyu and Jeon, Jaegi and Koo, Hyeng Keun},
  journal = {arXiv preprint arXiv:2504.11116},
  year   = {2025}
}

---

## 🎓 Educational Version (Colab-ready)

We also provide a simplified educational version of PG-DPO that can be run directly in Google Colab.  
This version is **single-asset**, based on the Merton model with **constant coefficients**, and has been simplified in various ways to better suit educational purposes — for example:

- Removes exogenous state variables, control variates, and residual learning  
- Focuses only on the core idea of PG-DPO and the comparison between the Stage-1 direct policy and the P-PGDPO projected policy

The goal is to make it easier to understand the core mechanics without the complexity of the full framework.

📎 **Colab-ready educational version**: [Google Drive Link](https://drive.google.com/file/d/1JfheqSXIIq2pZY8nLnbgYx_9E-xVb9re/view?usp=sharing)


