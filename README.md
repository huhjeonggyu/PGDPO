# PG-DPO: Pontryagin-Guided Direct Policy Optimization

This repository is the **official implementation** of the following paper:

[**Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization for Continuous-Time Multi-Asset Portfolio**](https://arxiv.org/abs/2504.11116)  <!-- arXiv link 고정 -->

📄 Detailed code-level instructions are provided in the associated PDF guide:  
<https://drive.google.com/file/d/1b-HoFFeEu1tmXH0LWuaD-wszEic0GKfu/view?usp=sharing>

---

## Abstract

> Solving large-scale, continuous-time portfolio optimization problems involving numerous assets and state-dependent dynamics has long been challenged by the curse of dimensionality. Traditional dynamic programming and PDE-based methods, while rigorous, typically become computationally intractable beyond a few state variables (~3–6 limit in prior studies). To overcome this critical barrier, we introduce the Pontryagin-Guided Direct Policy Optimization (PG-DPO) framework. Our framework accurately captures both myopic demand and complex intertemporal hedging demands, a feat often elusive for other methods in high-dimensional settings. P-PGDPO delivers near-optimal policies, offering a practical and powerful alternative for a broad class of high-dimensional continuous-time control problems. PG-DPO leverages Pontryagin's Maximum Principle (PMP) and backpropagation-through-time (BPTT) to directly inform neural-network-based policy learning. A key contribution is our highly efficient Projected PG-DPO (P-PGDPO) variant. This approach uniquely utilizes BPTT to obtain rapidly stabilizing estimates of the Pontryagin costates and their crucial derivatives with respect to the state variables. These estimates are then analytically projected onto the manifold of optimal controls dictated by PMP's first-order conditions, significantly reducing training overhead and enhancing accuracy. This enables a breakthrough in scalability: numerical experiments demonstrate that P-PGDPO successfully tackles problems with dimensions previously considered far out of reach (up to 50 assets and 10 state variables). :contentReference[oaicite:0]{index=0}

---

## Citation

```bibtex
@article{huh2025breaking,
  title  = {Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization for Continuous-Time Multi-Asset Portfolio},
  author = {Huh, Jeonggyu and Jeon, Jaegi and Koo, Hyeng Keun},
  journal = {arXiv preprint arXiv:2504.11116},
  year   = {2025}
}
