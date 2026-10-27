---
sidebar_position: 3
title: Taxonomy of Optimizers
---

# Taxonomy of Optimizers

Opytimizer organizes algorithms into distinct categories based on their underlying inspiration and behavior:

* **Swarm-Based**: Algorithms modeled after collective intelligence (e.g., Particle Swarm Optimization, Ant Colony Optimization, Artificial Bee Colony).
* **Evolutionary**: Algorithms based on natural selection and genetic operators (e.g., Genetic Algorithms, Differential Evolution, Genetic Programming).
* **Science / Physics-Based**: Algorithms inspired by physical laws, chemical processes, or mathematical theories (e.g., Simulated Annealing, Gravitational Search Algorithm, Harmony Search).
* **Population-Based**: Algorithms modeled after the hunting and survival dynamics of animal populations (e.g., Grey Wolf Optimizer, Harris Hawks Optimization, Aquila Optimizer).
* **Social-Based**: Algorithms inspired by human social behavior (e.g., Brain Storm Optimization, Cohort Intelligence, Interactive Search Algorithm).
* **Boolean**: Algorithms that operate over binary decision variables (e.g., Boolean Particle Swarm Optimization, Boolean Manta Ray Foraging Optimization, UMDA).
* **Miscellaneous**: Algorithms that do not fit the categories above (e.g., Grid Search, Hill-Climbing, Cross-Entropy Method).
* **Multi-Objective**: Algorithms designed to optimize multiple conflicting objectives concurrently (e.g., NSGA-II, MOEA/D).

## Package Layout

Each category maps to a sub-package of `opytimizer.optimizers`:

* **`single_objective.swarm`**: `ABC`, `ABO`, `ACO`, `AF`, `BA`, `BOA`, `BWO`, `CS`, `CSA`, `EHO`, `FA`, `FFOA`, `FPA`, `FSO`, `GOA`, `JS`, `KH`, `MFO`, `MRFO`, `PIO`, `PSO`, `SBO`, `SCA`, `SFO`, `SOS`, `SSA`, `SSO`, `STOA`, `WAOA`, `WOA`, and variants.
* **`single_objective.evolutionary`**: `BSA`, `CE`, `CMAES`, `DE`, `EP`, `ES`, `FOA`, `GA`, `GP`, `HS`, `IWO`, `LSHADE`, `RRA`, and variants.
* **`single_objective.science`**: `AIG`, `ASO`, `BH`, `CDO`, `EFO`, `EO`, `ESA`, `GSA`, `HGSO`, `LSA`, `MOA`, `MVO`, `SA`, `SMA`, `TEO`, `TWO`, `WCA`, `WDO`, `WEO`, `WWO`.
* **`single_objective.population`**: `AEO`, `AO`, `COA`, `EPO`, `GCO`, `GWO`, `HHO`, `LOA`, `OSA`, `PPA`, `PVS`, `RFO`.
* **`single_objective.social`**: `BSO`, `CI`, `ISA`, `LEO`, `MVPA`, `QSA`, `SSD`.
* **`single_objective.boolean`**: `BMRFO`, `BPSO`, `UMDA`.
* **`single_objective.misc`**: `AOA`, `CEM`, `DOA`, `GS`, `HC`, `NDS`.
* **`multi_objective.evolutionary`**: `KnEA`, `MOCE`, `MOEAD`, `NSGA2`, `NSGA3`, `RVEA`, `SPEA2`, and variants.

## Tensorized Variants

Some optimizers provide vectorized versions that update the whole population at once and can run on GPU:

* **`*Tensor`**: Vectorized implementations that require a tensorized space (e.g., `PSOTensor`, `CMAESTensor`, `LSHADETensor`, `NSGA2Tensor`, `MOEADTensor`, `RVEATensor`).
* **`*Cuda`**: Implementations tailored for CuPy execution (e.g., `PSOCuda`, `CMAESCuda`, `RVEACuda`).
