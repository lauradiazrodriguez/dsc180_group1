# DSC180A Quarter 1 Project — Causal Copilot: Data Simulation & Causal Discovery

This project replicates and extends two core components of the *Causal Copilot* framework:

1. **Synthetic Data Simulation**  
   Using configurable structural equation models (SEMs), we generate synthetic datasets under varying graph structures, sample sizes, noise distributions, and nonlinearity conditions. This enables controlled experimentation with causal discovery methods and highlights how data-generating assumptions impact learned causal relationships.

2. **Causal Discovery via the PC Algorithm**  
   We implement and test the **constraint-based PC algorithm** on the simulated datasets.  
   Our work explores:
   - Conditional independence testing  
   - Graph skeleton recovery  
   - Edge orientation rules  
   - Visualization of learned causal graphs  
   - How graph recovery quality changes under different noise levels, graph densities, and functional forms  

Together, these components form an end-to-end causal inference pipeline:
**data generation → causal discovery → graph evaluation and visualization**.

---

## 📓 Main Notebooks Included

### **1. generating_simulated_data.ipynb**
Generates synthetic datasets using:
- Linear and nonlinear SEMs  
- Gaussian and non-Gaussian noise  
- Adjustable graph density and variable counts  

Each run creates a timestamped folder under `simulated_data/`.

### **2. PC_alg.ipynb**
Runs the **PC algorithm** from the `causal-learn` package and uses utility functions (`pc_analysis_utils.py`) to:
- Perform conditional independence tests
- Recover the Completed Partially Directed Acyclic Graph (CPDAG) structure
- Visualize the output graph
- Compute Structural Hamming Distance (SHD)
- Compare the inferred structure to the true simulated causal graph  

This notebook completes the first causal discovery component of the project.

---

## Example Simulation Output Directory

```
simulated_data
└── 20251030_231358_LinearGaussian_d5_n1000_nodes5_samples1000
      ├── LinearGaussian_d5_n1000_config.json
      ├──LinearGaussian_d5_n1000_data.csv
      └──LinearGaussian_d5_n1000_graph.npy
```
---

# Running the Project Using Docker

To ensure reproducibility, we provide a minimal Docker environment that supports the dependencies required for:

- `generating_simulated_data.ipynb`
- `PC_alg.ipynb`

---

## 1. Build the Docker Image

From the repository root:

```bash
docker build -t causal-copilot-notebooks .
```

---

## 2. Run a Container With the Project Mounted

```bash
docker run --rm -it \
  -v "$(pwd):/workspace" \
  -p 8888:8888 \
  causal-copilot-notebooks
```

Then inside the container:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

Open:

```
http://localhost:8888
```

---

## 3. Running the Notebooks

Open and execute:

```
* generating_simulated_data.ipynb
* PC_alg.ipynb
```

The simulation notebook will automatically create timestamped output folders under simulated_data/.

The PC algorithm notebook will:

- Load simulated data
- Run the PC algorithm
- Visualize the inferred CPDAG
- Compute Structural Hamming Distance (SHD)
- Compare inferred graphs against the true simulated structure

![Figure 2: DAG comparison for simple linear gaussian dataset](figure/DAG.png)

---

## 4. Exiting the Container

To exit the running container:

```bash
exit
```

---

# Additional Notes

- This repository currently includes only the dependencies required for Quarter 1 deliverables.
- Full LaTeX, GPU, and advanced Causal Copilot tooling will be added in future project phases.

---

## Dependencies & Versions Installed in the Docker Image

The environment is built on:

**Base image:**
- `python:3.12-slim`

**System packages:**
- `graphviz` (required for PC algorithm graph visualization)

**Python libraries (as pinned in requirements_notebooks.txt):**
- `numpy==1.26.4`
- `pandas==1.5.3`
- `matplotlib==3.9.2`
- `scipy==1.13.1`
- `scikit-learn==1.5.2`
- `networkx==3.2.1`
- `python-igraph==0.11.8`
- `texttable==1.7.0` (igraph dependency)
- `causal-learn==0.1.3.9`
- `pydot==3.0.2` (GraphViz wrapper)
- `typing-extensions==4.12.2`
- `tqdm==4.66.5`

These packages are the **only** dependencies required to run the Quarter 1 deliverables.  

---

## Acknowledgments

Portions of the data simulation pipeline used in this project are adapted from the open-source implementation provided by the **Causal Copilot** research team.

We acknowledge and thank the authors of the following work:

**Causal-Copilot: Autonomous Causal Analysis Agent**  
*Xinyue Wang, Kun Zhou, Wenyi Wu, Har Simrat Singh, Fang Nan, Songyao Jin, Aryan Philip, Saloni Patnaik, Hou Zhu, Shivam Singh, Parjanya Prashant, Qian Shen, Biwei Huang*  
(2024)

Their publicly released codebase supplied the foundations for our synthetic data generation module, including configurable structural equation models, graph sampling utilities, and noise distribution functions.  
Our project extends these components for course-specific experimentation and analysis.

We gratefully recognize their contributions to open causal inference research.
