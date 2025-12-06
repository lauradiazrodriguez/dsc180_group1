# DSC180A Quarter 1 Project - Data Simulation (Causal Copilot)

This project replicates and extends the **Data Simulation and Experimentation** module from the *Causal Copilot* framework.
The goal is to reproduce the automated process for generating **synthetic causal datasets** using structural equation models with configurable graph structures, noise distributions, and function types.
By adjusting parameters such as noise type, graph density, and nonlinearity, this project demonstrates how different data conditions affect the causal relationships within the generated datasets.

## Running the Simulation

This project's main notebooks are:

generating_simulated_data.ipynb
PC_alg.ipynb


You can run it through Jupyter Notebook.
Then, execute each cell to generate synthetic datasets.
Each run will automatically create a folder under:
```
simulated_data
```

Example output:
```
simulated_data
└── 20251030_231358_LinearGaussian_d5_n1000_nodes5_samples1000
      ├── LinearGaussian_d5_n1000_config.json
      ├──LinearGaussian_d5_n1000_data.csv
      └──LinearGaussian_d5_n1000_graph.npy
```
