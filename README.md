# EarthBench - Biodiversity Curve Optimization

## Introduction

This project focuses on optimizing biodiversity curves using an Evolutionary Algorithm (EA). We are specifically working with the `124, 278, 904, 934, 2538, 2582` dataset to find the optimal sequence combination through algorithmic means.

## Dependencies

This project relies on several Python libraries. Before starting, please ensure that you have installed all the necessary dependencies.

You can install all dependencies using the following `pip` command:

```bash
conda env create -f environment.yml
```

## How to Run

We provide a convenient script to run this project. This script will execute the Evolutionary Algorithm on the `124` dataset using predefined parameters.

Please execute the following command in the root directory of the project:

```bash
bash scripts/run_124.sh
```

<!-- ## Output Description

Once the script has run successfully, you can find the output results in the `data/earth_124/` directory.

The output file is a `.json` file, which includes the following information:

*   `x_best`: The optimal sequence found.
*   `y_best`: The score corresponding to the optimal sequence.
*   `epoch`: The iteration number at which the optimal solution was found.

Additionally, during the execution, the program saves intermediate results and checkpoints in the `data/earth_2574/tmp_best/` and `data/earth_2574/checkpoints/` directories. -->