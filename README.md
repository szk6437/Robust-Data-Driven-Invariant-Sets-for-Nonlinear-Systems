# Robust Data-Driven Invariant Sets for Nonlinear Systems

Please check out the paper for more details:
Robust Data-Driven Invariant Sets for Nonlinear Systems

## Prerequisites
* **Python 3.x** (`numpy`, `scipy`, `pandas`, `matplotlib`, `gurobipy`)
* **MATLAB** (Tested with R2021a or newer)
* **Gurobi Optimizer** (A valid license is required for solving the QCQP problems).

## Execution Pipeline
The scripts must be executed in the following order to correctly pass data between the simulation, vertex enumeration, and synthesis phases:

1. **`NDD.py` (Python):** Simulates the system, generates input signals, and builds the inequality matrices (`M.csv`, `N.csv`) bounding the parameter uncertainty.
2. **`Matrices.m` (MATLAB):** Reads `M.csv` and `N.csv`, utilizes `con2vert.m` to perform vertex enumeration, and exports the vertices to `V.csv`.
3. **`NDD2.py` (Python):** Loads `V.csv` and computes the maximum scaling factor for the candidate polytope using DC decomposition. Outputs the final RPI set geometry to `Poly.csv`.

## Directory Structure
* `data/`: Generated constraint matrices (`M.csv`, `N.csv`) and vertex representations (`V.csv`).
* `images/`: Generated plots illustrating the initial constraints and final robust invariant sets.

## Acknowledgments & Licensing
The primary Python and MATLAB synthesis scripts in this repository are licensed under the MIT License. 

**Third-Party Code:**
The file `con2vert.m` is a MATLAB File Exchange utility used for vertex enumeration. It was originally authored by **Michael Kleder** (2005) and updated by **Stephen Becker** (2021). It is included here under the standard MathWorks File Exchange default BSD License.
