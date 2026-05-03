#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:39:41 2025

@author: sahand
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.linalg import block_diag
import pandas as pd
from gurobipy import Model, GRB, QuadExpr, LinExpr


#%% Hyperparameters
T = 0.01
t_total = 0.3
num_steps = int(t_total / T)


u_min = 2 
u_max = 1
w_bounds = 0.002

m = 1
n = 2
L =3

mu=0.9

vector = np.array([1,T,T*(1-mu),T,1,-4*T*(1-mu),T*mu,T*mu]).reshape(8, 1)

#Randomness
np.random.seed(12)
#%% Exciting Inputs with Random Values between -0.2 and 0.2
def generate_inputs(num_steps):
    time_points = np.linspace(0, 1, num_steps)  # Generate time points
    input_signals = np.random.uniform(-u_min, u_max, (m, num_steps))  # Generate random inputs between -0.2 and 0.2
    
    return input_signals, time_points

# Generate input signals
input_signals, time_points = generate_inputs(num_steps)


#%% Initial Conditions for the State Variables
def generate_initial_conditions():
    # Generating random initial conditions for x1 and x2 (2 state variables)
    initial_conditions = np.random.rand(n)  # Random values between 0 and 1 for x1 and x2
    return initial_conditions

initial_conditions = generate_initial_conditions()  # Initial state variables

#%% Simulate the System
# State update equations:

x1_n = np.zeros((num_steps + 1))
x2_n = np.zeros((num_steps + 1))

# Set the initial conditions at the first element (index 0)
x1_n[0] = initial_conditions[0]
x2_n[0] = initial_conditions[1]

# Simulate the system for num_steps iterations
for k in range(num_steps):  # This will loop from k = 0 to 29
    u_k = input_signals[0, k]  # Input at time step k
    w = np.random.uniform(-w_bounds, w_bounds, (n, 1))

    # Update the NEXT state (k+1) using the CURRENT state (k)
    x1_n[k+1] = x1_n[k] + T * x2_n[k] + T * mu * u_k + T * (1 - mu) * x1_n[k] * u_k + T * w[0, 0]
    x2_n[k+1] = x2_n[k] + T * x1_n[k] + T * mu * u_k - 4 * T * (1 - mu) * x2_n[k] * u_k + T * w[1, 0]


# Stack the results. x_n will have a shape of (2, 31)
x_n = np.vstack((x1_n, x2_n))

#%% Build the Matrices

def Build(x_states, u_inputs):
    """
    Constructs the matrices M and N for system identification.

    - M is built from states and inputs at time k.
    - N is built from states at time k+1.

    Args:
        x_states (np.ndarray): The state history, shape (n, num_steps + 1).
        u_inputs (np.ndarray): The input history, shape (m, num_steps).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the final matrices (M, N).
    """
    Ms_list = []
    Ns_list = []
    # The number of iterations will be 30 (the length of the input signal)
    num_iterations = u_inputs.shape[1]

    # Loop from k = 0 to 29
    for k in range(num_iterations):
        # --- Part 1: Build Ms_k using data from time k ---
        x1_k = x_states[0, k]
        x2_k = x_states[1, k]
        u_k = u_inputs[0, k]

        # Define the two unique rows of the Ms_k matrix
        row1 = [x1_k, x2_k, x1_k * u_k, 0,    0,    0,          u_k, 0]
        row2 = [0,    0,    0,          x1_k, x2_k, x2_k * u_k, 0,   u_k]

        Ms_k = np.array([
            row1,
            row2,
            [-x for x in row1],
            [-x for x in row2]
        ])
        Ms_list.append(Ms_k)

        # --- Part 2: Build Ns_k using data from time k+1 ---
        x1_k_plus_1 = x_states[0, k + 1]
        x2_k_plus_1 = x_states[1, k + 1]

        # Ns_k is a 4x1 column vector
        Ns_k = np.array([
            [x1_k_plus_1],
            [x2_k_plus_1],
            [-x1_k_plus_1],
            [-x2_k_plus_1]
        ])
        Ns_list.append(Ns_k)

    # Vertically stack all the small matrices into the final M and N
    M = np.vstack(Ms_list)
    N = np.vstack(Ns_list)
    
    # Add the Upper Bound of the Noise
    noise_effect_bound = T * w_bounds
    N_bound = np.full_like(N, noise_effect_bound)
    
    N = N + N_bound

    return M, N

# Assuming x_n and input_signals are already defined...

# Build both the M and N matrices in one call
M, N = Build(x_n, input_signals)

#%% Read Vertex Matrix from CSV
try:
    # Define the filename
    v_filename = "V.csv"
    
    # Load the data from the CSV file into a NumPy array
    V = np.loadtxt(v_filename, delimiter=",")
    
    print(f"\nSuccessfully loaded {v_filename} into a NumPy array.")
    print(f"Shape of the vertex matrix V: {V.shape}")

except FileNotFoundError:
    print(f"\nError: The file '{v_filename}' was not found.")
    print("Please ensure you have run the MATLAB script to generate it first.")
except Exception as e:
    print(f"An error occurred while reading {v_filename}: {e}")
    
    # Assuming M, N, and V are already defined in your script...
#%% Save Matrices for Matlab

# --- Save M and N to CSV files ---

# Save the matrix M to M.csv
np.savetxt("M.csv", M, delimiter=",")

# Save the vector N to N.csv
np.savetxt("N.csv", N, delimiter=",")

print("\nSuccessfully saved M.csv and N.csv to your current working directory.")
#%% Plot Results

# First, ensure your time array matches the state arrays' length
time = np.linspace(0, t_total, num_steps + 1)

plt.figure(figsize=(10, 5))

# Plot for x1
plt.subplot(2, 1, 1)
# The lengths of time (31) and x1_n (31) now match
plt.plot(time, x1_n, label='x1', color='b')
plt.title('State x1 over Time')
plt.xlabel('Time [s]')
plt.ylabel('x1')
plt.grid(True)

# Plot for x2
plt.subplot(2, 1, 2)
# The lengths of time (31) and x2_n (31) now match
plt.plot(time, x2_n, label='x2', color='r')
plt.title('State x2 over Time')
plt.xlabel('Time [s]')
plt.ylabel('x2')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()