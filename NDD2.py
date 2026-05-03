import numpy as np
from gurobipy import Model, GRB, QuadExpr
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
# ============================================================================
# 1. Define System Parameters and Polytope
# ============================================================================

# --- Load the uncertain system parameters from V.csv ---
# V is a matrix where each row is a different possible set of system parameters.
try:
    V = np.loadtxt("V.csv", delimiter=",")
    print(f"Successfully loaded {V.shape[0]} system parameter sets from V.csv.")
except FileNotFoundError:
    print("Error: V.csv not found. Using a default single parameter set.")
    # This is a fallback to the single deterministic system from the paper.
    # The columns are a1, a2, a3, a4, a5, a6, b1, b2
    V = np.array([[1, 0.01, 0.001, 0.01, 1, -0.004, 0.009, 0.009]])

u_bound = 2

# --- Define the Candidate Polytope (Omega) ---
# For a robust problem with high uncertainty, the initial guess must be
# small to ensure the problem is feasible.
scaling = 1 # Drastically reduced scaling to handle high uncertainty
print(f"\nUsing initial polytope scaling factor: {scaling}")

# --- Define the Candidate Polytope (Omega) ---

# Original size
# scaling = 0.1 

# --- Define the Candidate Polytope (Omega) ---

# Set the overall scaling for the shape
scaling = 0.15

# Define factors to make the shape "fat"
x_stretch_factor = 1.5  # Make it 50% wider
y_compress_factor = 1.5 # Make it 40% shorter

print(f"Using x-stretch: {x_stretch_factor}, y-compress: {y_compress_factor}")

# Base vertices
base_Poly_v = np.array([
    [1.125, -1.125], [0.75, 0.], [0., 0.75],
    [-1.125, 1.125], [-0.75, 0.], [0., -0.75]
])

# Create the new, fat Poly_v
Poly_v = base_Poly_v.copy()
Poly_v[:, 0] *= x_stretch_factor   # Stretch x-coordinates
Poly_v[:, 1] *= y_compress_factor  # Compress y-coordinates
Poly_v *= scaling

# Base H-representation
base_H = np.array([
    [4/3, 4/9], [4/3, 4/3], [4/9, 4/3],
    [-4/3, -4/9], [-4/3, -4/3], [-4/9, -4/3]
])

# Create the new, corresponding H
H = base_H.copy()
H[:, 0] /= x_stretch_factor  # Apply inverse stretch to x
H[:, 1] /= y_compress_factor # Apply inverse compress to y
H /= scaling


# --- Hyperparameters ---
# To compute the robust set (dashed line), set w_bounds = 0.4
w_bounds = 0.004
# The paper sets lambda = 1 to find an invariant set (not contractive).
lambda_w = 1.0

# ============================================================================
# 2. Helper Function to Build the F Expression
# ============================================================================

def build_F_gurobi(params, x1_expr, x2_expr, u_var, c_vector):
    """
    Builds the Gurobi expression for F(x, u, c).
    This function implements the DC decomposition of the nonlinear system dynamics.
    f(x,u) = g(x,u) - h(x,u)
    """
    a1, a2, a3, a4, a5, a6, b1, b2 = params
    c1, c2 = c_vector

    # Linear parts of the dynamics
    linear_part1 = a1*x1_expr + a2*x2_expr + b1*u_var
    linear_part2 = a4*x1_expr + a5*x2_expr + b2*u_var

# --- DC decomposition for f1 ---
    if a3 >= 0:
        g1 = linear_part1 + (abs(a3)/4)*(x1_expr + u_var)**2
        h1 = (abs(a3)/4)*(x1_expr - u_var)**2
    else:
        g1 = linear_part1 + (abs(a3)/4)*(x1_expr - u_var)**2
        h1 = (abs(a3)/4)*(x1_expr + u_var)**2

    # --- DC decomposition for f2 (note the original term is -a6*x2*u) ---
    if a6 >= 0:
        g2 = linear_part2 + (abs(a6)/4)*(x2_expr - u_var)**2
        h2 = (abs(a6)/4)*(x2_expr + u_var)**2
    else:
        g2 = linear_part2 + (abs(a6)/4)*(x2_expr + u_var)**2
        h2 = (abs(a6)/4)*(x2_expr - u_var)**2
    
    # Linearizations of g and h at (0,0)
    g1_L, h1_L = linear_part1, 0
    g2_L, h2_L = linear_part2, 0

    # Build the convex function F(x,u,c)
    F_expr = QuadExpr()
    if c1 >= 0: F_expr += c1 * (g1 - h1_L)
    else: F_expr += c1 * (g1_L - h1)
    if c2 >= 0: F_expr += c2 * (g2 - h2_L)
    else: F_expr += c2 * (g2_L - h2)
    return F_expr

# ============================================================================
# 3. Main Optimization Loop (Iterative Method)
# ============================================================================

print("\n--- Starting Iterative Optimization Process ---")
all_system_alphas = []

# *** NEW OUTERMOST LOOP ***
# Iterate through each parameter set in V
for k, current_params in enumerate(V):
    print(f"\n--- Solving for System {k+1}/{len(V)} ---")
    gamma_results_for_this_system = []

    # For each vertex `v_p` of the candidate polytope `Poly_v`
    for p, v_p in enumerate(Poly_v):
        # Create a new Gurobi model for this subproblem
        model = Model(f"subproblem_k{k}_v{p}")
        model.setParam('OutputFlag', 0)
        model.setParam('NonConvex', 2)

        # Decision variables
        gamma = model.addVar(name="gamma", lb=0.0)
        u = model.addVar(name="u", lb=-u_bound, ub=u_bound)
        model.setObjective(gamma, GRB.MAXIMIZE)

        x1_expr = gamma * v_p[0]
        x2_expr = gamma * v_p[1]

        # For each face `h_j` of the candidate polytope `H`
        for j, h_j in enumerate(H):
            # Build F using only the current system's parameters
            F = build_F_gurobi(current_params, x1_expr, x2_expr, u, h_j)
            phi_W = w_bounds * np.linalg.norm(h_j, ord=1)
            model.addConstr(F <= lambda_w * gamma - phi_W, name=f"c_{p}_{j}")

        # Solve the optimization for this vertex
        model.optimize()

        if model.status == GRB.OPTIMAL:
            gamma_results_for_this_system.append(gamma.X)
        else:
            print(f"  Subproblem for v_p={p} failed. This system may be unstable.")
            gamma_results_for_this_system.append(0) # Assign worst-case alpha
            break

    # Find the alpha for the current system
    if gamma_results_for_this_system and len(gamma_results_for_this_system) == len(Poly_v):
        alpha_for_this_system = min(gamma_results_for_this_system)
        all_system_alphas.append(alpha_for_this_system)
        print(f"  Alpha for this system: {alpha_for_this_system:.4f}")
    
    # Optional: Stop early if a zero-alpha is found, as it's the worst case
    if all_system_alphas and all_system_alphas[-1] < 1e-6:
        print("  Found a system with zero-size invariant set. The final robust set will be zero.")
        # break # Uncomment to stop early and save time

# ============================================================================
# 4. Final Result and Visualization
# ============================================================================

# ============================================================================
# 4. Final Result and Visualization
# ============================================================================

if all_system_alphas:
    # The final robust alpha is the minimum of all individual system alphas
    final_robust_alpha = min(all_system_alphas)*1
    
    print("\n--- Optimization Complete ---")
    print(f"Final robust scaling factor alpha = min(all alphas) = {final_robust_alpha:.4f}")

    # --- Create the Combined Plot ---
    plt.figure(figsize=(9, 9))
    ax = plt.gca()

    # 1. Plot the State Constraint Set (a box from -4 to 4)
    # Define vertices for a closed box
    constraint_box = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4], [-4, -4]])
    plt.plot(constraint_box[:, 0], constraint_box[:, 1], color='gray', linestyle='-', label='State Constraints (|x| <= 4)')
    plt.fill(constraint_box[:, 0], constraint_box[:, 1], 'gray', alpha=0.15)

    # 2. Plot the Initial Polytope Shape (Omega)
    hull_initial = ConvexHull(Poly_v)
    # The vertices from ConvexHull need to be closed for a clean plot line
    initial_verts_ordered = Poly_v[hull_initial.vertices]
    initial_verts_closed = np.vstack([initial_verts_ordered, initial_verts_ordered[0]]) # Append first vertex to the end
    plt.plot(initial_verts_closed[:, 0], initial_verts_closed[:, 1], 'k--', label='Initial Shape (Omega)')

    # 3. Plot the Final Robust Invariant Set (alpha * Omega)
    if final_robust_alpha > 1e-6:
        final_set_vertices = final_robust_alpha * Poly_v
        hull_final = ConvexHull(final_set_vertices)
        
        # Plot the filled region
        plt.fill(final_set_vertices[hull_final.vertices, 0], final_set_vertices[hull_final.vertices, 1], 'b', alpha=0.3, label=f'Robust Invariant Set (alpha={final_robust_alpha:.3f})')
        
        # Plot the outline
        final_verts_ordered = final_set_vertices[hull_final.vertices]
        final_verts_closed = np.vstack([final_verts_ordered, final_verts_ordered[0]])
        plt.plot(final_verts_closed[:, 0], final_verts_closed[:, 1], 'b-')
    else:
        # If the set is just the origin, plot a single point
        plt.plot(0, 0, 'bo', markersize=8, label='Robust Invariant Set (alpha=0.0)')
        print("The only robust invariant set found is the origin (a set of size zero).")
        print("Try reducing the w_bounds or using a different initial polytope shape.")

    # --- Final Plotting Adjustments ---
    plt.title('Robust Invariant Set and Constraints')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    
    # Set axis limits to fully contain the state constraints
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    
    plt.show()

else:
    print("\n--- Optimization Failed ---")
    print("Could not find a valid invariant set for any of the provided systems.")

# The old plotting block for the initial polytope is no longer needed
# as it is now part of the combined plot.
# --- Prepare and Save Data to Poly.csv ---
print("\nSaving plotting data to Poly.csv...")

# 1. Constraint Box data
constraint_box = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4], [-4, -4]])
df_constraint = pd.DataFrame(constraint_box, columns=['x', 'y'])
df_constraint['type'] = 'constraint'

# 2. Initial Shape data
hull_initial = ConvexHull(Poly_v)
initial_verts_ordered = Poly_v[hull_initial.vertices]
initial_verts_closed = np.vstack([initial_verts_ordered, initial_verts_ordered[0]])
df_initial = pd.DataFrame(initial_verts_closed, columns=['x', 'y'])
df_initial['type'] = 'initial'

# 3. Final Shape data
df_final = pd.DataFrame() # Initialize empty dataframe
if final_robust_alpha > 1e-6:
    final_set_vertices = final_robust_alpha * Poly_v
    hull_final = ConvexHull(final_set_vertices)
    final_verts_ordered = final_set_vertices[hull_final.vertices]
    final_verts_closed = np.vstack([final_verts_ordered, final_verts_ordered[0]])
    df_final = pd.DataFrame(final_verts_closed, columns=['x', 'y'])
    df_final['type'] = 'final'
    df_final['alpha'] = final_robust_alpha # Store alpha value
else:
    # Handle case where the set is just the origin
    df_final = pd.DataFrame({'x': [0], 'y': [0], 'type': ['final_point'], 'alpha': [0]})
    
# Combine all dataframes and save to CSV
combined_df = pd.concat([df_constraint, df_initial, df_final], ignore_index=True)
combined_df.to_csv("Poly.csv", index=False)
print("Data saved successfully.")