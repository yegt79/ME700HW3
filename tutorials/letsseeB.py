from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import pre_process_demo_helper_fcns as pre_demo
from finiteelementanalysis.solver import hyperelastic_solver
from finiteelementanalysis import visualize as viz
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# for saving files later
tutorials_dir = Path(__file__).parent

# --- Beam geometry and base parameters ---
L = 30   # length in x
H = 1    # height in y
q = -0.2  # uniform downward load per unit length
E = 50000.0
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
kappa = E / (3.0 * (1.0 - 2.0 * nu))
material_props = np.array([mu, kappa])

# Analytical solution for comparison
E_eff = E / (1 - nu ** 2.0)  # Plane strain adjustment
I = H ** 3 / 12.0
w_analytical = q * L ** 4 / (8.0 * E_eff * I)

# Function to run simulation and return tip deflection
def run_simulation(ele_type, nx, ny, label):
    ndof = 2  # 2 DOFs per node (x, y)
    
    # Generate mesh
    coords, connect = pre.generate_rect_mesh_2d(ele_type, 0.0, 0.0, L, H, nx, ny)
    
    # Identify boundaries
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(
        coords, connect, ele_type, x_lower=0.0, x_upper=L, y_lower=0.0, y_upper=H
    )
    
    # Boundary conditions
    fixed_left = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0.0, 0.0)
    dload_info = pre.assign_uniform_load_rect(boundary_edges, "top", 0.0, q)
    fixed_nodes = fixed_left
    
    # Solve
    displacements_all, nr_info_all = hyperelastic_solver(
        material_props,
        ele_type,
        coords.T,
        connect.T,
        fixed_nodes,
        dload_info,
        nr_print=False,  # Suppress printing for cleaner output
        nr_num_steps=1,
        nr_tol=1e-10,
        nr_maxit=30,
    )
    
    final_disp = displacements_all[-1]
    
    # Find tip node (x=L, y=H/2)
    tip_node = None
    tol = 1e-3
    for i, (x, y) in enumerate(coords):
        if abs(x - L) < tol and abs(y - H/2) < H/(2*ny):
            tip_node = i
            break
    if tip_node is None:
        raise ValueError(f"Could not find tip node near x=L, y=H/2 for {label}.")
    
    tip_disp_y = final_disp[ndof*tip_node + 1]
    
    # Optional: Generate deformation GIF
    img_name = f"deformation_{label}.gif"
    fname = str(tutorials_dir / img_name)
    viz.make_deformation_gif(displacements_all, coords, connect, ele_type, fname)
    
    return tip_disp_y

# --- Define refinement cases ---
cases = [
    {"label": "Original", "ele_type": "D2_nn4_quad", "nx": 40, "ny": 2},
    {"label": "H-Refinement", "ele_type": "D2_nn4_quad", "nx": 80, "ny": 4},  # Double elements
    {"label": "P-Refinement", "ele_type": "D2_nn8_quad", "nx": 40, "ny": 2},  # Higher-order elements
]

# --- Run simulations and collect results ---
results = {}
for case in cases:
    tip_deflection = run_simulation(case["ele_type"], case["nx"], case["ny"], case["label"])
    results[case["label"]] = tip_deflection

# --- Print comparison ---
print("\n=== Tip Deflection Comparison ===")
print(f"Analytical Euler-Bernoulli deflection: {w_analytical:.6f}")
for label, tip_disp_y in results.items():
    error = abs(tip_disp_y - w_analytical)
    print(f"{label}:")
    print(f"  Computed tip deflection (y): {tip_disp_y:.6f}")
    print(f"  Absolute error: {error:.6e}")

aa = 44
