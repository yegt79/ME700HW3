from finiteelementanalysis import pre_process as pre
from finiteelementanalysis import solver as sl
from finiteelementanalysis import discretization as di
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def refinement_example(ele_type, num_gauss_pts, nx, ny, stretch=5.0, material_props=np.array([1.0, 2.0]), nr_num_steps=5):
    # Generate mesh
    x_lower, y_lower = 0, 0
    x_upper, y_upper = 10, 10
    coords, connect = pre.generate_rect_mesh_2d(ele_type, x_lower, y_lower, x_upper, y_upper, nx, ny)
    
    # Identify boundaries
    boundary_nodes, boundary_edges = pre.identify_rect_boundaries(coords, connect, ele_type, 0, 10.0, 0, 10.0)
    
    # Apply boundary conditions
    fixed_left = pre.assign_fixed_nodes_rect(boundary_nodes, "left", 0.0, 0.0)
    fixed_right = pre.assign_fixed_nodes_rect(boundary_nodes, "right", stretch, 0.0)
    fixed_top_y = pre.assign_fixed_nodes_rect(boundary_nodes, "top", None, 0.0)
    fixed_bottom_y = pre.assign_fixed_nodes_rect(boundary_nodes, "bottom", None, 0.0)
    fixed_nodes = np.hstack((fixed_left, fixed_right, fixed_top_y, fixed_bottom_y))
    
    # No distributed load
    _, ndof, _ = di.element_info(ele_type)
    dload_info = np.empty((ndof + 2, 0))
    
    # Solve the hyperelastic problem
    try:
        displacements_all, _ = sl.hyperelastic_solver(
            material_props=material_props,
            ele_type=ele_type,
            coords=coords.T,
            connect=connect.T,
            fixed_nodes=fixed_nodes,
            dload_info=dload_info,
            nr_print=False,
            nr_num_steps=nr_num_steps,
            nr_tol=1e-9,
            nr_maxit=30,
            matrix_solve_sparse=True
        )
    except Exception as e:
        print(f"Solver failed to converge: {e}")
        return 0.0, coords.shape[0] * ndof  # Return dummy values if solver fails
    
    # Get the final displacement
    displacement = displacements_all[-1]
    
    # Reshape displacement to (n_nodes, ndof)
    displacement_reshaped = displacement.reshape(-1, ndof)
    
    # Compute QoI: maximum x-displacement
    max_ux = np.max(displacement_reshaped[:, 0])
    
    # Compute total dofs
    total_dofs = coords.shape[0] * ndof  # n_nodes * ndof
    
    return max_ux, total_dofs

# Part A: Large deformation example with h- and p-refinement
print("\n--- Part A: Large Deformation Example with h- and p-Refinement ---\n")

# h-refinement with 3-node triangular elements
ele_type = "D2_nn3_tri"
num_gauss_pts = 1
h_refine_dofs = []
h_refine_qoi = []

for val in [2, 4, 8, 16, 32, 64, 128]:
    nx = val
    ny = val
    max_ux, dofs = refinement_example(ele_type, num_gauss_pts, nx, ny, stretch=5.0)
    h_refine_dofs.append(dofs)
    h_refine_qoi.append(max_ux)
    print(f"Part A h-refinement: nx={nx}, ny={ny}, dofs={dofs}, max_ux={max_ux}")

# p-refinement with 6-node triangular elements
ele_type = "D2_nn6_tri"
num_gauss_pts = 1
p_refine_dofs = []
p_refine_qoi = []

for val in [2, 4, 8, 16, 32, 64]:
    nx = val
    ny = val
    max_ux, dofs = refinement_example(ele_type, num_gauss_pts, nx, ny, stretch=5.0)
    p_refine_dofs.append(dofs)
    p_refine_qoi.append(max_ux)
    print(f"Part A p-refinement: nx={nx}, ny={ny}, dofs={dofs}, max_ux={max_ux}")

# Plot QoI vs. dofs for Part A
plt.figure()
plt.semilogx(h_refine_dofs, h_refine_qoi, 'o-', label='h-refinement (D2_nn3_tri)')
plt.semilogx(p_refine_dofs, p_refine_qoi, 's-', label='p-refinement (D2_nn6_tri)')
plt.xlabel('Degrees of Freedom (dofs)')
plt.ylabel('Maximum x-displacement (max_ux)')
plt.title('Part A: Convergence Study - QoI vs. dofs')
plt.legend()
plt.grid(True)
plt.savefig('part_a_convergence_study.png')
plt.show()
