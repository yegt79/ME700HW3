# Finite Element Analysis Tutorial: `tutorial_discretization.ipynb`

The project focuses on finite element analysis, and I’m starting with the `tutorial_discretization.ipynb` file. I am beginning the analysis from the entry point of this file and using the debug tool to dive into more detail and understand what this file is calling.

## Initial Setup and Visualization of Gauss Points

The script imports helper functions from a custom `finiteelementanalysis` package, specifically `discretization_demo_helper_fcns` (aliased as `di_demo`). It then sets up a 2D, 4-node quadrilateral element (`D2_nn4_quad`) with 4 Gauss points for numerical integration and calls `visualize_gauss_pts` to generate and save a plot of these points to `D2_nn4_quad_4gp.png`.

### 🔍 Stepping into `visualize_gauss_pts`

This function, defined in the `discretization_demo_helper_fcns` module, visualizes Gauss points. It retrieves points via `gauss_pts_and_weights`, defines reference nodes and edges for `ele_type`, and raises a `ValueError` for invalid types. It plots edges in black, nodes as blue dots ($N_1$ to $N_8$), and Gauss points as red dots ($G_1$ to $G_4$) using Matplotlib, saving to `fname` at 300 DPI.

**Improvement Idea**: Enhance input validation to check `num_pts` for non-numeric inputs.

## Interpolating a Scalar Field in Natural Coordinates

🔙 Stepping out to the main script:  
The script defines a bilinear field `fcn` in natural coordinates ($\xi$, $\eta$), computes `node_values` as $[1, 5, 7, 3]$ for `"D2_nn4_quad"`, and calls `di_demo.plot_interpolate_field_natural_coords_single_element`, saving to `D2_nn4_quad_interpolate_fcn.png`.

### 🔍 Stepping into `plot_interpolate_field_natural_coords_single_element`

This function plots a scalar field in natural coordinates. For `"D2_nn4_quad"`, it creates a $10 \times 10$ grid in $\xi$, $\eta$, interpolates values using `interpolate_field_natural_coords_single_element`, and scatter-plots with Matplotlib.

## Mapping to Physical Coordinates

🔙 Stepping out to the main script:  
The script defines `fcn` in physical coordinates ($x$, $y$), sets `node_coords` for `"D2_nn4_quad"`, computes `node_values` as $[6.5, 8, 13.5, 6]$, and calls `visualize_isoparametric_mapping_single_element`, saving to `D2_nn4_quad_interpolate_fcn_physical_coords.png`.

**Improvement Idea**: Centralize `node_coords` definitions to reduce redundancy.

### 🔍 Stepping into `visualize_isoparametric_mapping_single_element`

This function visualizes isoparametric mapping. For `"D2_nn4_quad"`, it samples a $20 \times 20$ grid in $\xi$, $\eta$, interpolates using `interpolate_field_natural_coords_single_element`, and plots natural and physical subplots with coolwarm scatter points.

## Gradient Transformation

🔙 Stepping out to the main script:  
The script defines a bilinear field with derivatives ($\partial f / \partial x$, $\partial f / \partial y$), sets `node_values` as $[0, 4, 10, 6]$ for `"D2_nn4_quad"`, computes analytical ($[3.5, 4.5]$) and numerical gradients at $\xi, \eta = 0, 0$, and transforms them to physical coordinates.

### 🔍 Stepping into `transform_gradient_to_physical`

This function transforms gradients from $\xi$, $\eta$ to $x$, $y$. It computes the Jacobian ($J$), inverts it, and multiplies $J_{inv}^T$ with `gradient_natural` from `interpolate_gradient_natural_coords_single_element`.

**Improvement Idea**: Add `try-except` around `np.linalg.inv(J)` for singular Jacobians.

## Computing Integrals

🔙 Stepping out to the main script:  
The script uses `element_area` for the quadrilateral area and `integral_of_deriv` for derivative integrals.

### 🔍 Stepping into `compute_integral_of_derivative`

This function computes gradients in natural coordinates, transforms them to physical coordinates using `transform_gradient_to_physical`, and sums weighted contributions with $det_J$.

# Finite Element Analysis Tutorial: `tutorial_sparse_solver.ipynb`

The script imports modules from `finiteelementanalysis` and sets up a hyperelastic problem with sparse solver testing.

## Defining the Geometry: `define_sample_problem_geom`

The script calls `define_sample_problem_geom`, which invokes `pre.generate_rect_mesh_2d` to assign `coords` and `connect`.

### 🔍 Stepping into `generate_rect_mesh_2d`

This function creates a 2D rectangular mesh. It dispatches to a helper based on `ele_type` (e.g., `"D2_nn8_quad"`) or raises a `ValueError` for unknown types.

## Defining Problem Information: `define_sample_problem_info`

The script calls `define_sample_problem_info`, which sets up boundaries, fixes the left boundary, applies a right-boundary load, defines material properties ($\mu = 10$, $\kappa = 100$), and creates an artificial displacement field ($u_x = 0.01 \cdot x$).

### 🔍 Stepping into `identify_rect_boundaries`

This function identifies boundary nodes and edges. It loops over nodes, assigning indices to sets (e.g., `left_nodes`) based on coordinate proximity to bounds within tolerance, and uses `local_faces_for_element_type`.

### 🔍 Stepping into `assign_fixed_nodes_rect`

This function sets boundary conditions. It builds constraints for non-`None` displacements (e.g., both DOFs to 0 for `"left"`) and returns a transposed array.

### 🔍 Stepping into `assign_uniform_load_rect`

This function assigns a uniform load (e.g., $q = 10.0$). It initializes `dload_info` with face data and traction components, returning the array.

## Mesh Visualization

🔙 Stepping out to the main script:  
The script calls `pre_demo.plot_mesh_2D`, saving to `"solver_mesh_1.png"`.

### 🔍 Stepping into `plot_mesh_2D`

This function plots the mesh. For `"D2_nn8_quad"`, it draws edges in gray (e.g., $(0,4)$ to $(7,0)$) and skips optional Gauss points here.

## Assembly Timing

🔙 Stepping out to the main script:  
The script tests assembly times with `num_runs = 5`.

### 🔍 Stepping into `time_assemble_global_stiffness`

This function times stiffness matrix assembly using `time_function_call` with `assemble.global_stiffness`.

### 🔍 Stepping into `time_assemble_global_traction`

This function times traction vector assembly using `time_function_call` with `assemble.global_traction`.

### 🔍 Stepping into `time_assemble_global_residual`

This function times residual vector assembly using `time_function_call` with `assemble.global_residual`.

## Matrix Solve Preparation and Timing

🔙 Stepping out to the main script:  
The script prepares $K$ and $R$ with `prep_for_matrix_solve` and times a dense solve.

### 🔍 Stepping into `prep_for_matrix_solve`

This function assembles $K$, $F$, and $R$ using `assemble` functions, applies fixed boundary conditions, and returns $K$ and $R$.

### 🔍 Stepping into `time_one_matrix_solve`

This function times a solve. For `"dense"`, it uses `np.linalg.solve` via `time_function_call`.

## Sparsity Analysis

🔙 Stepping out to the main script:  
The script visualizes $K$’s sparsity with `analyze_and_visualize_matrix`, saving to `"solver_global_stiffness_1.png"`.

### 🔍 Stepping into `analyze_and_visualize_matrix`

This function computes condition number, sparsity, and bandwidth, plotting the sparsity pattern with `imshow`.

## Sparse Assembly and Solvers

🔙 Stepping out to the main script:  
The script tests sparse assembly and solvers.

### 🔍 Stepping into `time_assemble_global_stiffness_sparse`

This function times sparse stiffness assembly with `assemble.global_stiffness_sparse` via `time_function_call`.

### 🔍 Stepping into `time_one_matrix_solve` (Sparse Variants)

For `"sparse"`, it uses `spla.spsolve`; for `"sparse_iterative"`, it uses `spla.gmres` with optional ILU preconditioning, both via `time_function_call`.

## Hyperelastic Solution and Visualization

🔙 Stepping out to the main script:  
The script runs `hyperelastic_solver` and creates a GIF with `make_deformation_gif`.

### 🔍 Stepping into `hyperelastic_solver`

This function solves a hyperelastic problem with Newton-Raphson, assembling $K$, $F$, and $R$ over `nr_num_steps`, and returns displacements and info.

### 🔍 Stepping into `make_deformation_gif`

This function animates deformation. It plots the undeformed mesh, updates lines and scatter with magnified displacements, and saves a GIF using `FuncAnimation`.
