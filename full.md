# Finite Element Analysis Tutorial: `tutorial_discretization.ipynb`

The project focuses on finite element analysis, and I’m starting with the `tutorial_discretization.ipynb` file. I am beginning the analysis from the entry point of this file and using the debug tool to dive into more detail and understand what this file is calling.

## Initial Setup and Visualization of Gauss Points

The script imports helper functions from a custom `finiteelementanalysis` package, specifically `discretization_demo_helper_fcns` (aliased as `di_demo`). It then sets up a 2D, 4-node quadrilateral element (`D2_nn4_quad`) with 4 Gauss points for numerical integration and calls `visualize_gauss_pts` to generate and save a plot of these points to `D2_nn4_quad_4gp.png`.

### 🔍 Stepping into `visualize_gauss_pts`

This function, defined in the `discretization_demo_helper_fcns` module, is designed to visualize Gauss points. It accepts `fname` (output filename), `ele_type` (e.g., `"D2_nn3_tri"`, `"D2_nn6_tri"`, `"D2_nn4_quad"`, `"D2_nn8_quad"`), and `num_pts` (number of Gauss points). It retrieves Gauss points and weights via `gauss_pts_and_weights(ele_type, num_pts)`. It defines reference nodes based on `ele_type`: for triangles, it supports a 3-node linear triangle (e.g., $[1, 0]$, $[0, 1]$, $[0, 0]$) and a 6-node quadratic triangle with mid-edge nodes; for quadrilaterals, it covers a 4-node bilinear quad (e.g., $[-1, -1]$ to $[-1, 1]$) and an 8-node quadratic quad with mid-edge nodes. It defines edges to connect these nodes appropriately and raises a `ValueError` if `ele_type` is invalid. Using Matplotlib, it plots edges as black lines, nodes as labeled blue dots (e.g., $N_1$ to $N_8$), and Gauss points as labeled red dots (e.g., $G_1$ to $G_4$ for `num_pts = 4`). It titles the plot with the element type, uses natural coordinates ($\xi$, $\eta$), and saves it to `fname` at 300 DPI.

**Improvement Idea**: Input validation could be enhanced. The function raises a `ValueError` for unsupported `ele_type`, but it could also validate `num_pts` (e.g., reject non-numeric inputs).

## Interpolating a Scalar Field in Natural Coordinates

🔙 Stepping out to the main tutorial script:  
The script defines `fcn`, a bilinear scalar field over natural coordinates ($\xi$, $\eta$), likely a test case for interpolation. Reusing `"D2_nn4_quad"`, it defines the same 4-node quadrilateral’s reference nodes as before. The function `node_values` computes `fcn` at these corners, yielding $[1, 5, 7, 3]$. It then calls `di_demo.plot_interpolate_field_natural_coords_single_element`, saving a plot to `D2_nn4_quad_interpolate_fcn.png`.

### 🔍 Stepping into `plot_interpolate_field_natural_coords_single_element`

This function plots a scalar field interpolated across a single element in natural coordinates, saving it to `fname`. It accepts `fname` (output filename), `ele_type` (e.g., `"D2_nn4_quad"`), `node_values` (field values at nodes), and `num_interp_pts` (optional, default 10). For `"D2_nn4_quad"`, it creates a $10 \times 10$ grid of points from $-1$ to $1$ in $\xi$ and $\eta$, flattening them into `xi_filtered` and `eta_filtered`. It defines reference nodes consistently with `visualize_gauss_pts`. It calls `interpolate_field_natural_coords_single_element` to compute interpolated values across these points and uses Matplotlib to scatter-plot the results.

## Mapping to Physical Coordinates

🔙 Stepping out to the main tutorial script:  
The script defines `fcn`, a bilinear field in physical coordinates ($x$, $y$). For `"D2_nn4_quad"`, `node_coords` specifies a quadrilateral in physical space ($[1, 1]$ to $[0, 2]$), and `node_values` computes `fcn` at these nodes ($[6.5, 8, 13.5, 6]$). It calls `visualize_isoparametric_mapping_single_element`, saving a plot to `D2_nn4_quad_interpolate_fcn_physical_coords.png`, likely showing the field mapped from natural to physical coordinates.

**Improvement Idea**: Centralize `node_coords` definitions to avoid redundancy across demos.

### 🔍 Stepping into `visualize_isoparametric_mapping_single_element`

This function visualizes the isoparametric mapping from natural to physical coordinates, saving to `fname`. For `"D2_nn4_quad"`, it samples a $20 \times 20$ grid ($-1$ to $1$) in $\xi$, $\eta$. It interpolates the field and coordinates using `interpolate_field_natural_coords_single_element`. It plots two subplots: the reference element (natural coords) and the mapped element (physical coords), using coolwarm scatter points, labeled nodes, and colorbars.

## Gradient Transformation

🔙 Stepping out to the main tutorial script:  
The script defines a bilinear field and its analytical derivatives ($\partial f / \partial x$, $\partial f / \partial y$). For `"D2_nn4_quad"`, a physical quadrilateral is set ($[0, 0]$ to $[0, 2]$), with `node_values` as $[0, 4, 10, 6]$. At the element center ($\xi, \eta = 0, 0$), it maps to physical coordinates ($x, y = 1, 1$). It computes the analytical derivative ($[3.5, 4.5]$), a numerical gradient in natural coordinates, transforms it to physical coordinates, and checks if they match.

### 🔍 Stepping into `transform_gradient_to_physical`

This function transforms a gradient from natural ($\xi$, $\eta$) to physical ($x$, $y$) coordinates. It accepts `ele_type` (e.g., `"D2_nn4_quad"`), `node_coords` (physical nodes, $[0, 0]$ to $[0, 2]$), `xi_vals` and `eta_vals` (here, $[0]$ and $[0]$), and `gradient_natural` (a $2 \times 1$ array from `interpolate_gradient_natural_coords_single_element`). It checks if the field is scalar or vector, loops over each ($\xi$, $\eta$) pair (here, just $(0, 0)$), computes the Jacobian ($J$), inverts $J$, and multiplies its transpose with `gradient_natural` to get `gradient_physical`.

**Improvement Idea**: Add a `try-except` around `np.linalg.inv(J)` to catch singular Jacobians, raising a descriptive error.

## Computing Integrals

🔙 Stepping out to the main tutorial script:  
`element_area` calculates the quadrilateral’s area using the shoelace formula. `integral_of_deriv` computes the analytical integral of the derivatives over the element.

### 🔍 Stepping into `compute_integral_of_derivative`

This function computes the gradient in natural coordinates using `interpolate_gradient_natural_coords_single_element` and transforms it to physical coordinates with `transform_gradient_to_physical`. It calculates the Jacobian determinant ($det_J$) via `compute_jacobian` and sums the weighted gradient contribution: $weight \cdot gradient_physical \cdot det_J$.

# Finite Element Analysis Tutorial: `tutorial_sparse_solver.ipynb`

The script starts by importing several modules from a custom `finiteelementanalysis` package and sets up a hyperelastic problem with sparse solver testing.

## Defining the Geometry: `define_sample_problem_geom`

The first function defined is `define_sample_problem_geom`. Inside this function, it calls `pre.generate_rect_mesh_2d`, assigning the returned values to `coords` and `connect`.

### 🔍 Stepping into `generate_rect_mesh_2d`

The function `generate_rect_mesh_2d` is designed to create a 2D rectangular mesh based on a specified element type and domain dimensions. Inside the function, it checks the value of `ele_type` and dispatches to a helper function accordingly. If `ele_type` doesn’t match any supported types (e.g., `"D2_nn3_tri"`, `"D2_nn6_tri"`, `"D2_nn4_quad"`, `"D2_nn8_quad"`), it raises a `ValueError` with a message indicating an unknown element type.

## Defining Problem Information: `define_sample_problem_info`

This function does several things in sequence to set up the problem. First, it identifies the boundaries by calling `pre.identify_rect_boundaries`, storing the results in `boundary_nodes` and `boundary_edges`. Next, it fixes the left boundary by calling `pre.assign_fixed_nodes_rect`. After that, it assigns a distributed load on the right boundary by setting $q$ to $10.0$ and calling `pre.assign_uniform_load_rect`, storing the result in `dload_info`. Then, it defines material properties by setting $\mu$ to $10$ and $\kappa$ to $100$, creating a NumPy array `material_props` with these values. Following that, it creates an artificial displacement field by initializing `displacement` as a zero array with the same shape as `coords` using `np.zeros((coords.shape))` and looping over each node index $kk$ from $0$ to the number of nodes (`coords.shape[0]`), setting the x-component of `displacement[kk, 0]` to `coords[kk, 0] * 0.01`, while the y-component remains zero. Finally, the function returns `displacement`, `material_props`, `fixed_nodes`, and `dload_info`.

### 🔍 Stepping into `identify_rect_boundaries`

This function identifies boundary nodes and edges for a rectangular 2D mesh. It starts by determining the number of nodes ($n_nodes$) and elements ($n_elems$) from the shapes of `coords` and `connect`. It initializes four empty sets—`left_nodes`, `right_nodes`, `bottom_nodes`, and `top_nodes`—to store boundary node indices. The function then loops over all node indices from $0$ to $n_nodes - 1$. For each node, it extracts the `xval` and `yval` from `coords[nid]` and checks their proximity to the domain bounds using the tolerance `tol`. If the absolute difference between `xval` and `x_lower` is less than `tol`, it adds the node index to `left_nodes`. Similarly, it checks `xval` against `x_upper` for `right_nodes`, `yval` against `y_lower` for `bottom_nodes`, and `yval` against `y_upper` for `top_nodes`. Next, it calls `local_faces_for_element_type`, which has been explained previously.

### 🔍 Stepping into `assign_fixed_nodes_rect`

The function `assign_fixed_nodes_rect` sets up prescribed boundary conditions for nodes on a specified boundary. It retrieves node indices for the specified boundary, returning an empty array if none exist. It builds a list of constraints by adding `(node_id, dof, value)` tuples for non-`None` displacements, returning an empty array if no constraints are added. Otherwise, it converts the list to a transposed NumPy array. For the call with `"left"`, it fixes both DOFs to zero, returning this array.

### 🔍 Stepping into `assign_uniform_load_rect`

The function `assign_uniform_load_rect` defines a uniform distributed load on boundary faces. It gets the face list for the boundary, returning an empty array if none exist. It initializes `dload_info` with shape $(5, n_face_loads)$, sets `elem_id` and `face_id` for each face, and assigns traction components from a load list. For the call with $q = 10.0$, it sets the x-direction load accordingly.

## Mesh Visualization

🔙 Stepping out to the main script:  
The script starts by defining variables for element type, mesh subdivisions, and domain size. It calls `define_sample_problem_geom`, which invokes `pre.generate_rect_mesh_2d` to create `coords` and `connect` for a 2D mesh. Stepping back, it then calls `define_sample_problem_info`, which uses `pre.identify_rect_boundaries` to find boundaries, `pre.assign_fixed_nodes_rect` to fix one boundary, and `pre.assign_uniform_load_rect` to apply a load on another, while also setting material properties and an artificial displacement field, returning these as variables. It then calls `pre_demo.plot_mesh_2D`, saving to `"solver_mesh_1.png"`.

### 🔍 Stepping into `plot_mesh_2D`

The function `pre_demo.plot_mesh_2D` creates a Matplotlib figure and axis. It loops over `connect`, plotting `"D2_nn8_quad"` edges in gray using predefined pairs like $(0,4)$ to $(7,0)$. If `gauss_points` were given, it’d plot them, but it’s `None` here. It labels elements and nodes, saving the plot.

## Assembly Timing

🔙 Stepping out to the main script:  
The script now tests assembly times for finite element computations. It sets `num_runs` to $5$ for repeated timing. It calls `solver_demo.time_assemble_global_stiffness` with `num_runs`, `ele_type`, transposed `coords` and `connect`, `material_props`, and transposed `displacement`, storing the average time in `avg_time_global_stiffness`.

### 🔍 Stepping into `time_assemble_global_stiffness`

The function `time_assemble_global_stiffness` measures the average time to assemble the global stiffness matrix. It calls `time_function_call` with `assemble.global_stiffness`, passing `ele_type`, `coords`, `connect`, `material_props`, `displacement`, and `num_runs`, returning the average time in seconds.

### 🔍 Stepping into `time_assemble_global_traction`

The function `time_assemble_global_traction` times the global traction vector assembly. It invokes `time_function_call` with `assemble.global_traction`, using `ele_type`, `coords`, `connect`, `dload_info`, and `num_runs`, returning the average execution time in seconds.

### 🔍 Stepping into `time_assemble_global_residual`

The function `time_assemble_global_residual` calculates the average time for global residual vector assembly. It uses `time_function_call` with `assemble.global_residual`, passing `ele_type`, `coords`, `connect`, `material_props`, `displacement`, and `num_runs`, returning the average time in seconds.

## Matrix Solve Preparation and Timing

🔙 Stepping out to the main script:  
The script prepares a matrix system and times its solve. It calls `solver_demo.prep_for_matrix_solve`, storing the stiffness matrix $K$ and residual $R$. It sets `method` to `"dense"` and `num_runs` to $5$, then calls `solver_demo.time_one_matrix_solve` with $K$, $R$, `method`, and `num_runs`, saving the average time in `avg_time_dense_solve`.

### 🔍 Stepping into `prep_for_matrix_solve`

The function `prep_for_matrix_solve` assembles the system for solving. It builds $K$ using `assemble.global_stiffness`, $F$ with `assemble.global_traction`, and computes $R$ as $F$ minus `assemble.global_residual`. It reshapes `displacement` and applies fixed boundary conditions by modifying $K$ and $R$ rows for each fixed DOF, returning $K$ and $R$.

### 🔍 Stepping into `time_one_matrix_solve`

The function `time_one_matrix_solve` times a matrix solve. For `method="dense"`, it sets `func` to `np.linalg.solve` with $K$ and $R$, then calls `time_function_call` with these and `num_runs`, returning the average solve time in seconds.

## Sparsity Analysis

🔙 Stepping out to the main script:  
The script examines the stiffness matrix’s sparsity. It sets `fname` to `"solver_global_stiffness_1.png"` and calls `solver_demo.analyze_and_visualize_matrix` with $K$ and `fname` to compute diagnostics and visualize the matrix pattern.

### 🔍 Stepping into `analyze_and_visualize_matrix`

The function `analyze_and_visualize_matrix` analyzes and plots the matrix. It converts $K$ to sparse CSR format if needed, computes the condition number with `compute_condition_number`, sparsity with `compute_sparsity`, and bandwidth with `compute_bandwidth`. It creates a plot, uses `imshow` by default to show the sparsity pattern, adds a title with metrics, and saves it to `fname` at 300 DPI.

## Sparse Assembly and Solvers

🔙 Stepping out to the main script:  
The script tests sparse matrix assembly timing. It sets `num_runs` to $5$ and calls `solver_demo.time_assemble_global_stiffness_sparse`, storing the average time in `avg_time_global_stiffness_sparse`. It also tests sparse solvers: for `"sparse"`, it uses `method = "sparse"` and `num_runs = 10`, calling `solver_demo.time_one_matrix_solve`; for `"sparse_iterative"`, it uses `method = "sparse_iterative"` and `num_runs = 10`, storing times in `avg_time_sparse_solve` and `avg_time_sparse_iterative_solve`.

### 🔍 Stepping into `time_assemble_global_stiffness_sparse`

The function `time_assemble_global_stiffness_sparse` measures sparse stiffness matrix assembly time. It calls `time_function_call` with `assemble.global_stiffness_sparse`, passing `ele_type`, `coords`, `connect`, `material_props`, `displacement`, and `num_runs`, returning the average time in seconds.

### 🔍 Stepping into `time_one_matrix_solve` (Sparse Variants)

For `"sparse"`, it converts $K$ to CSR sparse format and uses `spla.spsolve` via `time_function_call`. For `"sparse_iterative"`, it uses `spla.gmres` with an optional ILU preconditioner from `spla.spilu`, also via `time_function_call`, returning the average solve time in seconds.

## Hyperelastic Solution and Visualization

🔙 Stepping out to the main script:  
The script runs the example to look at the results. It calls `hyperelastic_solver` with `nr_num_steps = 5` and `nr_print = True`, storing results in `displacements_all` and `nr_info_all`. It sets `fname` to `"disp.gif"` and calls `viz.make_deformation_gif` to animate the results.

### 🔍 Stepping into `hyperelastic_solver`

The function `hyperelastic_solver` solves a hyperelastic finite element problem using a Newton-Raphson scheme with incremental loading. It initializes a displacement vector of length ($nnode \cdot ndof$) and iterates over `nr_num_steps`. For each step, it applies a load factor ($(step + 1) / nr_num_steps$), assembles $K$ with `assemble.global_stiffness`, $F$ with `assemble.global_traction`, and $R$ as $loadfactor \cdot F$ minus `assemble.global_residual`. It adjusts $K$ and $R$ for fixed nodes, solves for displacement corrections with `np.linalg.solve`, and updates until the correction norm is below `nr_tol` or `nr_maxit` is reached. It returns all displacements and iteration info, printing details if `nr_print` is `True`.

### 🔍 Stepping into `make_deformation_gif`

The function `viz.make_deformation_gif` creates an animated GIF showing deformation progression. It sets up a plot with fixed limits based on the maximum displacement from `displacements_all`, plots the undeformed mesh in gray, and initializes lines and a scatter for the deformed state with edges like $(0,4)$ to $(7,0)$ for `"D2_nn8_quad"`. It defines an `update` function to adjust lines and scatter with each frame’s magnified displacement, updating the title with frame numbers. It uses `animation.FuncAnimation` to generate the GIF, saving it to `gif_path` with the Pillow writer at a specified `interval`.
