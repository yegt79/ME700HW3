# Finite Element Analysis Tutorial: `tutorial_discretization.ipynb`

The project focuses on finite element analysis, and I’m starting with the `tutorial_discretization.ipynb` file. I am beginning the analysis from the entry point of this file and using the debug tool to dive into more detail and understand what this file is calling.

## Initial Setup and Visualization of Gauss Points

The script imports helper functions from a custom `finiteelementanalysis` package, specifically `discretization_demo_helper_fcns` (aliased as `di_demo`). It then sets up a 2D, 4-node quadrilateral element (`D2_nn4_quad`) with 4 Gauss points for numerical integration and calls `visualize_gauss_pts` to generate and save a plot of these points to `D2_nn4_quad_4gp.png`.

### Stepping into `visualize_gauss_pts`

This function, defined in the `discretization_demo_helper_fcns` module, is designed to visualize Gauss points. It accepts:

- `fname`: Output filename.
- `ele_type`: Element type (e.g., `"D2_nn3_tri"`, `"D2_nn6_tri"`, `"D2_nn4_quad"`, `"D2_nn8_quad"`).
- `num_pts`: Number of Gauss points.

The function:
1. Retrieves Gauss points and weights via `gauss_pts_and_weights(ele_type, num_pts)`.
2. Defines reference nodes based on `ele_type`:
   - **Triangles**: Supports a 3-node linear triangle (e.g., \([1, 0]\), \([0, 1]\), \([0, 0]\)) and a 6-node quadratic triangle with mid-edge nodes.
   - **Quadrrilaterals**: Covers a 4-node bilinear quad (e.g., \([-1, -1]\) to \([-1, 1]\)) and an 8-node quadratic quad with mid-edge nodes.
3. Defines edges to connect these nodes appropriately.
4. Raises a `ValueError` if `ele_type` is not in the correct format.
5. Plots using Matplotlib:
   - Edges as black lines.
   - Nodes as labeled blue dots (e.g., \(N_1\) to \(N_8\)).
   - Gauss points as labeled red dots (e.g., \(G_1\) to \(G_4\) for `num_pts = 4`).
6. Titles the plot with the element type, uses natural coordinates (\(\xi, \eta\)), and saves it to `fname` at 300 DPI.

**Improvement Idea**:  
Input validation could be enhanced. The function raises a `ValueError` for unsupported `ele_type`, but it could also validate `num_pts` (e.g., reject non-numeric inputs).

## Interpolating a Scalar Field in Natural Coordinates

Stepping out to the main tutorial script:  
The script defines `fcn`, a bilinear scalar field over natural coordinates (\(\xi, \eta\)), likely a test case for interpolation. Reusing `"D2_nn4_quad"`, it defines the same 4-node quadrilateral’s reference nodes as before. The function `node_values` computes `fcn` at these corners, yielding \([1, 5, 7, 3]\). It then calls `di_demo.plot_interpolate_field_natural_coords_single_element`, saving a plot to `D2_nn4_quad_interpolate_fcn.png`.

### Stepping into `plot_interpolate_field_natural_coords_single_element`

This function plots a scalar field interpolated across a single element in natural coordinates, saving it to `fname`. It takes:
- `fname`: Output filename.
- `ele_type`: Element type (e.g., `"D2_nn4_quad"`).
- `node_values`: Field values at nodes.
- `num_interp_pts`: Optional number of interpolation points (default 10).

For `"D2_nn4_quad"`, it:
1. Creates a \(10 \times 10\) grid of points from \(-1\) to \(1\) in \(\xi\) and \(\eta\), flattening them into `xi_filtered` and `eta_filtered`.
2. Defines reference nodes consistently with `visualize_gauss_pts`.
3. Calls `interpolate_field_natural_coords_single_element` to compute interpolated values across these points.
4. Uses Matplotlib to scatter-plot the results.

## Mapping to Physical Coordinates

Stepping out to the main tutorial script:  
The script defines `fcn`, a bilinear field in physical coordinates (\(x, y\)). For `"D2_nn4_quad"`, `node_coords` specifies a quadrilateral in physical space (\([1, 1]\) to \([0, 2]\)), and `node_values` computes `fcn` at these nodes (\([6.5, 8, 13.5, 6]\)). It calls `visualize_isoparametric_mapping_single_element`, saving a plot to `D2_nn4_quad_interpolate_fcn_physical_coords.png`, likely showing the field mapped from natural to physical coordinates.

**Improvement Idea**:  
Centralize `node_coords` definitions to avoid redundancy across demos.

### Stepping into `visualize_isoparametric_mapping_single_element`

This function visualizes the isoparametric mapping from natural to physical coordinates, saving to `fname`. For `"D2_nn4_quad"`, it:
1. Samples a \(20 \times 20\) grid (\(-1\) to \(1\)) in \(\xi, \eta\).
2. Interpolates the field and coordinates using `interpolate_field_natural_coords_single_element`.
3. Plots two subplots:
   - Reference element (natural coords).
   - Mapped element (physical coords).
4. Uses coolwarm scatter points, labeled nodes, and colorbars.

## Gradient Transformation

Stepping out to the main tutorial script:  
The script defines a bilinear field and its analytical derivatives (\(\partial f / \partial x\), \(\partial f / \partial y\)). For `"D2_nn4_quad"`, a physical quadrilateral is set (\([0, 0]\) to \([0, 2]\)), with `node_values` as \([0, 4, 10, 6]\). At the element center (\(\xi, \eta = 0, 0\)), it maps to physical coordinates (\(x, y = 1, 1\)). It computes:
- Analytical derivative: \([3.5, 4.5]\).
- Numerical gradient in natural coordinates.
- Transforms it to physical coordinates and checks if they match.

### Stepping into `transform_gradient_to_physical`

This function transforms a gradient from natural (\(\xi, \eta\)) to physical (\(x, y\)) coordinates. It takes:
- `ele_type`: E.g., `"D2_nn4_quad"`.
- `node_coords`: Physical nodes (\([0, 0]\) to \([0, 2]\)).
- `xi_vals`, `eta_vals`: Here, \([0]\) and \([0]\).
- `gradient_natural`: A \(2 \times 1\) array from `interpolate_gradient_natural_coords_single_element`.

#### Stepping into `interpolate_gradient_natural_coords_single_element`

Called within `transform_gradient_to_physical`, this function:
1. Identifies the field as scalar.
2. Initializes `gradient_natural`.
3. Loops over (\(\xi, \eta\)) pairs.
4. Evaluates shape function derivatives (\(dN/d\xi\)) at (\(\xi, \eta\)).
5. Computes the gradient as \(dN/d\xi^T @ node_values\), yielding \(\partial f / \partial \xi\) and \(\partial f / \partial \eta\).

Stepping out:  
`transform_gradient_to_physical` then:
1. Checks if the field is scalar or vector.
2. Loops over each (\(\xi, \eta\)) pair (here, just \((0, 0)\)).
3. Computes the Jacobian (\(J\)).
4. Inverts \(J\) and multiplies its transpose with `gradient_natural` to get `gradient_physical`.

**Improvement Idea**:  
Add a `try-except` around `np.linalg.inv(J)` to catch singular Jacobians, raising a descriptive error.

## Computing Integrals

Stepping out to the main tutorial script:  
- `element_area` calculates the quadrilateral’s area using the shoelace formula.
- `integral_of_deriv` computes the analytical integral of the derivatives over the element.

### Stepping into `compute_integral_of_derivative`

This function:
1. Computes the gradient in natural coordinates using `interpolate_gradient_natural_coords_single_element`.
2. Transforms it to physical coordinates with `transform_gradient_to_physical`.

#### Stepping into `transform_gradient_to_physical` (Again)

It:
1. Identifies a scalar field, initializes `gradient_physical`, and iterates over points.
2. For each (\(\xi, \eta\)):
   - Computes the Jacobian (\(J\)) via `compute_jacobian`.

#### Stepping into `compute_jacobian`

This function:
1. Maps `ele_type` to shape function derivatives.
2. Evaluates \(dN/d\xi\) at (\(\xi, \eta\)).
3. Computes the Jacobian as \(node_coords^T @ dN/d\xi\).

Stepping out:  
`transform_gradient_to_physical` then:
1. Inverts \(J\) and applies \(J_{inv}^T @ gradient_natural\), transforming \(\partial f / \partial \xi\), \(\partial f / \partial \eta\) to \(\partial f / \partial x\), \(\partial f / \partial y\).

Stepping out:  
`compute_integral_of_derivative` then:
1. Calculates the Jacobian determinant (`det_J`) via `compute_jacobian`.
2. Sums the weighted gradient contribution: \(weight \cdot gradient_physical \cdot det_J\).

## Call Hierarchy

```plaintext
tutorial_discretization.ipynb
├── visualize_gauss_pts
├── plot_interpolate_field_natural_coords_single_element
├── visualize_isoparametric_mapping_single_element
├── transform_gradient_to_physical
│   ├── interpolate_gradient_natural_coords_single_element
├── compute_integral_of_derivative
│   ├── interpolate_gradient_natural_coords_single_element
│   ├── transform_gradient_to_physical
│   │   ├── compute_jacobian
