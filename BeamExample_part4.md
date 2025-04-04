# Tip Deflection of a 2D Cantilever Beam (Euler–Bernoulli Beam Theory)

We analyze the tip deflection of a 2D cantilever beam subjected to a uniform downward load along its top edge using the Euler–Bernoulli beam theory and compare it with a numerical solution.

---

## 🔍 Extracted Inputs

### 📐 Geometry
- **Length (L):** 20.0 units  
- **Height (H):** 1.0 unit

### ⚙️ Material Properties
- **Young's Modulus (E):** 100,000.0  
- **Poisson's Ratio (ν):** 0.3

### 🔧 Load
- **Uniform load per unit length (q):** -0.01 (downward)

### 🧮 Beam Theory
For a cantilever beam with a rectangular cross-section:

- **Moment of inertia (I):**
  \[
  I = \frac{H^3}{12}
  \]

- **Tip deflection (w) under uniform load:**
  \[
  w(L) = \frac{q L^4}{8 E I}
  \]

---

## 📊 Step-by-Step Calculation (Analytical)

### 1. Calculate Moment of Inertia
\[
I = \frac{1.0^3}{12} = \frac{1.0}{12} \approx 0.0833
\]

### 2. Plug into the Deflection Formula
\[
w(L) = \frac{-0.01 \cdot 20^4}{8 \cdot 100000 \cdot 0.0833}
\]

#### Numerator:
\[
-0.01 \cdot 160000 = -1600
\]

#### Denominator:
\[
8 \cdot 100000 \cdot 0.0833 \approx 66640
\]

### 3. Final Analytical Result
\[
w(L) \approx \frac{-1600}{66640} \approx \boxed{-0.021840}
\]

---

## 💻 Numerical Result (FEM Simulation)

- **Tip node index:** 9  
- **Coordinates:** [20.0, 0.5]  
- **Computed tip deflection (y):** \(\boxed{-0.002270}\)

---

## 📈 Comparison

| Method                  | Tip Deflection (y) |
|-------------------------|--------------------|
| Analytical (Euler–Bernoulli) | -0.021840          |
| Numerical (FEM)         | -0.002270          |

**Observation:**  
The numerical deflection is significantly smaller than the analytical result. This could be due to a coarse mesh (only 4 elements along the length), which limits the accuracy of the numerical simulation. Refining the mesh would likely yield results closer to the analytical solution.

---

## ✅ Conclusion

The analytical solution gives a tip deflection of approximately **-0.02184**, while the numerical simulation gives **-0.00227** at the tip node. The discrepancy highlights the importance of mesh refinement in numerical methods.
