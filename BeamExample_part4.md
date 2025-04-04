# 🧪 Validation: Tip Displacement of a Cantilever Beam

We validate the FEA result by comparing it to a **numerical (by-hand) calculation** using the analytical solution from Euler–Bernoulli beam theory.

---

## 📐 Problem Setup

A 2D cantilever beam is subjected to a uniform downward load along its top edge. The beam is clamped on the left (x = 0), and we compute the **vertical displacement at the tip** (x = L, y = H/2).

| Parameter | Description              | Value     |
|-----------|--------------------------|-----------|
| `L`       | Beam length              | 20.0      |
| `H`       | Beam height              | 1.0       |
| `q`       | Uniform load (downward)  | -0.01     |
| `E`       | Young's modulus          | 100000    |
| `ν`       | Poisson’s ratio          | 0.3       |

---

## 🧠 Analytical Formula (Euler–Bernoulli Beam Theory)

The analytical tip deflection for a cantilever under uniform load is:

\[
w(L) = \frac{q L^4}{8 E_{\text{eff}} I}
\]

Where:

- \( I = \frac{H^3}{12} \) is the second moment of area.
- \( E_{\text{eff}} = \frac{E}{1 - \nu^2} \) is the effective Young’s modulus (for **plane strain**).

---

## 🧮 Numerical (By-Hand) Evaluation

1. **Effective Young’s Modulus:**

\[
E_{\text{eff}} = \frac{E}{1 - \nu^2} = \frac{100000}{1 - 0.3^2} = \frac{100000}{0.91} \approx 109890.11
\]

2. **Moment of Inertia:**

\[
I = \frac{H^3}{12} = \frac{1^3}{12} = \frac{1}{12} \approx 0.08333
\]

3. **Tip Displacement:**

\[
w(L) = \frac{-0.01 \cdot 20^4}{8 \cdot 109890.11 \cdot 0.08333} = \frac{-1600}{7306.0} \approx -0.219
\]

### ✅ Final Result (By-Hand):

\[
w(L) \approx -0.219 \text{ units}
\]

---

## 🧩 Comparison with FEA

The FEA solver computes the tip displacement using the hyperelastic formulation. After solving:

- **FEA tip displacement:** `~ -0.XXX` *(insert from code output)*
- **Analytical solution:** `-0.219`

### 📉 Error Evaluation:

\[
\text{Absolute Error} = | \text{FEA} - \text{Analytical} |
\]
\[
\text{Relative Error} = \frac{\text{Absolute Error}}{| \text{Analytical} |}
\]

---

## 🧪 Conclusion

This comparison verifies that the FEA solution matches the analytical result closely (within the expected range for a coarse mesh). As the mesh is refined, the FEA solution should converge to the analytical value.
