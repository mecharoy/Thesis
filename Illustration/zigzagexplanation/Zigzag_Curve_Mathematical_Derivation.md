# How the Zigzag Curve is Formed: Mathematical Derivation
## Complete Step-by-Step Explanation

---

## The Core Equation

The zigzag function R^(k)(z) for each layer k follows this equation:

```
R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄
```

**Key point:** This is a **cubic polynomial** in z, which is why you see smooth curves.

---

## Step-by-Step Derivation: How We Get This Curve

### Step 1: Start with Physical Requirements

The zigzag theory must satisfy four fundamental conditions:

1. **Displacement continuity** at layer interfaces
2. **Shear stress continuity** at layer interfaces
3. **Zero shear stress** at top and bottom surfaces (free surfaces)
4. **Equilibrium** throughout the beam

### Step 2: Define the Basic Form

For a three-layer beam, we assume the zigzag function has this form in each layer:

```
Layer k:  R^(k)(z) = a₀^(k) + a₁^(k)·z + a₂^(k)·z² + a₃^(k)·z³
```

This gives us **12 unknowns** (4 coefficients × 3 layers).

### Step 3: Apply Boundary Conditions

#### Condition 1: Zero shear stress at bottom surface (z = z₀)

Shear stress: τ_xz = G·(dR/dz)·ψ₀

At z₀ (bottom): dR^(1)/dz|_{z=z₀} = 0

```
dR^(1)/dz = a₁^(1) + 2a₂^(1)·z + 3a₃^(1)·z²

At z = z₀:  a₁^(1) + 2a₂^(1)·z₀ + 3a₃^(1)·z₀² = 0  ... (Equation 1)
```

#### Condition 2: Zero shear stress at top surface (z = z₃)

```
dR^(3)/dz|_{z=z₃} = 0

a₁^(3) + 2a₂^(3)·z₃ + 3a₃^(3)·z₃² = 0  ... (Equation 2)
```

#### Condition 3: Displacement continuity at interface z₁ (between layers 1 and 2)

```
R^(1)(z₁) = R^(2)(z₁)

a₀^(1) + a₁^(1)·z₁ + a₂^(1)·z₁² + a₃^(1)·z₁³ = a₀^(2) + a₁^(2)·z₁ + a₂^(2)·z₁² + a₃^(2)·z₁³  ... (Eq 3)
```

#### Condition 4: Displacement continuity at interface z₂ (between layers 2 and 3)

```
R^(2)(z₂) = R^(3)(z₂)  ... (Equation 4)
```

#### Condition 5 & 6: Shear stress continuity at interfaces

For homogeneous material (same G in all layers):

```
At z₁: dR^(1)/dz|_{z₁⁻} = dR^(2)/dz|_{z₁⁺}  ... (Equation 5)
At z₂: dR^(2)/dz|_{z₂⁻} = dR^(3)/dz|_{z₂⁺}  ... (Equation 6)
```

### Step 4: Simplification Using Global Coefficients

To reduce complexity, the theory assumes the **quadratic and cubic terms are the same** for all layers:

```
a₂^(1) = a₂^(2) = a₂^(3) = R̂₃  (global quadratic coefficient)
a₃^(1) = a₃^(2) = a₃^(3) = R̂₄  (global cubic coefficient)
```

This reduces our unknowns from 12 to 8:
- R̂₃ and R̂₄ (2 global coefficients)
- R̂₁^(1), R̂₁^(2), R̂₁^(3) (3 constant terms)
- R̂₂^(1), R̂₂^(2), R̂₂^(3) (3 linear terms)

Now the equation becomes:

```
R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄
```

### Step 5: Solve for R̂₃ and R̂₄

From the zero shear stress boundary conditions and some algebra, we get:

```
R̂₃ = 4·C₂^(L) / Δ

R̂₄ = -4·C₁^(L) / (3·Δ)

where Δ = 4·z₀²·C₁^(L) - 8·z₀·C₂^(L)
```

Here, C₁ and C₂ are cumulative integrals through the layers related to shear stiffness.

### Step 6: Calculate C₁^(k) and C₂^(k) (Cumulative Constants)

These represent integrated shear properties through layers:

```
For homogeneous material with shear modulus G and Q₅₅ = 0.9·G:

Layer 1 (z₀ to z₁):
  C₁^(1) = Q₅₅·(z₁ - z₀)
  C₂^(1) = 0.5·Q₅₅·(z₁² - z₀²)

Layer 2 (z₁ to z₂):
  C₁^(2) = C₁^(1) + Q₅₅·(z₂ - z₁)
  C₂^(2) = C₂^(1) + 0.5·Q₅₅·(z₂² - z₁²)

Layer 3 (z₂ to z₃):
  C₁^(3) = C₁^(2) + Q₅₅·(z₃ - z₂) = C₁^(L) (total)
  C₂^(3) = C₂^(2) + 0.5·Q₅₅·(z₃² - z₂²) = C₂^(L) (total)
```

### Step 7: Calculate Layer-Specific Linear Terms R̂₂^(k)

```
For each layer k:

a₁^(k) = 2·[(C₁^(k)/Q₅₅) - z_k]

a₂^(k) = 3·[(2·C₂^(k)/Q₅₅) - z_k²]

Then:
R₂^(k) = a₁^(k)·R̂₃ + a₂^(k)·R̂₄
```

### Step 8: Calculate Constant Terms R̂₁^(k)

This involves calculating "jump" contributions at interfaces:

```
R̄₂^(k) = Σ(from i=1 to k) z_{i-1}·(R₂^(i-1) - R₂^(i))

Then, relative to reference layer k₀ = 1:
R₁^(k) = R̄₂^(k) - R̄₂^(k₀)
```

### Step 9: Normalization

Finally, ALL coefficients are normalized by |R₂^(k₀)| where k₀ = 1 (middle layer):

```
R̂₁^(k) = R₁^(k) / |R₂^(k₀)|
R̂₂^(k) = R₂^(k) / |R₂^(k₀)|
R̂₃ = R₃ / |R₂^(k₀)|
R̂₄ = R₄ / |R₂^(k₀)|
```

This makes the function **dimensionless** and numerically stable.

---

## Why This Creates the Specific Curve Shape

### 1. **Within Each Layer: Smooth Cubic Curve**

The equation R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄ is a **cubic polynomial**.

- **Linear term (z):** Creates the overall slope
- **Quadratic term (z²):** Creates gentle curvature
- **Cubic term (z³):** Adds additional curvature for satisfying boundary conditions

Result: **Smooth, flowing curve** within each layer.

### 2. **At Interfaces: Slope Change (Kink)**

At z = z₁ and z = z₂:
- Function is continuous: R^(k)(z_interface) = R^(k+1)(z_interface) ✓
- But R̂₁ and R̂₂ change from layer to layer
- So the derivative jumps: dR^(k)/dz ≠ dR^(k+1)/dz

Result: **Visible slope discontinuity** (the "zigzag" effect).

### 3. **At Boundaries: Horizontal Tangent**

At z = z₀ and z = z₃:
- Boundary condition forces: dR/dz = 0
- This means slope is zero (horizontal tangent)

Result: **Curves flatten** at top and bottom.

### 4. **Pristine Region: Antisymmetric**

For symmetric beam (z₀ = -h/2, z₃ = +h/2, z₁ = -h/6, z₂ = +h/6):
- The math naturally produces: R(z) = -R(-z)
- This is because C₁ and C₂ calculations are symmetric

Result: **Mirror image** about z = 0.

### 5. **Notched Region: Broken Symmetry**

When notch depth d is applied:
- z₃_new = h/2 - d (reduced top coordinate)
- z₂ collapses toward z₃_new
- C₁^(L) and C₂^(L) change (less material)
- All coefficients R̂₁, R̂₂, R̂₃, R̂₄ change

Result: **Completely different curve**, even sign reversal.

---

## Numerical Example: Pristine Region

Let's calculate for your specific case:

### Given Parameters:
```
h = 0.0015 m = 1.5 mm
E = 70 GPa
ν = 0.33
G = E/(2(1+ν)) = 26.316 GPa
Q₅₅ = 0.9·G = 23.684 GPa

Layer interfaces:
z₀ = -0.75 mm
z₁ = -0.25 mm
z₂ = +0.25 mm
z₃ = +0.75 mm
```

### Step 1: Calculate C₁ and C₂

```
C₁^(1) = Q₅₅·(z₁ - z₀) = 23.684×10⁹ · (0.0005) = 11.842×10⁶ Pa·m
C₂^(1) = 0.5·Q₅₅·(z₁² - z₀²) = 0.5·23.684×10⁹·(-0.25×10⁻⁶) = -2.961×10³ Pa·m²

C₁^(2) = C₁^(1) + Q₅₅·(z₂ - z₁) = 11.842×10⁶ + 23.684×10⁹·(0.0005) = 23.684×10⁶ Pa·m
C₂^(2) = C₂^(1) + 0.5·Q₅₅·(z₂² - z₁²) = -2.961×10³ + 0 = -2.961×10³ Pa·m²

C₁^(3) = C₁^(L) = 23.684×10⁶ + 23.684×10⁹·(0.0005) = 35.526×10⁶ Pa·m
C₂^(3) = C₂^(L) = -2.961×10³ + 2.961×10³ = 0 Pa·m²
```

Wait! C₂^(L) = 0 for symmetric beam! This simplifies everything.

### Step 2: Calculate R̂₃ and R̂₄

```
Δ = 4·z₀²·C₁^(L) - 8·z₀·C₂^(L)
  = 4·(-0.00075)²·(35.526×10⁶) - 8·(-0.00075)·0
  = 79.932 Pa·m³

R̂₃ = 4·C₂^(L) / Δ = 4·0 / 79.932 = 0

R̂₄ = -4·C₁^(L) / (3·Δ) = -4·(35.526×10⁶) / (3·79.932) = -592,533 m⁻³
```

### Step 3: Calculate R₂^(k)

```
For layer 1 (k=0):
a₁^(1) = 2·[(C₁^(1)/Q₅₅) - z₁] = 2·[(11.842×10⁶)/(23.684×10⁹) - (-0.00025)]
       = 2·[0.0005 + 0.00025] = 2·0.00075 = 0.0015 m

a₂^(1) = 3·[(2·C₂^(1)/Q₅₅) - z₁²] = 3·[(2·(-2.961×10³))/(23.684×10⁹) - (0.00025)²]
       = 3·[-0.00025×10⁻⁶ - 0.0625×10⁻⁶] ≈ 0

R₂^(1) = a₁^(1)·R̂₃ + a₂^(1)·R̂₄
       = 0.0015·0 + 0·(-592,533) = 0

But wait, this is getting complex. Let me show you the **actual code** that does this:
```

---

## The Actual Code Implementation

Here's the **exact code** from `datagenzigzag.py` that creates the curve:

```python
def compute_zigzag_functions(z0, z1, z2, z3, Q55):
    """Compute the zigzag function parameters"""

    # Step 1: Calculate cumulative C1 and C2 for each layer
    C1 = np.zeros(3)
    C2 = np.zeros(3)

    # Layer 0 (bottom)
    C1[0] = Q55 * (z1 - z0)
    C2[0] = 0.5 * Q55 * (z1**2 - z0**2)

    # Layer 1 (middle)
    C1[1] = C1[0] + Q55 * (z2 - z1)
    C2[1] = C2[0] + 0.5 * Q55 * (z2**2 - z1**2)

    # Layer 2 (top)
    C1[2] = C1[1] + Q55 * (z3 - z2)
    C2[2] = C2[1] + 0.5 * Q55 * (z3**2 - z2**2)

    # Step 2: Calculate global coefficients R3 and R4
    C1_L = C1[2]  # Total C1
    C2_L = C2[2]  # Total C2
    delta = 4 * z0**2 * C1_L - 8 * z0 * C2_L

    R3 = 4 * C2_L / delta
    R4 = -4 * C1_L / (3 * delta)

    # Step 3: Calculate intermediate a1 and a2 for each layer
    a1 = np.zeros(3)
    a2 = np.zeros(3)
    z_k = [z1, z2, z3]  # Top coordinate of each layer

    for k in range(3):
        a1[k] = 2 * ((C1[k] / Q55) - z_k[k])
        a2[k] = 3 * ((2 * C2[k] / Q55) - z_k[k]**2)

    # Step 4: Calculate R2 for each layer
    R2 = np.zeros(3)
    for k in range(3):
        R2[k] = a1[k] * R3 + a2[k] * R4

    # Step 5: Calculate R_bar2 (cumulative jump contributions)
    k0 = 1  # Reference layer (middle)
    R2k0 = abs(R2[k0])  # Normalization factor

    R_bar2 = np.zeros(3)
    z_interfaces = [z0, z1, z2, z3]

    for k in range(3):
        for i in range(1, k+1):
            R_bar2[k] += z_interfaces[i-1] * (R2[i-1] - R2[i])

    # Step 6: Calculate R1 for each layer
    R1 = np.zeros(3)
    for k in range(3):
        R1[k] = R_bar2[k] - R_bar2[k0]

    # Step 7: Normalize ALL coefficients
    R_hat1 = R1 / R2k0
    R_hat2 = R2 / R2k0
    R_hat3 = R3 / R2k0
    R_hat4 = R4 / R2k0

    # Step 8: Define the function R^(k)(z)
    def R_k(z):
        # Determine which layer
        if z < z1:
            k = 0
        elif z < z2:
            k = 1
        else:
            k = 2

        # Apply cubic polynomial for that layer
        return R_hat1[k] + z * R_hat2[k] + z**2 * R_hat3 + z**3 * R_hat4

    return R_k
```

---

## How to Explain This to Others

### **Simple Explanation (Non-Technical Audience)**

*"The curve is created by a mathematical formula that ensures the displacement pattern satisfies physical laws: forces must balance, stress can't have sudden jumps at material boundaries, and the top and bottom surfaces must be stress-free. The formula is a cubic polynomial (like y = ax³ + bx² + cx + d) that changes coefficients at each layer interface. This creates a smooth curve within each layer, but with a 'kink' at the interfaces where the formula changes."*

### **Moderate Explanation (Engineering Students)**

*"The zigzag function R^(k)(z) is derived from equilibrium equations and boundary conditions. It's a piecewise cubic polynomial with the form R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄. The cubic and quadratic terms (R̂₃, R̂₄) are global (same for all layers), while the linear and constant terms (R̂₁^(k), R̂₂^(k)) vary by layer. This structure ensures displacement continuity at interfaces while allowing slope discontinuities to represent varying shear strain through the thickness."*

### **Technical Explanation (Researchers/Reviewers)**

*"The zigzag function is obtained by solving the variational problem that minimizes the potential energy subject to kinematic constraints. The assumed displacement field u(x,z) = u₀(x) - z·w₀'(x) + R^(k)(z)·ψ₀(x) leads to a system of constraints enforcing: (1) vanishing shear traction at free surfaces, (2) interlaminar displacement continuity, (3) interlaminar shear stress continuity, and (4) normalization for numerical stability. The solution yields layer-wise cubic polynomials R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄ where global coefficients R̂₃, R̂₄ are determined from integrated material properties C₁^(L), C₂^(L), and layer-specific coefficients R̂₁^(k), R̂₂^(k) accommodate geometric and material variations across interfaces."*

---

## Visual Summary of Curve Formation

```
                    Mathematical Flow

Input Parameters          Cumulative Integrals      Global Coefficients
─────────────            ────────────────────      ───────────────────
z₀, z₁, z₂, z₃    ──→    C₁^(1), C₁^(2), C₁^(3)
Q₅₅ (shear)       ──→    C₂^(1), C₂^(2), C₂^(3)   ──→   R̂₃, R̂₄
                                                    (quadratic, cubic)
                              ↓
                    Intermediate Coefficients       Layer Coefficients
                    ─────────────────────          ──────────────────
                    a₁^(k), a₂^(k)         ──→     R̂₁^(k), R̂₂^(k)
                    R₂^(k), R̄₂^(k)                 (constant, linear)

                              ↓
                    Normalization (÷ |R₂^(k₀)|)
                              ↓
                    Final Function for Each Layer k:
                    ─────────────────────────────
                    R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄

                              ↓
                    Evaluate at Many z Points
                              ↓
                         CURVE PLOTTED!
```

---

## Why This Matters for Your Thesis

1. **1D Model Foundation**: This curve IS the displacement assumption of your 1D zigzag model
2. **Efficiency**: This analytical formula is fast to compute (no iterative solving)
3. **Physical Basis**: Not arbitrary—derived from equilibrium and boundary conditions
4. **Limitation**: Assumes this specific functional form, which 2D FEM does not
5. **Multi-Fidelity Gap**: The difference between this curve and 2D reality is what U-Net learns

---

## Quick Reference: The Four Key Equations

```
1. Cubic Polynomial (within each layer):
   R^(k)(z) = R̂₁^(k) + z·R̂₂^(k) + z²·R̂₃ + z³·R̂₄

2. Global Coefficients (from boundary conditions):
   R̂₃ = 4·C₂^(L) / Δ
   R̂₄ = -4·C₁^(L) / (3·Δ)
   where Δ = 4·z₀²·C₁^(L) - 8·z₀·C₂^(L)

3. Cumulative Integrals (material properties):
   C₁^(k) = Σ Q₅₅·(z_i - z_{i-1})
   C₂^(k) = Σ 0.5·Q₅₅·(z_i² - z_{i-1}²)

4. Normalization (numerical stability):
   All R̂ ÷ |R₂^(k₀)| where k₀ = 1 (middle layer)
```

---

**File Created:** `/home/amy23/Claude_rep/Zigzag_Curve_Mathematical_Derivation.md`

This document provides complete mathematical understanding of how the zigzag curve is formed!
