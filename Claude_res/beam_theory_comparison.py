"""
Comprehensive Beam Theory Comparison: Zigzag vs Euler-Bernoulli vs Timoshenko
Single-case simulation with identical configuration for all three theories

Beam Configuration:
- Length: 3.0 m
- Height: 0.0015 m
- Width: 1.0 m
- Notch center: 1.75 m
- Notch width: 0.001 m
- Notch depth: 0.001 m
- Response location: 1.95 m, z=0.00075 m
- Elements: 2000
- Material: Aluminum (E=70 GPa, rho=2700 kg/m³)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.sparse import csr_matrix, lil_matrix
import gc
import time as python_time
import os
import pandas as pd
import time

# Configuration Parameters
L = 3.0                    # Beam length (m)
h = 0.0015                 # Beam height (m)
b = 1.0                    # Beam width (m)
notch_center = 1.75        # Notch center position (m)
notch_width = 0.001        # Notch width (m)
notch_depth = 0.001        # Notch depth (m)
response_x = 1.95          # Response measurement point (m)
target_elements = 2000     # Number of elements

# Material Properties (Aluminum)
E = 70e9                   # Young's modulus (Pa)
rho = 2700                 # Density (kg/m³)
nu = 0.33                  # Poisson's ratio
G = E / (2 * (1 + nu))     # Shear modulus
Q11 = E/(1 - nu**2)        # Reduced stiffness component
Q55 = G * 0.9              # Reduced shear stiffness component

# Simulation Parameters
total_duration = 300e-6    # Total simulation time (s)
active_duration = 50e-6    # Excitation duration (s)
frequency = 100e3          # Excitation frequency (Hz)
response_z = h/2           # Response point through thickness (mid-plane)

# Numerical parameters
dtype = np.float32
torch_dtype = torch.float32

# Set default tensor type for better memory efficiency
torch.set_default_dtype(torch_dtype)

# Enable TF32 for faster GPU operations on Ampere GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ============================================================================
# ZIGZAG BEAM THEORY IMPLEMENTATION (Copy from previous code)
# ============================================================================

def zigzag_get_beam_properties_at_x(x, h, notch_center, notch_width, notch_depth):
    """
    Get beam properties at position x with parameterized notch (Zigzag Theory)
    """
    x_original = x
    z0_local = -h/2  # Bottom coordinate

    # Calculate notch boundaries
    notch_start = notch_center - notch_width/2
    notch_end = notch_center + notch_width/2

    if notch_start <= x_original <= notch_end:  # Notch region
        # Reduce top coordinate by notch depth
        z3_notch = h/2 - notch_depth
        z1_local = z0_local + h/3   # First interface remains the same
        z2_local = h/2 - notch_depth        # Second interface

        # If notch is too deep, adjust interfaces
        if z3_notch <= z2_local:
            z2_local = z3_notch
        if z2_local <= z1_local:
            z1_local = z2_local

        return z0_local, z1_local, z2_local, z3_notch
    else:
        # Regular three-layer region
        z3_local = h/2            # Regular top coordinate
        z1_local = z0_local + h/3   # First interface
        z2_local = z0_local + 2*h/3     # Second interface
        return z0_local, z1_local, z2_local, z3_local

def zigzag_compute_zigzag_functions(z0, z1, z2, z3, Q55):
    """Compute the zigzag function parameters according to the exact theory"""
    # Calculate C_1^k and C_2^k for each layer
    C1 = np.zeros(3, dtype=dtype)
    C2 = np.zeros(3, dtype=dtype)

    # Layer 0 (Bottom layer)
    C1[0] = Q55 * (z1 - z0)
    C2[0] = 0.5 * Q55 * (z1**2 - z0**2)

    # Layer 1 (Middle layer)
    C1[1] = C1[0] + Q55 * (z2 - z1)
    C2[1] = C2[0] + 0.5 * Q55 * (z2**2 - z1**2)

    # Layer 2 (Top layer)
    C1[2] = C1[1] + Q55 * (z3 - z2)
    C2[2] = C2[1] + 0.5 * Q55 * (z3**2 - z2**2)

    # Calculate delta (determinant)
    C1_L = C1[2]
    C2_L = C2[2]
    delta = 4 * z0**2 * C1_L - 8 * z0 * C2_L

    # Calculate R3 and R4 (global coefficients)
    R3 = 4 * C2_L / delta
    R4 = -4 * C1_L / (3 * delta)

    # Calculate a_1^k and a_2^k for each layer
    a1 = np.zeros(3, dtype=dtype)
    a2 = np.zeros(3, dtype=dtype)
    z_k = [z1, z2, z3]

    for k in range(3):
        a1[k] = 2 * ((C1[k] / Q55) - z_k[k])
        a2[k] = 3 * ((2 * C2[k] / Q55) - z_k[k]**2)

    # Calculate R2^k for each layer
    R2 = np.zeros(3, dtype=dtype)
    for k in range(3):
        R2[k] = a1[k] * R3 + a2[k] * R4

    # Define the reference layer (middle layer)
    k0 = 1
    R2k0 = abs(R2[k0])

    # Calculate R_bar_2^k for each layer
    R_bar2 = np.zeros(3, dtype=dtype)
    z_interfaces = [z0, z1, z2, z3]

    for k in range(3):
        for i in range(1, k+1):
            R_bar2[k] += z_interfaces[i-1] * (R2[i-1] - R2[i])

    # Calculate R1^k for each layer
    R1 = np.zeros(3, dtype=dtype)
    for k in range(3):
        R1[k] = R_bar2[k] - R_bar2[k0]

    # Normalize all coefficients with respect to R2^k0
    R_hat1 = R1 / R2k0
    R_hat2 = R2 / R2k0
    R_hat3 = R3 / R2k0
    R_hat4 = R4 / R2k0

    # Define the R^k(z) function
    def R_k(z):
        """Calculate the zigzag function R^k(z) at any point z through the thickness"""
        # Determine which layer z is in
        if z < z1:
            k = 0
        elif z < z2:
            k = 1
        else:
            k = 2

        return R_hat1[k] + z * R_hat2[k] + z**2 * R_hat3 + z**3 * R_hat4

    # Define the derivative dR^k(z)/dz
    def dR_k_dz(z):
        """Calculate the derivative of the zigzag function at any point z"""
        # Determine which layer z is in
        if z < z1:
            k = 0
        elif z < z2:
            k = 1
        else:
            k = 2

        return R_hat2[k] + 2 * R_hat3 * z + 3 * R_hat4 * z**2

    return R_k, dR_k_dz

def zigzag_compute_C_matrices(L):
    # Compute C1 matrix
    C1 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=dtype) * L / 6.0

    # Compute C5 matrix
    C5 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype) / L

    # Compute C6 matrix
    C6 = np.array([[0.0, 1.0/L, 0.0, -1/L], [0, -1.0/L, 0, 1.0/L]], dtype=dtype)

    # Compute C7 matrix
    C7 = np.array([
        [12.0/L**3, 6.0/L**2, -12.0/L**3, 6.0/L**2],
        [6.0/L**2, 4.0/L, -6.0/L**2, 2.0/L],
        [-12.0/L**3, -6.0/L**2, 12.0/L**3, -6.0/L**2],
        [6.0/L**2, 2.0/L, -6.0/L**2, 4.0/L]
    ], dtype=dtype)

    return C1, C5, C6, C7

def zigzag_compute_A_matrices(Q11, Q55, z0, z1, z2, z3, R_k, dRz, b):
    """Integration helper function"""
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):
        # Get Gauss points and weights in [-1, 1]
        xi, w = np.polynomial.legendre.leggauss(n_points)

        # Map xi in [-1, 1] to z in [z_lower, z_upper]
        z_mapped = 0.5 * (z_upper - z_lower) * xi + 0.5 * (z_upper + z_lower)

        # Compute weighted sum using vectorized operations
        f_values = np.array([f(z_val) for z_val in z_mapped], dtype=dtype)
        integral = np.sum(w * f_values)

        # Multiply by the Jacobian of the transformation
        integral *= 0.5 * (z_upper - z_lower)

        return integral

    # Define the integrand functions
    def integrand_A11(z): return Q11
    def integrand_A12(z): return Q11 * z
    def integrand_A13(z): return Q11 * R_k(z)
    def integrand_A22(z): return Q11 * z**2
    def integrand_A23(z): return Q11 * z * R_k(z)
    def integrand_A33(z): return Q11 * R_k(z)**2
    def integrand_A33_bar(z): return Q55 * dRz(z)**2

    # Compute A matrices by integration over each layer
    A_values = np.zeros(7, dtype=dtype)
    integrands = [integrand_A11, integrand_A12, integrand_A13, integrand_A22,
                  integrand_A23, integrand_A33, integrand_A33_bar]

    # Vectorized computation over layers
    layer_bounds = [(z0, z1), (z1, z2), (z2, z3)]

    for z_lower, z_upper in layer_bounds:
        for i, integrand in enumerate(integrands):
            A_values[i] += integrate_over_thickness(integrand, z_lower, z_upper)

    # Multiply by width
    A_values *= b

    return tuple(A_values)

def zigzag_compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7):
    """Compute the element stiffness matrix using the provided formulation"""

    K11 = A11 * C5
    K12 = -A12 * C6
    K13 = A13 * C5
    K22 = A22 * C7
    K23 = -A23 * C6.T
    K33 = A33 * C5 + A33_bar * C1

    K_e = np.zeros((8, 8), dtype=dtype)

    K_e[0:2, 0:2] = K11
    K_e[0:2, 2:6] = K12
    K_e[0:2, 6:8] = K13

    K_e[2:6, 0:2] = K12.T
    K_e[2:6, 2:6] = K22
    K_e[2:6, 6:8] = K23

    K_e[6:8, 0:2] = K13.T
    K_e[6:8, 2:6] = K23.T
    K_e[6:8, 6:8] = K33

    return K_e

def zigzag_assemble_global_stiffness_matrix(x_coords, element_lengths):
    """Assemble global stiffness matrix for the non-uniform mesh"""
    num_elements = len(element_lengths)

    size_per_node = 4
    num_nodes = num_elements + 1
    size_global = size_per_node * num_nodes

    # Use sparse matrix format for memory efficiency
    K = lil_matrix((size_global, size_global), dtype=dtype)

    for e in range(num_elements):
        # Use the specific length and midpoint for this element
        L_e = element_lengths[e]
        x_mid = x_coords[e] + L_e / 2.0

        # Get beam properties at this position
        z0, z1, z2, z3 = zigzag_get_beam_properties_at_x(x_mid, h, notch_center, notch_width, notch_depth)

        # Calculate zigzag functions for this position
        R_k, R_k_z = zigzag_compute_zigzag_functions(z0, z1, z2, z3, Q55)

        # Calculate C matrices for this element's length
        C1, C5, C6, C7 = zigzag_compute_C_matrices(L_e)

        # Calculate A matrices
        A11, A12, A13, A22, A23, A33, A33_bar = zigzag_compute_A_matrices(Q11, Q55, z0, z1, z2, z3, R_k, R_k_z, b)

        # Calculate element stiffness matrix
        K_e = zigzag_compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7)

        # Global node indices for this element
        node1_idx = e
        node2_idx = e + 1

        global_dofs = np.array([
            node1_idx * size_per_node + 0,
            node2_idx * size_per_node + 0,
            node1_idx * size_per_node + 1,
            node1_idx * size_per_node + 2,
            node2_idx * size_per_node + 1,
            node2_idx * size_per_node + 2,
            node1_idx * size_per_node + 3,
            node2_idx * size_per_node + 3
        ], dtype=int)

        # Add the element stiffness to the global stiffness
        for i in range(len(global_dofs)):
            for j in range(len(global_dofs)):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]

    return K.tocsr()

def zigzag_compute_inertia_integrals(rho, b, z0, z1, z2, z3, R_k_func):
    """Compute inertia integrals for mass matrix"""
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):
        xi, w = np.polynomial.legendre.leggauss(n_points)
        z_mapped = 0.5 * (z_upper - z_lower) * xi + 0.5 * (z_upper + z_lower)
        jacobian = 0.5 * (z_upper - z_lower)

        # Vectorized computation
        f_values = np.array([f(z_val) for z_val in z_mapped], dtype=dtype)
        integral = np.sum(w * f_values)
        return integral * jacobian

    interfaces = [z0, z1, z2, z3]
    I = np.zeros((4, 4), dtype=dtype)

    rho_layer = [rho, rho, rho]  # Assuming constant density

    def integrand_I00(z, k_layer): return rho_layer[k_layer] * 1 * 1
    def integrand_I01(z, k_layer): return rho_layer[k_layer] * 1 * z
    def integrand_I02(z, k_layer): return rho_layer[k_layer] * 1 * R_k_func(z)
    def integrand_I11(z, k_layer): return rho_layer[k_layer] * z * z
    def integrand_I12(z, k_layer): return rho_layer[k_layer] * z * R_k_func(z)
    def integrand_I22(z, k_layer): return rho_layer[k_layer] * R_k_func(z) * R_k_func(z)

    I_values = np.zeros(6, dtype=dtype)
    integrands = [integrand_I00, integrand_I01, integrand_I02, integrand_I11, integrand_I12, integrand_I22]

    for k in range(3):
        z_lower, z_upper = interfaces[k], interfaces[k+1]
        for i, integrand in enumerate(integrands):
            I_values[i] += integrate_over_thickness(lambda z: integrand(z, k), z_lower, z_upper)

    I[0, 0] = I_values[0]  # I00
    I[0, 2] = I[2, 0] = I_values[1]  # I01
    I[0, 3] = I[3, 0] = I_values[2]  # I02
    I[2, 2] = I_values[3]  # I11
    I[2, 3] = I[3, 2] = I_values[4]  # I12
    I[3, 3] = I_values[5]  # I22
    I[1, 1] = I_values[0]  # Ihat = I00

    I *= b
    return I

def zigzag_compute_element_mass_matrix(I_bar, L_e):
    """Compute the element mass matrix using the inertia integrals"""
    a = L_e  # Element length

    # c1 matrix (2x2)
    c1 = (a/3) * np.array([
        [1, 1/2],
        [1/2, 1]
    ], dtype=dtype)

    # c2 matrix (2x4)
    c2 = (1/2) * np.array([
        [-1, a/6, 1, -a/6],
        [-1, -a/6, 1, a/6]
    ], dtype=dtype)

    # c3 matrix (4x4)
    c3 = np.array([
        [6/(5*a), 1/10, -6/(5*a), 1/10],
        [1/10, 2*a/15, -1/10, -a/30],
        [-6/(5*a), -1/10, 6/(5*a), -1/10],
        [1/10, -a/30, -1/10, 2*a/15]
    ], dtype=dtype)

    # c4 matrix (4x4)
    c4 = np.array([
        [13*a/35, 11*a**2/210, 9*a/70, -13*a**2/420],
        [11*a**2/210, a**3/105, 13*a**2/420, -a**3/140],
        [9*a/70, 13*a**2/420, 13*a/35, -11*a**2/210],
        [-13*a**2/420, -a**3/140, -11*a**2/210, a**3/105]
    ], dtype=dtype)

    # Extract needed components from I_bar
    I11 = I_bar[0, 0]
    I12 = I_bar[0, 2]
    I13 = I_bar[0, 3]
    I22 = I_bar[2, 2]
    I23 = I_bar[2, 3]
    I33 = I_bar[3, 3]

    # Initialize the element mass matrix
    M_e = np.zeros((8, 8), dtype=dtype)

    # Build the matrix blocks
    u_indices = np.array([0, 1])
    w_indices = np.array([2, 3, 4, 5])
    psi_indices = np.array([6, 7])

    # u-u block
    M_e[np.ix_(u_indices, u_indices)] = I11 * c1

    # u-w block
    M_e[np.ix_(u_indices, w_indices)] = -I12 * c2
    M_e[np.ix_(w_indices, u_indices)] = (-I12 * c2).T

    # u-psi block
    M_e[np.ix_(u_indices, psi_indices)] = I13 * c1
    M_e[np.ix_(psi_indices, u_indices)] = (I13 * c1).T

    # w-w block
    M_e[np.ix_(w_indices, w_indices)] = I22 * c3 + I11 * c4

    # w-psi block
    M_e[np.ix_(w_indices, psi_indices)] = -I23 * c2.T
    M_e[np.ix_(psi_indices, w_indices)] = (-I23 * c2.T).T

    # psi-psi block
    M_e[np.ix_(psi_indices, psi_indices)] = I33 * c1

    return M_e

def zigzag_assemble_global_mass_matrix(x_coords, element_lengths):
    """Assemble the global mass matrix for the non-uniform mesh"""
    num_elements = len(element_lengths)

    # Determine the size of the global matrix
    num_nodes = num_elements + 1
    size_per_node = 4  # u, w, w_x, psi
    size_global = num_nodes * size_per_node

    # Initialize the global mass matrix as sparse
    M = lil_matrix((size_global, size_global), dtype=dtype)

    # For each element
    for e in range(num_elements):
        # Use the specific length and midpoint for this element
        L_e = element_lengths[e]
        x_mid = x_coords[e] + L_e / 2.0

        # Get beam properties at this position
        z0, z1, z2, z3 = zigzag_get_beam_properties_at_x(x_mid, h, notch_center, notch_width, notch_depth)

        # Calculate zigzag functions for this position
        R_k, dR_k_dz = zigzag_compute_zigzag_functions(z0, z1, z2, z3, Q55)

        # Calculate inertia integrals
        I_bar = zigzag_compute_inertia_integrals(rho, b, z0, z1, z2, z3, R_k)

        # Calculate element mass matrix for this element's length
        M_e = zigzag_compute_element_mass_matrix(I_bar, L_e)

        # Global DOF indices for this element
        node1 = e
        node2 = e + 1

        # Map from local to global DOFs
        global_dofs = np.array([
            node1 * size_per_node + 0,    # u1
            node2 * size_per_node + 0,    # u2
            node1 * size_per_node + 1,    # w1
            node1 * size_per_node + 2,    # w1'
            node2 * size_per_node + 1,    # w2
            node2 * size_per_node + 2,    # w2'
            node1 * size_per_node + 3,    # psi1
            node2 * size_per_node + 3     # psi2
        ], dtype=int)

        # Add the element mass matrix to the global mass matrix
        for i in range(8):
            for j in range(8):
                M[global_dofs[i], global_dofs[j]] += M_e[i, j]

    return M.tocsr()

def zigzag_get_response_at_specific_point(x_coords, u_full, target_x, target_z):
    """Extract response at a specific point through the thickness (Zigzag Theory)"""
    # Find node closest to target_x
    r_node = np.argmin(np.abs(x_coords - target_x))

    # Extract DOFs at the node
    node_idx = r_node
    u0 = u_full[node_idx*4, :]   # Axial displacement at mid-plane
    w0_x = u_full[node_idx*4+2, :] # Rotation (w0,x)
    psi0 = u_full[node_idx*4+3, :]   # Shear rotation

    # Get beam properties at this position
    z0, z1, z2, z3 = zigzag_get_beam_properties_at_x(x_coords[r_node], h, notch_center, notch_width, notch_depth)

    # Calculate zigzag function for this position
    R_k, _ = zigzag_compute_zigzag_functions(z0, z1, z2, z3, Q55)

    # Calculate zigzag function value at target_z
    R_value = R_k(target_z)

    # Calculate displacement at the specific point through the thickness
    u_at_point = u0 - target_z * w0_x + R_value * psi0

    return u_at_point, x_coords[r_node]

# ============================================================================
# EULER-BERNOULLI BEAM THEORY IMPLEMENTATION
# ============================================================================

def get_cross_section_properties(x, h, b, notch_center, notch_width, notch_depth):
    """Calculate position-dependent cross-sectional properties for classical theories"""
    notch_start = notch_center - notch_width/2
    notch_end = notch_center + notch_width/2

    if notch_start <= x <= notch_end:
        # Notched region - reduced effective height
        h_eff = h - notch_depth
        A = b * h_eff
        I = b * h_eff**3 / 12
    else:
        # Regular region
        A = b * h
        I = b * h**3 / 12

    return A, I

def euler_bernoulli_element_matrices(L_e, A, I, E, rho):
    """
    Compute element stiffness and mass matrices for Euler-Bernoulli beam theory
    Using Hermite cubic shape functions (C¹ continuity)
    """
    # Element stiffness matrix (6x6 for 2 nodes, 3 DOFs per node: u, w, dw/dx)
    K_e = np.zeros((6, 6), dtype=dtype)

    # Axial stiffness (EA/L)
    K_axial = E * A / L_e
    K_e[0, 0] = K_axial
    K_e[1, 1] = K_axial
    K_e[0, 1] = -K_axial
    K_e[1, 0] = -K_axial

    # Bending stiffness (EI/L³ terms)
    K_bend = E * I / L_e**3
    K_e[2, 2] = 12 * K_bend
    K_e[2, 3] = 6 * K_bend * L_e
    K_e[2, 4] = -12 * K_bend
    K_e[2, 5] = 6 * K_bend * L_e

    K_e[3, 2] = 6 * K_bend * L_e
    K_e[3, 3] = 4 * K_bend * L_e**2
    K_e[3, 4] = -6 * K_bend * L_e
    K_e[3, 5] = 2 * K_bend * L_e**2

    K_e[4, 2] = -12 * K_bend
    K_e[4, 3] = -6 * K_bend * L_e
    K_e[4, 4] = 12 * K_bend
    K_e[4, 5] = -6 * K_bend * L_e

    K_e[5, 2] = 6 * K_bend * L_e
    K_e[5, 3] = 2 * K_bend * L_e**2
    K_e[5, 4] = -6 * K_bend * L_e
    K_e[5, 5] = 4 * K_bend * L_e**2

    # Element mass matrix (consistent mass matrix)
    M_e = np.zeros((6, 6), dtype=dtype)

    # Axial mass (ρAL/6)
    M_axial = rho * A * L_e / 6
    M_e[0, 0] = 2 * M_axial
    M_e[1, 1] = 2 * M_axial
    M_e[0, 1] = M_axial
    M_e[1, 0] = M_axial

    # Bending and rotational mass
    M_bend = rho * A * L_e / 420
    M_rot = rho * I / L_e

    # Transverse mass
    M_e[2, 2] = 156 * M_bend
    M_e[2, 3] = 22 * M_bend * L_e
    M_e[2, 4] = 54 * M_bend
    M_e[2, 5] = -13 * M_bend * L_e

    M_e[3, 2] = 22 * M_bend * L_e
    M_e[3, 3] = 4 * M_bend * L_e**2
    M_e[3, 4] = 13 * M_bend * L_e
    M_e[3, 5] = -3 * M_bend * L_e**2

    M_e[4, 2] = 54 * M_bend
    M_e[4, 3] = 13 * M_bend * L_e
    M_e[4, 4] = 156 * M_bend
    M_e[4, 5] = -22 * M_bend * L_e

    M_e[5, 2] = -13 * M_bend * L_e
    M_e[5, 3] = -3 * M_bend * L_e**2
    M_e[5, 4] = -22 * M_bend * L_e
    M_e[5, 5] = 4 * M_bend * L_e**2

    # Rotational inertia (diagonal terms)
    M_e[3, 3] += 2 * M_rot * L_e / 15
    M_e[5, 5] += 2 * M_rot * L_e / 15

    return K_e, M_e

def euler_bernoulli_assemble_matrices(x_coords, element_lengths):
    """Assemble global stiffness and mass matrices for Euler-Bernoulli beam"""
    num_elements = len(element_lengths)
    num_nodes = num_elements + 1
    dofs_per_node = 3  # u, w, dw/dx
    total_dofs = num_nodes * dofs_per_node

    K = lil_matrix((total_dofs, total_dofs), dtype=dtype)
    M = lil_matrix((total_dofs, total_dofs), dtype=dtype)

    for e in range(num_elements):
        L_e = element_lengths[e]
        x_mid = x_coords[e] + L_e / 2.0

        # Get cross-sectional properties at element midpoint
        A, I = get_cross_section_properties(x_mid, h, b, notch_center, notch_width, notch_depth)

        # Compute element matrices
        K_e, M_e = euler_bernoulli_element_matrices(L_e, A, I, E, rho)

        # Global DOF indices
        node1_dofs = [e * dofs_per_node + i for i in range(dofs_per_node)]
        node2_dofs = [(e + 1) * dofs_per_node + i for i in range(dofs_per_node)]
        global_dofs = node1_dofs + node2_dofs

        # Assemble into global matrices
        for i in range(6):
            for j in range(6):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]
                M[global_dofs[i], global_dofs[j]] += M_e[i, j]

    return K.tocsr(), M.tocsr()

def euler_bernoulli_get_response(x_coords, u_full, target_x, target_z):
    """Extract response at specific point for Euler-Bernoulli beam theory"""
    # Find node closest to target_x
    r_node = np.argmin(np.abs(x_coords - target_x))

    # Extract DOFs at the node
    node_idx = r_node
    u0 = u_full[node_idx*3, :]     # Axial displacement
    w0 = u_full[node_idx*3 + 1, :] # Transverse displacement
    w0_x = u_full[node_idx*3 + 2, :] # Rotation (dw/dx)

    # Calculate displacement at specific point through thickness
    # u(x,z,t) = u₀(x,t) - z * ∂w₀/∂x(x,t)
    u_at_point = u0 - target_z * w0_x

    return u_at_point, x_coords[r_node]

# ============================================================================
# TIMOSHENKO BEAM THEORY IMPLEMENTATION
# ============================================================================

def timoshenko_element_matrices(L_e, A, I, E, G, rho, kappa=5/6):
    """
    Compute element stiffness and mass matrices for Timoshenko beam theory
    Using linear shape functions with reduced integration to avoid shear locking
    """
    # Element stiffness matrix (6x6 for 2 nodes, 3 DOFs per node: u, w, phi)
    K_e = np.zeros((6, 6), dtype=dtype)

    # Axial stiffness
    K_axial = E * A / L_e
    K_e[0, 0] = K_axial
    K_e[1, 1] = K_axial
    K_e[0, 1] = -K_axial
    K_e[1, 0] = -K_axial

    # Bending stiffness
    K_bend = E * I / L_e**3
    K_e[2, 2] = 4 * K_bend * L_e**2
    K_e[2, 4] = 2 * K_bend * L_e**2
    K_e[2, 5] = -6 * K_bend * L_e

    K_e[3, 2] = -6 * K_bend * L_e
    K_e[3, 3] = 12 * K_bend
    K_e[3, 4] = 6 * K_bend * L_e
    K_e[3, 5] = -12 * K_bend

    K_e[4, 2] = 2 * K_bend * L_e**2
    K_e[4, 3] = 6 * K_bend * L_e
    K_e[4, 4] = 4 * K_bend * L_e**2
    K_e[4, 5] = -6 * K_bend * L_e

    K_e[5, 2] = -6 * K_bend * L_e
    K_e[5, 3] = -12 * K_bend
    K_e[5, 4] = -6 * K_bend * L_e
    K_e[5, 5] = 12 * K_bend

    # Shear stiffness (with shear correction factor)
    K_shear = kappa * G * A / L_e
    K_e[3, 3] += K_shear
    K_e[3, 5] += -K_shear
    K_e[5, 3] += -K_shear
    K_e[5, 5] += K_shear

    # Element mass matrix (consistent mass matrix)
    M_e = np.zeros((6, 6), dtype=dtype)

    # Axial mass
    M_axial = rho * A * L_e / 6
    M_e[0, 0] = 2 * M_axial
    M_e[1, 1] = 2 * M_axial
    M_e[0, 1] = M_axial
    M_e[1, 0] = M_axial

    # Transverse and rotational mass
    M_trans = rho * A * L_e / 6
    M_rot = rho * I / L_e

    M_e[3, 3] = 2 * M_trans
    M_e[3, 5] = M_trans
    M_e[5, 3] = M_trans
    M_e[5, 5] = 2 * M_trans

    # Rotational inertia (diagonal terms)
    M_e[2, 2] = 2 * M_rot * L_e / 3
    M_e[4, 4] = 2 * M_rot * L_e / 3
    M_e[2, 4] = M_rot * L_e / 3
    M_e[4, 2] = M_rot * L_e / 3

    return K_e, M_e

def timoshenko_assemble_matrices(x_coords, element_lengths):
    """Assemble global stiffness and mass matrices for Timoshenko beam"""
    num_elements = len(element_lengths)
    num_nodes = num_elements + 1
    dofs_per_node = 3  # u, w, phi
    total_dofs = num_nodes * dofs_per_node

    K = lil_matrix((total_dofs, total_dofs), dtype=dtype)
    M = lil_matrix((total_dofs, total_dofs), dtype=dtype)

    for e in range(num_elements):
        L_e = element_lengths[e]
        x_mid = x_coords[e] + L_e / 2.0

        # Get cross-sectional properties at element midpoint
        A, I = get_cross_section_properties(x_mid, h, b, notch_center, notch_width, notch_depth)

        # Compute element matrices
        K_e, M_e = timoshenko_element_matrices(L_e, A, I, E, G, rho)

        # Global DOF indices
        node1_dofs = [e * dofs_per_node + i for i in range(dofs_per_node)]
        node2_dofs = [(e + 1) * dofs_per_node + i for i in range(dofs_per_node)]
        global_dofs = node1_dofs + node2_dofs

        # Assemble into global matrices
        for i in range(6):
            for j in range(6):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]
                M[global_dofs[i], global_dofs[j]] += M_e[i, j]

    return K.tocsr(), M.tocsr()

def timoshenko_get_response(x_coords, u_full, target_x, target_z):
    """Extract response at specific point for Timoshenko beam theory"""
    # Find node closest to target_x
    r_node = np.argmin(np.abs(x_coords - target_x))

    # Extract DOFs at the node
    node_idx = r_node
    u0 = u_full[node_idx*3, :]     # Axial displacement
    w0 = u_full[node_idx*3 + 1, :] # Transverse displacement
    phi = u_full[node_idx*3 + 2, :] # Shear rotation

    # Calculate displacement at specific point through thickness
    # u(x,z,t) = u₀(x,t) - z * φ(x,t)
    u_at_point = u0 - target_z * phi

    return u_at_point, x_coords[r_node]

# ============================================================================
# COMMON FUNCTIONS
# ============================================================================

def create_non_uniform_mesh(L, notch_center, notch_width, coarse_elements_on_L=target_elements, remove_smallest_element=True):
    """
    Creates a non-uniform mesh with refinement around the notch
    """
    L_total = L

    # Create a coarse uniform mesh over the total length
    num_coarse_elements = coarse_elements_on_L
    initial_nodes = np.linspace(0, L_total, num_coarse_elements + 1, dtype=dtype)
    coarse_dx = L_total / num_coarse_elements

    # Define the exact coordinates for the notch boundaries
    notch_start_coord = notch_center - notch_width / 2
    notch_end_coord = notch_center + notch_width / 2

    # Combine initial nodes with notch nodes
    all_nodes = np.unique(np.concatenate([initial_nodes, [notch_start_coord, notch_end_coord]]))

    # Merge nodes that are too close together
    merge_tolerance = coarse_dx * 0.1

    # Sort the nodes before merging
    sorted_nodes = np.sort(all_nodes)

    merged_nodes = [sorted_nodes[0]]
    for i in range(1, len(sorted_nodes)):
        # If the distance to the last accepted node is greater than the tolerance, keep it
        if (sorted_nodes[i] - merged_nodes[-1]) > merge_tolerance:
            merged_nodes.append(sorted_nodes[i])

    final_nodes = np.array(merged_nodes, dtype=dtype)

    if len(final_nodes) < len(all_nodes):
        print(f"Merged {len(all_nodes) - len(final_nodes)} node(s) due to proximity.")

    # Calculate element lengths and the minimum length
    element_lengths = np.diff(final_nodes)
    min_dx = np.min(element_lengths) if len(element_lengths) > 0 else 0

    # Remove smallest element to allow larger time steps
    if remove_smallest_element and len(element_lengths) > 3:
        min_idx = np.argmin(element_lengths)
        print(f"Removing smallest element (index {min_idx}, length {min_dx*1e6:.2f} μm) to allow larger time steps")

        # Remove the node that creates the smallest element
        if min_idx == 0:  # Remove first node (except boundary)
            final_nodes = np.delete(final_nodes, 1)
        elif min_idx == len(element_lengths) - 1:  # Remove last node (except boundary)
            final_nodes = np.delete(final_nodes, -2)
        else:  # Remove middle node
            final_nodes = np.delete(final_nodes, min_idx + 1)

        # Recalculate element lengths
        element_lengths = np.diff(final_nodes)
        new_min_dx = np.min(element_lengths)
        print(f"New minimum element size: {new_min_dx*1e6:.2f} μm (was {min_dx*1e6:.2f} μm)")
        min_dx = new_min_dx

    # Ensure no zero-length elements were created
    if min_dx < 1e-9:
        raise ValueError("Mesh generation resulted in a zero-length element after merging.")

    return final_nodes, element_lengths, min_dx

def memory_efficient_time_integration(M, K, C, dt, num_steps, excitation_signal,
                                     active_samples, d1_node, d2_node, theory_type, save_interval=500):
    """Memory-efficient time integration using Newmark method"""
    n = M.shape[0]
    device = M.device

    save_indices = list(range(0, num_steps, save_interval))
    if num_steps - 1 not in save_indices:
        save_indices.append(num_steps - 1)
    save_steps = len(save_indices)
    u_saved = torch.zeros((n, save_steps), dtype=torch_dtype, device=device)

    u = torch.zeros(n, dtype=torch_dtype, device=device)
    v = torch.zeros(n, dtype=torch_dtype, device=device)
    a = torch.zeros(n, dtype=torch_dtype, device=device)

    # Newmark parameters
    beta = 0.3025; gamma = 0.6
    a0 = 1.0/(beta*dt**2); a1 = gamma/(beta*dt); a2 = 1.0/(beta*dt)
    a3 = 1.0/(2.0*beta)-1.0; a4 = gamma/beta-1.0; a5 = dt*(gamma/(2.0*beta)-1.0)

    K_eff = K + a0 * M + a1 * C

    try:
        K_eff_LU = torch.linalg.lu_factor(K_eff)
        use_lu = True
    except:
        print(f"LU decomposition failed for {theory_type}, using direct solve...")
        use_lu = False

    F = torch.zeros(n, dtype=torch_dtype, device=device)
    save_counter = 0

    # Pre-allocate temporary tensors for better memory efficiency
    temp_vec1 = torch.zeros(n, dtype=torch_dtype, device=device)
    temp_vec2 = torch.zeros(n, dtype=torch_dtype, device=device)
    F_eff = torch.zeros(n, dtype=torch_dtype, device=device)

    print(f"Starting {theory_type} time integration...")
    with torch.no_grad():
        for step in range(num_steps):
            if step < active_samples:
                F.zero_()
                if d1_node < n and d1_node >= 0:
                    F[d1_node] = excitation_signal[step]
                    if d1_node + 2 < n: F[d1_node + 2] = -excitation_signal[step] * (h/2)
                if d2_node < n and d2_node >= 0:
                    F[d2_node] = -excitation_signal[step]
                    if d2_node + 2 < n: F[d2_node + 2] = excitation_signal[step] * (h/2)
            else:
                F.zero_()

            # Optimize matrix-vector operations
            torch.mv(M, a0*u + a2*v + a3*a, out=temp_vec1)
            torch.mv(C, a1*u + a4*v + a5*a, out=temp_vec2)
            torch.add(F, temp_vec1, out=F_eff)
            F_eff.add_(temp_vec2)

            if use_lu:
                u_new = torch.linalg.lu_solve(K_eff_LU[0], K_eff_LU[1], F_eff.unsqueeze(1)).squeeze(1)
            else:
                u_new = torch.linalg.solve(K_eff, F_eff)

            # Vectorized updates
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

            if step in save_indices:
                u_saved[:, save_counter] = u
                save_counter += 1

            u, v, a = u_new, v_new, a_new

            # Reduce print frequency for better performance
            if (step + 1) % 10000 == 0:
                print(f"Step {step + 1}/{num_steps} ({(step + 1)/num_steps*100:.1f}%)")

    return u_saved, save_indices

def run_zigzag_simulation(x_coords, element_lengths, dt, num_steps, active_samples, excitation_signal, device):
    """Run Zigzag beam theory simulation"""
    print("\n" + "="*50)
    print("ZIGZAG BEAM THEORY SIMULATION")
    print("="*50)

    # Assemble matrices
    K_sparse = zigzag_assemble_global_stiffness_matrix(x_coords, element_lengths)
    M_sparse = zigzag_assemble_global_mass_matrix(x_coords, element_lengths)
    C_sparse = None  # No damping

    # Convert to tensors
    M_torch = torch.tensor(M_sparse.toarray(), dtype=torch_dtype, device=device)
    K_torch = torch.tensor(K_sparse.toarray(), dtype=torch_dtype, device=device)
    C_torch = torch.zeros_like(K_torch, dtype=torch_dtype, device=device)

    # Clean up
    del K_sparse, M_sparse
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Find excitation nodes
    x = torch.tensor(x_coords, device=device, dtype=torch_dtype)
    d1_node = torch.argmin(torch.abs(x - 1.6535))
    d2_node = torch.argmin(torch.abs(x - 1.6465))
    d1_node_adj = d1_node * 4
    d2_node_adj = d2_node * 4

    # Run time integration
    u_saved, save_indices = memory_efficient_time_integration(
        M_torch, K_torch, C_torch, dt, num_steps, excitation_signal, active_samples,
        d1_node_adj, d2_node_adj, "Zigzag", save_interval=max(1, num_steps // 500)
    )

    # Convert to numpy and extract response
    u_full_saved = u_saved.cpu().numpy()
    u_response, actual_x = zigzag_get_response_at_specific_point(x_coords, u_full_saved, response_x, response_z)

    # Clean up GPU memory
    if device.type == 'cuda':
        del M_torch, K_torch, C_torch, u_saved
        torch.cuda.empty_cache()

    return u_response, actual_x, save_indices

def run_euler_bernoulli_simulation(x_coords, element_lengths, dt, num_steps, active_samples, excitation_signal, device):
    """Run Euler-Bernoulli beam theory simulation"""
    print("\n" + "="*50)
    print("EULER-BERNOULLI BEAM THEORY SIMULATION")
    print("="*50)

    # Assemble matrices
    K_sparse, M_sparse = euler_bernoulli_assemble_matrices(x_coords, element_lengths)
    C_sparse = None  # No damping

    # Convert to tensors
    M_torch = torch.tensor(M_sparse.toarray(), dtype=torch_dtype, device=device)
    K_torch = torch.tensor(K_sparse.toarray(), dtype=torch_dtype, device=device)
    C_torch = torch.zeros_like(K_torch, dtype=torch_dtype, device=device)

    # Clean up
    del K_sparse, M_sparse
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Find excitation nodes
    x = torch.tensor(x_coords, device=device, dtype=torch_dtype)
    d1_node = torch.argmin(torch.abs(x - 1.6535))
    d2_node = torch.argmin(torch.abs(x - 1.6465))
    d1_node_adj = d1_node * 3  # 3 DOFs per node for EB
    d2_node_adj = d2_node * 3

    # Run time integration
    u_saved, save_indices = memory_efficient_time_integration(
        M_torch, K_torch, C_torch, dt, num_steps, excitation_signal, active_samples,
        d1_node_adj, d2_node_adj, "Euler-Bernoulli", save_interval=max(1, num_steps // 500)
    )

    # Convert to numpy and extract response
    u_full_saved = u_saved.cpu().numpy()
    u_response, actual_x = euler_bernoulli_get_response(x_coords, u_full_saved, response_x, response_z)

    # Clean up GPU memory
    if device.type == 'cuda':
        del M_torch, K_torch, C_torch, u_saved
        torch.cuda.empty_cache()

    return u_response, actual_x, save_indices

def run_timoshenko_simulation(x_coords, element_lengths, dt, num_steps, active_samples, excitation_signal, device):
    """Run Timoshenko beam theory simulation"""
    print("\n" + "="*50)
    print("TIMOSHENKO BEAM THEORY SIMULATION")
    print("="*50)

    # Assemble matrices
    K_sparse, M_sparse = timoshenko_assemble_matrices(x_coords, element_lengths)
    C_sparse = None  # No damping

    # Convert to tensors
    M_torch = torch.tensor(M_sparse.toarray(), dtype=torch_dtype, device=device)
    K_torch = torch.tensor(K_sparse.toarray(), dtype=torch_dtype, device=device)
    C_torch = torch.zeros_like(K_torch, dtype=torch_dtype, device=device)

    # Clean up
    del K_sparse, M_sparse
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Find excitation nodes
    x = torch.tensor(x_coords, device=device, dtype=torch_dtype)
    d1_node = torch.argmin(torch.abs(x - 1.6535))
    d2_node = torch.argmin(torch.abs(x - 1.6465))
    d1_node_adj = d1_node * 3  # 3 DOFs per node for Timoshenko
    d2_node_adj = d2_node * 3

    # Run time integration
    u_saved, save_indices = memory_efficient_time_integration(
        M_torch, K_torch, C_torch, dt, num_steps, excitation_signal, active_samples,
        d1_node_adj, d2_node_adj, "Timoshenko", save_interval=max(1, num_steps // 500)
    )

    # Convert to numpy and extract response
    u_full_saved = u_saved.cpu().numpy()
    u_response, actual_x = timoshenko_get_response(x_coords, u_full_saved, response_x, response_z)

    # Clean up GPU memory
    if device.type == 'cuda':
        del M_torch, K_torch, C_torch, u_saved
        torch.cuda.empty_cache()

    return u_response, actual_x, save_indices

def run_comprehensive_comparison():
    """Run comprehensive comparison of all three beam theories"""
    print("="*80)
    print("COMPREHENSIVE BEAM THEORY COMPARISON")
    print("Zigzag vs Euler-Bernoulli vs Timoshenko")
    print("="*80)
    print(f"Beam Configuration:")
    print(f"  Length: {L} m, Height: {h} m, Width: {b} m")
    print(f"  Material: Aluminum (E={E/1e9:.1f} GPa, ρ={rho} kg/m³)")
    print(f"Notch Configuration:")
    print(f"  Center: {notch_center} m, Width: {notch_width*1000:.1f} mm, Depth: {notch_depth*1000:.1f} mm")
    print(f"Response Location: ({response_x}, {response_z}) m")
    print("="*80)

    # Generate mesh
    print("\nGenerating non-uniform mesh...")
    x_coords, element_lengths, min_dx = create_non_uniform_mesh(L, notch_center, notch_width)
    num_elements = len(element_lengths)
    num_nodes = len(x_coords)
    print(f"Mesh: {num_nodes} nodes, {num_elements} elements")
    print(f"Minimum element size: {min_dx * 1e3:.4f} mm")

    # Calculate time step
    c_wave = np.sqrt(E / rho)
    CFL = 0.7
    dt = CFL * min_dx / c_wave
    num_steps = int(total_duration / dt)
    active_samples = int(num_steps * active_duration / total_duration)

    print(f"\nTime discretization:")
    print(f"  Wave speed: {c_wave:.2f} m/s")
    print(f"  Time step: {dt * 1e9:.2f} ns")
    print(f"  Total steps: {num_steps}")

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Generate excitation signal
    def generate_excitation(t_steps, active_samples, frequency, device):
        t_active = torch.linspace(0, active_duration, active_samples, dtype=torch_dtype, device=device)
        n = torch.arange(active_samples, dtype=torch_dtype, device=device)
        hanning = 0.5 * (1 - torch.cos(2 * torch.pi * n / (active_samples - 1)))
        active_signal = hanning * torch.sin(2 * torch.pi * frequency * t_active)
        return active_signal

    excitation_signal = generate_excitation(num_steps, active_samples, frequency, device)

    # Run all three simulations
    results = {}

    # Zigzag
    u_zigzag, x_zigzag, save_indices = run_zigzag_simulation(
        x_coords, element_lengths, dt, num_steps, active_samples, excitation_signal, device
    )
    results['Zigzag'] = u_zigzag

    # Euler-Bernoulli
    u_eb, x_eb, _ = run_euler_bernoulli_simulation(
        x_coords, element_lengths, dt, num_steps, active_samples, excitation_signal, device
    )
    results['Euler-Bernoulli'] = u_eb

    # Timoshenko
    u_tim, x_tim, _ = run_timoshenko_simulation(
        x_coords, element_lengths, dt, num_steps, active_samples, excitation_signal, device
    )
    results['Timoshenko'] = u_tim

    # Process and interpolate all results to uniform time grid
    saved_times = np.array(save_indices) * dt
    time_array = np.linspace(0, total_duration, num_steps)
    time_us = time_array * 1e6  # Convert to microseconds

    processed_results = {}

    for theory, u_response in results.items():
        if len(u_response.shape) > 1:
            u_response = u_response.flatten()

        from scipy.interpolate import interp1d
        interp_func = interp1d(saved_times, u_response, kind='linear', bounds_error=False, fill_value=0)
        u_full_response = interp_func(time_array)

        # Normalize response
        max_abs_val = np.max(np.abs(u_full_response))
        if max_abs_val > 1e-12:
            u_normalized = u_full_response / max_abs_val
        else:
            u_normalized = u_full_response

        processed_results[theory] = u_normalized

    # Save individual CSV files
    for theory, u_normalized in processed_results.items():
        results_df = pd.DataFrame({
            'time_us': time_us,
            'displacement': np.round(u_normalized, 8)
        })

        output_file = f'Claude_res/{theory.lower().replace("-", "_")}_beam_results.csv'
        results_df.to_csv(output_file, index=False, float_format='%.8f')
        print(f"\n✅ {theory} results saved to: {output_file}")

    # Create comparison plot
    print("\n" + "="*50)
    print("CREATING COMPARISON PLOT")
    print("="*50)

    plt.figure(figsize=(14, 8))

    colors = {'Zigzag': 'blue', 'Euler-Bernoulli': 'red', 'Timoshenko': 'green'}
    linestyles = {'Zigzag': '-', 'Euler-Bernoulli': '--', 'Timoshenko': ':'}

    for theory, u_normalized in processed_results.items():
        plt.plot(time_us, u_normalized,
                color=colors[theory],
                linestyle=linestyles[theory],
                linewidth=2,
                label=f'{theory} Theory',
                alpha=0.8)

    plt.xlabel('Time (μs)', fontsize=14)
    plt.ylabel('Normalized Displacement', fontsize=14)
    plt.title(f'Beam Theory Comparison at x={response_x}m, z={response_z}m\n(Notch: center={notch_center}m, width={notch_width*1000:.1f}mm, depth={notch_depth*1000:.1f}mm)',
              fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save comparison plot
    plot_file = 'Claude_res/beam_theory_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {plot_file}")

    # Also save combined CSV
    combined_df = pd.DataFrame({'time_us': time_us})
    for theory, u_normalized in processed_results.items():
        combined_df[f'displacement_{theory.lower().replace("-", "_")}'] = np.round(u_normalized, 8)

    combined_file = 'Claude_res/combined_beam_theory_results.csv'
    combined_df.to_csv(combined_file, index=False, float_format='%.8f')
    print(f"📊 Combined results saved to: {combined_file}")

    plt.close()

    return processed_results

if __name__ == "__main__":
    results = run_comprehensive_comparison()
    print("\n🎯 All simulations completed successfully!")
    print("📁 Results saved in Claude_res directory:")
    print("   - zigzag_beam_results.csv")
    print("   - euler_bernoulli_beam_results.csv")
    print("   - timoshenko_beam_results.csv")
    print("   - combined_beam_theory_results.csv")
    print("   - beam_theory_comparison.png")