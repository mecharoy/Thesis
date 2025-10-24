#Comparative analysis of Timoshenko vs Zigzag beam theory for homogeneous beams
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

# Use float32 for better GPU memory efficiency
dtype = np.float32
torch_dtype = torch.float32

# Set default tensor type for better memory efficiency
torch.set_default_dtype(torch_dtype)

# Enable TF32 for faster GPU operations on Ampere GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes

def get_beam_properties_no_notch(x, h):
    """
    Get beam properties at position x for uniform beam (no notch)
    """
    z0_local = -h/2  # Bottom coordinate
    z1_local = z0_local + h/3   # First interface
    z2_local = z0_local + 2*h/3  # Second interface
    z3_local = h/2            # Top coordinate
    return z0_local, z1_local, z2_local, z3_local

def compute_zigzag_functions(z0, z1, z2, z3, Q55):
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
    C1_L = C1[2]  # Last layer
    C2_L = C2[2]
    delta = 4 * z0**2 * C1_L - 8 * z0 * C2_L

    # Calculate R3 and R4 (global coefficients for quadratic and cubic terms)
    R3 = 4 * C2_L / delta
    R4 = -4 * C1_L / (3 * delta)

    # Calculate a_1^k and a_2^k for each layer
    a1 = np.zeros(3, dtype=dtype)
    a2 = np.zeros(3, dtype=dtype)
    z_k = [z1, z2, z3]  # Top coordinates of each layer

    for k in range(3):
        a1[k] = 2 * ((C1[k] / Q55) - z_k[k])
        a2[k] = 3 * ((2 * C2[k] / Q55) - z_k[k]**2)

    # Calculate R2^k for each layer (shear rotation variables)
    R2 = np.zeros(3, dtype=dtype)
    for k in range(3):
        R2[k] = a1[k] * R3 + a2[k] * R4

    # Define the reference layer (middle layer in this case)
    k0 = 1
    R2k0 = abs(R2[k0])

    # Calculate R_bar_2^k for each layer (sum for i=2 to k)
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

    # Define the R^k(z) function with normalized coefficients
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

        # Calculate derivative
        return R_hat2[k] + 2 * R_hat3 * z + 3 * R_hat4 * z**2

    return R_k, dR_k_dz

def compute_C_matrices(L):
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

def compute_A_matrices_zigzag(Q11, Q55, z0, z1, z2, z3, R_k, dRz, b):
    """Compute A matrices for zigzag theory"""
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):
        xi, w = np.polynomial.legendre.leggauss(n_points)
        z_mapped = 0.5 * (z_upper - z_lower) * xi + 0.5 * (z_upper + z_lower)
        f_values = np.array([f(z_val) for z_val in z_mapped], dtype=dtype)
        integral = np.sum(w * f_values)
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

    layer_bounds = [(z0, z1), (z1, z2), (z2, z3)]

    for z_lower, z_upper in layer_bounds:
        for i, integrand in enumerate(integrands):
            A_values[i] += integrate_over_thickness(integrand, z_lower, z_upper)

    # Multiply by width
    A_values *= b
    return tuple(A_values)

def compute_A_matrices_timoshenko(Q11, Q55, z0, z1, z2, z3, b):
    """Compute A matrices for Timoshenko beam theory (no zigzag functions)"""
    # For Timoshenko: u(x,z,t) = u0(x,t) - z*w0,x(x,t) + z*phi(x,t)
    # Where phi is the rotation of the cross-section
    h = z3 - z0

    # Direct analytical integration for uniform beam
    A11 = Q11 * h * b          # ∫Q11*dA
    A12 = 0                    # ∫Q11*z*dA = 0 for symmetric cross-section
    A13 = Q55 * h**3 * b / 12  # ∫Q55*z*dA for Timoshenko
    A22 = Q11 * h**3 * b / 12  # ∫Q11*z^2*dA
    A23 = 0                    # ∫Q11*z*R_k(z)*dA = 0 for Timoshenko
    A33 = Q11 * h**3 * b / 12  # ∫Q11*R_k(z)^2*dA where R_k(z) = z for Timoshenko
    A33_bar = Q55 * h * b      # ∫Q55*(dR/dz)^2*dA where dR/dz = 1 for Timoshenko

    return A11, A12, A13, A22, A23, A33, A33_bar

def compute_element_stiffness_zigzag(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7):
    """Compute the element stiffness matrix using zigzag theory formulation."""
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

def compute_element_stiffness_timoshenko(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7):
    """Compute the element stiffness matrix using Timoshenko beam theory formulation."""
    # Same structure as zigzag but with different A matrices
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

def assemble_global_stiffness_matrix(x_coords, element_lengths, params, theory_type='zigzag'):
    """Assemble global stiffness matrix for either zigzag or timoshenko theory"""
    num_elements = len(element_lengths)
    b = params['b']
    Q11 = params['Q11']
    Q55 = params['Q55']
    h = params['h']

    size_per_node = 4
    num_nodes = num_elements + 1
    size_global = size_per_node * num_nodes

    # Use sparse matrix format for memory efficiency
    K = lil_matrix((size_global, size_global), dtype=dtype)

    for e in range(num_elements):
        # Use the specific length and midpoint for this element
        L_e = element_lengths[e]
        x_mid = x_coords[e] + L_e / 2.0

        # Get beam properties at this position (no notch)
        z0, z1, z2, z3 = get_beam_properties_no_notch(x_mid, h)

        # Calculate C matrices for this element's length
        C1, C5, C6, C7 = compute_C_matrices(L_e)

        if theory_type == 'zigzag':
            # Calculate zigzag functions for this position
            R_k, R_k_z = compute_zigzag_functions(z0, z1, z2, z3, Q55)
            # Calculate A matrices with zigzag
            A11, A12, A13, A22, A23, A33, A33_bar = compute_A_matrices_zigzag(Q11, Q55, z0, z1, z2, z3, R_k, R_k_z, b)
            # Calculate element stiffness matrix with zigzag
            K_e = compute_element_stiffness_zigzag(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7)
        else:  # Timoshenko
            # Calculate A matrices without zigzag
            A11, A12, A13, A22, A23, A33, A33_bar = compute_A_matrices_timoshenko(Q11, Q55, z0, z1, z2, z3, b)
            # Calculate element stiffness matrix for Timoshenko
            K_e = compute_element_stiffness_timoshenko(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7)

        # Global node indices for this element
        node1_idx = e
        node2_idx = e + 1

        global_dofs = np.array([
            node1_idx * size_per_node + 0, # Global index for u0 at node1
            node2_idx * size_per_node + 0, # Global index for u0 at node2
            node1_idx * size_per_node + 1, # Global index for w0 at node1
            node1_idx * size_per_node + 2, # Global index for w0x at node1
            node2_idx * size_per_node + 1, # Global index for w0 at node2
            node2_idx * size_per_node + 2, # Global index for w0x at node2
            node1_idx * size_per_node + 3, # Global index for psi0 at node1
            node2_idx * size_per_node + 3  # Global index for psi0 at node2
        ], dtype=int)

        # Add the element stiffness to the global stiffness using sparse operations
        for i in range(len(global_dofs)):
            for j in range(len(global_dofs)):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]

    return K.tocsr()

def compute_inertia_integrals_zigzag(rho, b, z0, z1, z2, z3, R_k_func):
    """Compute inertia integrals for zigzag theory"""
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):
        xi, w = np.polynomial.legendre.leggauss(n_points)
        z_mapped = 0.5 * (z_upper - z_lower) * xi + 0.5 * (z_upper + z_lower)
        jacobian = 0.5 * (z_upper - z_lower)
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

def compute_inertia_integrals_timoshenko(rho, b, z0, z1, z2, z3):
    """Compute inertia integrals for Timoshenko beam theory"""
    # For Timoshenko: u(x,z,t) = u0(x,t) - z*w0,x(x,t) + z*phi(x,t)
    h = z3 - z0

    # Direct analytical calculation for uniform beam
    I00 = rho * h * b
    I01 = 0  # Symmetric cross-section
    I02 = rho * h**3 * b / 12  # ∫rho*z*dA
    I11 = rho * h**3 * b / 12  # ∫rho*z^2*dA
    I12 = 0  # Symmetric cross-section
    I22 = rho * h**3 * b / 12  # ∫rho*z^2*dA where z = R_k(z) for Timoshenko
    Ihat = I00

    I = np.zeros((4, 4), dtype=dtype)
    I[0, 0] = I00
    I[0, 2] = I[2, 0] = I01
    I[0, 3] = I[3, 0] = I02
    I[2, 2] = I11
    I[2, 3] = I[3, 2] = I12
    I[3, 3] = I22
    I[1, 1] = Ihat

    return I

def compute_element_mass_matrix(I_bar, L_e):
    """Compute the element mass matrix using the inertia integrals and shape functions."""
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
    I11 = I_bar[0, 0]  # I00 in the code
    I12 = I_bar[0, 2]  # I01 in the code
    I13 = I_bar[0, 3]  # I02 in the code
    I22 = I_bar[2, 2]  # I11 in the code
    I23 = I_bar[2, 3]  # I12 in the code
    I33 = I_bar[3, 3]  # I22 in the code

    # Initialize the element mass matrix
    M_e = np.zeros((8, 8), dtype=dtype)

    # Build matrix
    # u-u block
    u_indices = np.array([0, 1])
    M_e[np.ix_(u_indices, u_indices)] = I11 * c1

    # u-w block
    u_indices = np.array([0, 1])
    w_indices = np.array([2, 3, 4, 5])
    M_e[np.ix_(u_indices, w_indices)] = -I12 * c2
    M_e[np.ix_(w_indices, u_indices)] = (-I12 * c2).T

    # u-psi block
    u_indices = np.array([0, 1])
    psi_indices = np.array([6, 7])
    M_e[np.ix_(u_indices, psi_indices)] = I13 * c1
    M_e[np.ix_(psi_indices, u_indices)] = (I13 * c1).T

    # w-w block
    w_indices = np.array([2, 3, 4, 5])
    M_e[np.ix_(w_indices, w_indices)] = I22 * c3 + I11 * c4

    # w-psi block
    w_indices = np.array([2, 3, 4, 5])
    psi_indices = np.array([6, 7])
    M_e[np.ix_(w_indices, psi_indices)] = -I23 * c2.T
    M_e[np.ix_(psi_indices, w_indices)] = (-I23 * c2.T).T

    # psi-psi block
    psi_indices = np.array([6, 7])
    M_e[np.ix_(psi_indices, psi_indices)] = I33 * c1

    return M_e

def assemble_global_mass_matrix(x_coords, element_lengths, params, theory_type='zigzag'):
    """Assemble global mass matrix for either zigzag or timoshenko theory"""
    num_elements = len(element_lengths)
    rho = params['rho']
    b = params['b']
    h = params['h']
    Q55 = params['Q55']

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

        # Get beam properties at this position (no notch)
        z0, z1, z2, z3 = get_beam_properties_no_notch(x_mid, h)

        if theory_type == 'zigzag':
            # Calculate zigzag functions for this position
            R_k, _ = compute_zigzag_functions(z0, z1, z2, z3, Q55)
            # Calculate inertia integrals with zigzag
            I_bar = compute_inertia_integrals_zigzag(rho, b, z0, z1, z2, z3, R_k)
        else:  # Timoshenko
            # Calculate inertia integrals without zigzag
            I_bar = compute_inertia_integrals_timoshenko(rho, b, z0, z1, z2, z3)

        # Calculate element mass matrix for this element's length
        M_e = compute_element_mass_matrix(I_bar, L_e)

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

def get_response_at_specific_point(x_coords, u_full, target_x, target_z, params, theory_type='zigzag'):
    """Get response at specific point for either theory"""
    if torch.is_tensor(x_coords):
        x_coords_np = x_coords.cpu().numpy()
    else:
        x_coords_np = x_coords

    # Find node closest to target_x
    r_node = np.argmin(np.abs(x_coords_np - target_x))

    # Extract DOFs at the node
    node_idx = r_node
    u0 = u_full[node_idx*4, :]   # Axial displacement at mid-plane
    w0_x = u_full[node_idx*4+2, :] # Rotation (w0,x)
    psi0 = u_full[node_idx*4+3, :]   # Shear rotation

    if theory_type == 'zigzag':
        # Get beam properties at this position
        h = params['h']
        Q55 = params['Q55']
        z0, z1, z2, z3 = get_beam_properties_no_notch(x_coords_np[r_node], h)
        # Calculate zigzag function for this position
        R_k, _ = compute_zigzag_functions(z0, z1, z2, z3, Q55)
        # Calculate zigzag function value at target_z
        R_value = R_k(target_z)
    else:  # Timoshenko
        # For Timoshenko, R(z) = z (linear variation)
        R_value = target_z

    # Calculate displacement at the specific point through the thickness
    # u_at_point = u0 - target_z * w0_x + R_value * psi0
    u_at_point = u0 - target_z * w0_x + R_value * psi0

    # Check if output is PyTorch tensor and convert if needed
    if torch.is_tensor(u_at_point):
        return u_at_point.cpu().numpy(), x_coords_np[r_node]
    else:
        return u_at_point, x_coords_np[r_node]

def create_uniform_mesh(L, num_elements=6000):
    """Create a uniform mesh for the beam (no notch)"""
    x_coords = np.linspace(0, L, num_elements + 1, dtype=dtype)
    element_lengths = np.diff(x_coords)
    min_dx = np.min(element_lengths)
    return x_coords, element_lengths, min_dx

def run_comparative_analysis(L=2.0, E=70e9, rho=2700):
    """Run comparative analysis between Zigzag and Timoshenko beam theories"""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: ZIGZAG VS TIMOSHENKO BEAM THEORY")
    print("="*80)

    # Material properties
    nu = 0.33  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus

    # Beam geometry
    h = 0.0015  # Total height of the beam (m)
    b = 1  # Width in m

    # Reduced stiffness matrix components (for isotropic material)
    Q11 = E/(1- nu**2)
    Q55 = G*(0.9)  # Shear correction factor

    # Create parameters dictionary
    params = {
        'b': b, 'h': h, 'Q11': Q11, 'Q55': Q55, 'rho': rho
    }

    print(f"\nBeam Properties:")
    print(f"  Length: {L} m")
    print(f"  Height: {h} m")
    print(f"  Young's Modulus: {E/1e9:.1f} GPa")
    print(f"  Density: {rho} kg/m³")
    print(f"  Shear Modulus: {G/1e9:.1f} GPa")

    # Create uniform mesh (no notch)
    x_coords, element_lengths, min_dx = create_uniform_mesh(L, num_elements=6000)
    num_elements = len(element_lengths)
    num_nodes = len(x_coords)
    print(f"\nMesh: {num_nodes} nodes, {num_elements} elements")
    print(f"Minimum element size: {min_dx * 1e3:.4f} mm")

    # Time parameters
    c_wave = np.sqrt(E / rho)
    CFL = 0.7
    dt = CFL * min_dx / c_wave

    total_duration = 300e-6
    active_duration = 50e-6
    f = 100e3  # Frequency of 100 kHz
    num_steps = int(total_duration/dt)

    print(f"\nTime Discretization:")
    print(f"Wave speed: {c_wave:.2f} m/s")
    print(f"Time step: {dt * 1e9:.2f} ns")
    print(f"Total steps: {num_steps}")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Response points
    response_points = [0.5, 1.0, 1.5, 2.0]  # Multiple points along the beam
    target_z = 0.00075  # Mid-thickness

    results = {}

    # Run analysis for both theories
    for theory_type in ['zigzag', 'timoshenko']:
        print(f"\n{'-'*60}")
        print(f"Running {theory_type.upper()} Beam Theory Analysis")
        print(f"{'-'*60}")

        # Assemble global matrices
        print("Assembling global matrices...")
        K_sparse = assemble_global_stiffness_matrix(x_coords, element_lengths, params, theory_type)
        M_sparse = assemble_global_mass_matrix(x_coords, element_lengths, params, theory_type)

        # No damping for free vibration comparison
        C_sparse = None

        n_dofs = K_sparse.shape[0]

        # Convert to tensors
        M_torch = torch.tensor(M_sparse.toarray(), dtype=torch_dtype, device=device)
        K_torch = torch.tensor(K_sparse.toarray(), dtype=torch_dtype, device=device)
        C_torch = torch.zeros_like(K_torch, dtype=torch_dtype, device=device)

        del K_sparse, M_sparse
        gc.collect()

        # Excitation nodes
        x_torch = torch.tensor(x_coords, device=device, dtype=torch_dtype)
        d1_node = torch.argmin(torch.abs(x_torch - 0.6535))
        d2_node = torch.argmin(torch.abs(x_torch - 0.6465))

        d1_node_adj = d1_node * 4
        d2_node_adj = d2_node * 4

        def generate_excitation(t_steps, active_samples, frequency, device):
            t_active = torch.linspace(0, active_duration, active_samples, dtype=torch_dtype, device=device)
            n = torch.arange(active_samples, dtype=torch_dtype, device=device)
            hanning = 0.5 * (1 - torch.cos(2 * torch.pi * n / (active_samples - 1)))
            active_signal = hanning * torch.sin(2 * torch.pi * frequency * t_active)
            return active_signal

        def time_integration(M, K, C, dt, num_steps, excitation_signal, active_samples, d1_node, d2_node, h):
            n = M.shape[0]
            device = M.device

            u = torch.zeros(n, dtype=torch_dtype, device=device)
            v = torch.zeros(n, dtype=torch_dtype, device=device)
            a = torch.zeros(n, dtype=torch_dtype, device=device)

            # Newmark-beta parameters
            beta = 0.3025; gamma = 0.6
            a0=1.0/(beta*dt**2); a1=gamma/(beta*dt); a2=1.0/(beta*dt); a3=1.0/(2.0*beta)-1.0; a4=gamma/beta-1.0; a5=dt*(gamma/(2.0*beta)-1.0)

            K_eff = K + a0 * M + a1 * C

            try:
                K_eff_LU = torch.linalg.lu_factor(K_eff)
                use_lu = True
            except:
                print("LU decomposition failed, using direct solve...")
                use_lu = False

            F = torch.zeros(n, dtype=torch_dtype, device=device)

            # Store responses at specific points
            responses = {x_point: [] for x_point in response_points}
            time_array = []

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

                # Effective force
                F_eff = F + M @ (a0*u + a2*v + a3*a) + C @ (a1*u + a4*v + a5*a)

                # Solve for displacement
                if use_lu:
                    u_new = torch.linalg.lu_solve(K_eff_LU[0], K_eff_LU[1], F_eff.unsqueeze(1)).squeeze(1)
                else:
                    u_new = torch.linalg.solve(K_eff, F_eff)

                # Update acceleration and velocity
                a_new = a0 * (u_new - u) - a2 * v - a3 * a
                v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

                # Extract responses at specified points
                if step % 100 == 0:  # Save every 100th step
                    current_time = step * dt
                    time_array.append(current_time)

                    for x_point in response_points:
                        u_response, _ = get_response_at_specific_point(
                            x_coords, u, x_point, target_z, params, theory_type
                        )
                        responses[x_point].append(u_response)

                u, v, a = u_new, v_new, a_new

                if (step + 1) % 10000 == 0:
                    print(f"  Step {step + 1}/{num_steps} ({(step + 1)/num_steps*100:.1f}%)")

            return responses, np.array(time_array)

        # Generate excitation
        active_samples = int(num_steps * active_duration / total_duration)
        excitation_signal = generate_excitation(num_steps, active_samples, f, device)

        # Run time integration
        start_time = python_time.time()
        responses, time_array = time_integration(
            M_torch, K_torch, C_torch, dt, num_steps, excitation_signal,
            active_samples, d1_node_adj, d2_node_adj, h
        )
        end_time = python_time.time()

        print(f"{theory_type.capitalize()} analysis completed in {end_time - start_time:.3f} seconds")

        results[theory_type] = {
            'responses': responses,
            'time_array': time_array
        }

        # Clean up GPU memory
        if device.type == 'cuda':
            del M_torch, K_torch, C_torch
            torch.cuda.empty_cache()

    return results, response_points

def plot_comparison_results(results, response_points):
    """Create comprehensive comparison plots"""
    print("\nGenerating comparison plots...")

    # Create output directory
    os.makedirs('/home/mecharoy/Thesis/Claude_res', exist_ok=True)

    # Time array
    time_array = results['zigzag']['time_array'] * 1e6  # Convert to microseconds

    # Plot responses for each point
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Zigzag vs Timoshenko Beam Theory: Displacement Response Comparison', fontsize=16)

    axes = axes.flatten()

    for i, x_point in enumerate(response_points):
        ax = axes[i]

        # Get responses
        zigzag_response = np.array(results['zigzag']['responses'][x_point])
        timoshenko_response = np.array(results['timoshenko']['responses'][x_point])

        # Normalize responses for better comparison
        zigzag_max = np.max(np.abs(zigzag_response))
        timoshenko_max = np.max(np.abs(timoshenko_response))

        if zigzag_max > 1e-12:
            zigzag_norm = zigzag_response / zigzag_max
        else:
            zigzag_norm = zigzag_response

        if timoshenko_max > 1e-12:
            timoshenko_norm = timoshenko_response / timoshenko_max
        else:
            timoshenko_norm = timoshenko_response

        # Plot both responses
        ax.plot(time_array, zigzag_norm, 'b-', linewidth=2, label='Zigzag Theory', alpha=0.8)
        ax.plot(time_array, timoshenko_norm, 'r--', linewidth=2, label='Timoshenko Theory', alpha=0.8)

        # Calculate and display difference
        diff = zigzag_norm - timoshenko_norm
        max_diff = np.max(np.abs(diff))
        relative_error = max_diff / np.max(np.abs(zigzag_norm)) * 100 if np.max(np.abs(zigzag_norm)) > 1e-12 else 0

        ax.set_title(f'x = {x_point} m\nMax Difference: {max_diff:.4f} ({relative_error:.2f}%)', fontsize=12)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Normalized Displacement')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Add secondary y-axis for absolute difference
        ax2 = ax.twinx()
        ax2.plot(time_array, diff, 'g:', linewidth=1, alpha=0.6, label='Difference')
        ax2.set_ylabel('Absolute Difference', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

    plt.tight_layout()

    # Save main comparison plot
    main_plot_path = '/home/mecharoy/Thesis/Claude_res/zigzag_vs_timoshenko_comparison.png'
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"Main comparison plot saved to: {main_plot_path}")

    # Create detailed difference analysis plot
    fig, axes = plt.subplots(1, len(response_points), figsize=(20, 6))
    fig.suptitle('Zigzag - Timoshenko Difference Analysis', fontsize=16)

    if len(response_points) == 1:
        axes = [axes]

    for i, x_point in enumerate(response_points):
        ax = axes[i]

        zigzag_response = np.array(results['zigzag']['responses'][x_point])
        timoshenko_response = np.array(results['timoshenko']['responses'][x_point])

        # Normalize to maximum amplitude
        combined_response = np.concatenate([zigzag_response, timoshenko_response])
        max_amplitude = np.max(np.abs(combined_response))

        if max_amplitude > 1e-12:
            zigzag_norm = zigzag_response / max_amplitude
            timoshenko_norm = timoshenko_response / max_amplitude
        else:
            zigzag_norm = zigzag_response
            timoshenko_norm = timoshenko_response

        diff = zigzag_norm - timoshenko_norm

        ax.plot(time_array, diff, 'g-', linewidth=2)
        ax.fill_between(time_array, 0, diff, alpha=0.3, color='green')
        ax.set_title(f'x = {x_point} m', fontsize=12)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Normalized Difference')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Add statistics
        rms_diff = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))
        ax.text(0.02, 0.98, f'RMS: {rms_diff:.6f}\nMax: {max_diff:.6f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save difference analysis plot
    diff_plot_path = '/home/mecharoy/Thesis/Claude_res/zigzag_timoshenko_differences.png'
    plt.savefig(diff_plot_path, dpi=300, bbox_inches='tight')
    print(f"Difference analysis plot saved to: {diff_plot_path}")

    # Create quantitative comparison table
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON RESULTS")
    print("="*60)

    comparison_data = []
    for x_point in response_points:
        zigzag_response = np.array(results['zigzag']['responses'][x_point])
        timoshenko_response = np.array(results['timoshenko']['responses'][x_point])

        # Normalize to same amplitude
        zigzag_max = np.max(np.abs(zigzag_response))
        timoshenko_max = np.max(np.abs(timoshenko_response))

        # Calculate metrics
        diff = zigzag_response - timoshenko_response
        rms_diff = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))

        # Relative errors
        if zigzag_max > 1e-12:
            relative_rms = (rms_diff / zigzag_max) * 100
            relative_max = (max_diff / zigzag_max) * 100
        else:
            relative_rms = 0
            relative_max = 0

        comparison_data.append({
            'Position (m)': x_point,
            'Zigzag Max Amplitude': zigzag_max,
            'Timoshenko Max Amplitude': timoshenko_max,
            'RMS Difference': rms_diff,
            'Max Difference': max_diff,
            'Relative RMS Error (%)': relative_rms,
            'Relative Max Error (%)': relative_max
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False, float_format='%.6e'))

    # Save comparison data
    data_path = '/home/mecharoy/Thesis/Claude_res/zigzag_timoshenko_comparison_data.csv'
    df.to_csv(data_path, index=False)
    print(f"\nComparison data saved to: {data_path}")

    plt.show()

if __name__ == "__main__":
    # Run comparative analysis
    results, response_points = run_comparative_analysis()

    # Plot comparison results
    plot_comparison_results(results, response_points)