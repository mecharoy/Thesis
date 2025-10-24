#generates several samples using 2000 elements with adjustible meshing
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

# PML functionality removed - using free-free boundaries

def get_beam_properties_at_x(x, h, notch_center, notch_width, notch_depth):
    """
    Get beam properties at position x with parameterized notch
    Note: coordinates are in original beam coordinates (no PML)
    """
    # No PML offset needed for free-free boundaries
    x_original = x

    z0_local = -h/2  # Bottom coordinate is always the same

    # Calculate notch boundaries in original coordinates
    notch_start = notch_center - notch_width/2
    notch_end = notch_center + notch_width/2

    if notch_start <= x_original <= notch_end:  # Notch region
        # Reduce top coordinate by notch depth
        z3_notch = h/2 - notch_depth  # Top coordinate is reduced
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
    C2[1] = C2[0] + 0.5 * Q55 * (z2 **2 - z1**2)

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

def compute_A_matrices(Q11, Q55, z0, z1, z2, z3, R_k, dRz, b):
    # Reduced integration points for efficiency
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):  # Further reduced for speed
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
    def integrand_A11(z):
        return Q11

    def integrand_A12(z):
        return Q11 * z

    def integrand_A13(z):
        return Q11 * R_k(z)

    def integrand_A22(z):
        return Q11 * z**2

    def integrand_A23(z):
        return Q11 * z * R_k(z)

    def integrand_A33(z):
        return Q11 * R_k(z)**2

    def integrand_A33_bar(z):
        return Q55 * dRz(z)**2

    # Compute A matrices by integration over each layer
    A_values = np.zeros(7, dtype=dtype)  # [A11, A12, A13, A22, A23, A33, A33_bar]

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

def compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7):
    """Compute the element stiffness matrix using the provided formulation."""

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

def assemble_global_stiffness_matrix(x_coords, element_lengths, params):
    num_elements = len(element_lengths)
    b = params['b']
    Q11 = params['Q11']
    Q55 = params['Q55']
    h = params['h']
    notch_center = params['notch_center']
    notch_width = params['notch_width']
    notch_depth = params['notch_depth']

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
        z0, z1, z2, z3 = get_beam_properties_at_x(x_mid, h, notch_center, notch_width, notch_depth)

        # Calculate zigzag functions for this position
        R_k, R_k_z = compute_zigzag_functions(z0, z1, z2, z3, Q55)

        # Calculate C matrices for this element's length
        C1, C5, C6, C7 = compute_C_matrices(L_e)

        # Calculate A matrices
        A11, A12, A13, A22, A23, A33, A33_bar = compute_A_matrices(Q11, Q55, z0, z1, z2, z3, R_k, R_k_z, b)

        # Calculate element stiffness matrix
        K_e = compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7)

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
        ], dtype=int) # Ensure indices are integers

        # Add the element stiffness to the global stiffness using sparse operations
        for i in range(len(global_dofs)):
            for j in range(len(global_dofs)):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]

    return K.tocsr()  # Convert to CSR format for efficient operations

def compute_inertia_integrals(rho, b, z0, z1, z2, z3, R_k_func):
    # Integration helper function with reduced points
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):  # Further reduced for speed
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

    I_values = np.zeros(6, dtype=dtype)  # [I00, I01, I02, I11, I12, I22]
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

def compute_element_mass_matrix(I_bar, L_e):
    """
    Compute the element mass matrix using the inertia integrals and shape functions.
    """
    # Define c1, c2, c3, c4 matrices as given in the equations
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

    # Build the matrix
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

def assemble_global_mass_matrix(x_coords, element_lengths, params):
    """
    Assemble the global mass matrix accounting for the non-uniform mesh.
    """
    num_elements = len(element_lengths)
    rho = params['rho']
    b = params['b']
    h = params['h']
    Q55 = params['Q55']
    notch_center = params['notch_center']
    notch_width = params['notch_width']
    notch_depth = params['notch_depth']

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
        z0, z1, z2, z3 = get_beam_properties_at_x(x_mid, h, notch_center, notch_width, notch_depth)

        # Calculate zigzag functions for this position
        R_k, dR_k_dz = compute_zigzag_functions(z0, z1, z2, z3, Q55)

        # Calculate inertia integrals
        I_bar = compute_inertia_integrals(rho, b, z0, z1, z2, z3, R_k)

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

    return M.tocsr()  # Convert to CSR format for efficient operations
# PML damping matrix assembly removed - using free-free boundaries with zero damping

def get_response_at_specific_point(x_coords, u_full, target_x, target_z, params):
    # Check if input is PyTorch tensor and convert if needed
    if torch.is_tensor(x_coords):
        x_coords_np = x_coords.cpu().numpy()
    else:
        x_coords_np = x_coords

    # Find node closest to target_x (no PML offset needed)
    r_node = np.argmin(np.abs(x_coords_np - target_x))

    # Extract DOFs at the node
    node_idx = r_node
    u0 = u_full[node_idx*4, :]   # Axial displacement at mid-plane
    w0_x = u_full[node_idx*4+2, :] # Rotation (w0,x)
    psi0 = u_full[node_idx*4+3, :]   # Shear rotation

    # Get beam properties at this position
    h = params['h']
    notch_center = params['notch_center']
    notch_width = params['notch_width']
    notch_depth = params['notch_depth']
    Q55 = params['Q55']

    z0, z1, z2, z3 = get_beam_properties_at_x(x_coords_np[r_node], h, notch_center, notch_width, notch_depth)

    # Calculate zigzag function for this position
    R_k, _ = compute_zigzag_functions(z0, z1, z2, z3, Q55)

    # Calculate zigzag function value at target_z
    R_value = R_k(target_z)

    # Calculate displacement at the specific point through the thickness
    u_at_point = u0 - target_z * w0_x + R_value * psi0

    # Check if output is PyTorch tensor and convert if needed
    if torch.is_tensor(u_at_point):
        return u_at_point.cpu().numpy(), x_coords_np[r_node]  # Return original coordinate
    else:
        return u_at_point, x_coords_np[r_node]  # Return original coordinate

def create_non_uniform_mesh(L, notch_center, notch_width, coarse_elements_on_L=100, remove_smallest_element=True):
    """
    Creates a non-uniform mesh with refinement around the notch for free-free boundaries.
    Merges nodes that are too close to prevent stability issues.
    Optionally removes the smallest element to allow larger time steps.
    """
    # For free-free boundaries, total length is just the original length
    L_total = L

    # --- 1. Create a coarse uniform mesh over the total length ---
    num_coarse_elements = coarse_elements_on_L
    initial_nodes = np.linspace(0, L_total, num_coarse_elements + 1, dtype=dtype)
    coarse_dx = L_total / num_coarse_elements

    # --- 2. Define the exact coordinates for the notch boundaries ---
    notch_start_coord = notch_center - notch_width / 2
    notch_end_coord = notch_center + notch_width / 2

    # --- 3. Combine initial nodes with notch nodes ---
    all_nodes = np.unique(np.concatenate([initial_nodes, [notch_start_coord, notch_end_coord]]))

    # --- 4. NEW: Merge nodes that are too close together ---
    # Define a tolerance as a fraction of the coarse element size
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

    # --- 5. Calculate element lengths and the minimum length ---
    element_lengths = np.diff(final_nodes)
    min_dx = np.min(element_lengths) if len(element_lengths) > 0 else 0

    # --- 6. OPTIONAL: Remove smallest element to allow larger time steps ---
    if remove_smallest_element and len(element_lengths) > 3:  # Keep at least 3 elements
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


def run_wave_propagation_analysis_free_free(L, E, rho, notch_center, notch_width, notch_depth, response_points, coarse_elements_on_L=6000):
    """
    Run parametric wave propagation analysis with free-free boundaries and a variable number of elements.
    """
    print(f"\n--- Starting Wave Propagation Analysis with {coarse_elements_on_L} Coarse Elements ---")
    print(f"Beam length: {L} m, Notch center: {notch_center} m, Notch width: {notch_width} m")


    # Material properties
    nu = 0.33  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus

    # Beam geometry
    h = 0.0015  # Total height of the beam (m)
    b = 1  # Width in m

    # Reduced stiffness matrix components (for isotropic material)
    Q11 = E/(1- nu**2)
    Q55 = G*(0.9)  # Shear

    # Create parameters dictionary (no PML needed)
    params = {
        'b': b, 'h': h, 'Q11': Q11, 'Q55': Q55, 'rho': rho,
        'notch_center': notch_center, 'notch_width': notch_width, 'notch_depth': notch_depth
    }

    # --- MESHING & TIME STEP CALCULATION ---
    print("\n--- Generating Non-Uniform Mesh ---")
    # Generate the mesh with refinement around the notch
    x_coords, element_lengths, min_dx = create_non_uniform_mesh(L, notch_center, notch_width, coarse_elements_on_L=coarse_elements_on_L, remove_smallest_element=True)
    num_elements = len(element_lengths)
    num_nodes = len(x_coords)
    print(f"Mesh generated with {num_nodes} nodes and {num_elements} elements.")
    print(f"Minimum element size (min_dx): {min_dx * 1e3:.4f} mm")

    # --- Calculate Time Step based on CFL Condition ---
    # Wave speed (use longitudinal wave speed for a conservative estimate)
    c_wave = np.sqrt(E / rho)
    CFL = 0.7  # Courant number (increased from 0.5 for faster computation, still stable)
    dt = CFL * min_dx / c_wave  # Time step based on smallest element

    # Time parameters
    total_duration = 300e-6
    active_duration = 50e-6
    f = 100e3  # Frequency of 100 kHz
    num_steps = int(total_duration/dt)

    print("\n--- Time Discretization ---")
    print(f"Wave speed estimated at: {c_wave:.2f} m/s")
    print(f"Calculated stable time step (dt): {dt * 1e9:.2f} ns")
    print(f"Total number of time steps: {num_steps}")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    # Generate node positions tensor
    x = torch.tensor(x_coords, device=device, dtype=torch_dtype)

    # Find nodes for excitation (in original coordinates, shifted +1m)
    d1_node = torch.argmin(torch.abs(x - 1.6535))
    d2_node = torch.argmin(torch.abs(x - 1.6465))

    # --- Assemble global matrices using the new non-uniform mesh data ---
    print("\nAssembling global matrices...")
    K_sparse = assemble_global_stiffness_matrix(x_coords, element_lengths, params)
    gc.collect()
    M_sparse = assemble_global_mass_matrix(x_coords, element_lengths, params)
    gc.collect()
    # No damping matrix needed for free-free boundaries
    C_sparse = None

    n_dofs = K_sparse.shape[0]

    print("Converting matrices to dense tensors for solver...")
    # Convert matrices to tensors for GPU processing
    M_torch = torch.tensor(M_sparse.toarray(), dtype=torch_dtype, device=device)
    K_torch = torch.tensor(K_sparse.toarray(), dtype=torch_dtype, device=device)
    # Create zero damping matrix for free-free boundaries
    C_torch = torch.zeros_like(K_torch, dtype=torch_dtype, device=device)

    del K_sparse, M_sparse
    gc.collect()
    # Remove frequent empty_cache calls that slow down GPU operations
    # Only clear cache once after major allocations
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Adjust node indices for excitation
    d1_node_adj = d1_node * 4
    d2_node_adj = d2_node * 4

    def generate_excitation_sparse(t_steps, active_samples, frequency, device):
        t_active = torch.linspace(0, active_duration, active_samples, dtype=torch_dtype, device=device)
        n = torch.arange(active_samples, dtype=torch_dtype, device=device)
        hanning = 0.5 * (1 - torch.cos(2 * torch.pi * n / (active_samples - 1)))
        active_signal = hanning * torch.sin(2 * torch.pi * frequency * t_active)
        return active_signal, active_samples

    def memory_efficient_time_integration_with_damping(M, K, C, dt, num_steps, excitation_signal,
                                                      active_samples, d1_node, d2_node, h, save_interval=500):
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
        save_counter = 0
        
        # Pre-allocate temporary tensors for better memory efficiency
        temp_vec1 = torch.zeros(n, dtype=torch_dtype, device=device)
        temp_vec2 = torch.zeros(n, dtype=torch_dtype, device=device)
        F_eff = torch.zeros(n, dtype=torch_dtype, device=device)

        print("\nStarting optimized time integration...")
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

                # Optimize matrix-vector operations using more efficient operations
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

    # Generate excitation
    active_samples = int(num_steps * active_duration / total_duration)
    excitation_signal, _ = generate_excitation_sparse(num_steps, active_samples, f, device)

    # Run time integration
    start_time = python_time.time()
    u_saved, save_indices = memory_efficient_time_integration_with_damping(
        M_torch, K_torch, C_torch, dt, num_steps, excitation_signal, active_samples,
        d1_node_adj, d2_node_adj, h, save_interval=max(1, num_steps // 500)
    )
    end_time = python_time.time()
    print(f"Time integration completed in {end_time - start_time:.3f} seconds!")

    u_full_saved = u_saved.cpu().numpy()

    if device.type == 'cuda':
        del M_torch, K_torch, C_torch, u_saved, excitation_signal
        # Only clear cache at the end to avoid performance penalties
        torch.cuda.empty_cache()

    # --- Extract and process responses ---
    target_z = 0.00075
    responses = {}
    t_np = np.linspace(0, total_duration, num_steps)
    
    print(f"\nExtracting responses at {len(response_points)} points...")
    for target_x in response_points:
        u_specific_saved, actual_x = get_response_at_specific_point(
            x.cpu().numpy(), u_full_saved, target_x, target_z, params
        )

        from scipy.interpolate import interp1d
        
        saved_times = np.array(save_indices) * dt
        if len(u_specific_saved.shape) > 1: u_specific_saved = u_specific_saved.flatten()

        interp_func = interp1d(saved_times, u_specific_saved, kind='linear', bounds_error=False, fill_value=0)
        u_specific_full = interp_func(t_np)

        responses[target_x] = {
            'displacement': u_specific_full,
            'actual_x': actual_x
        }

    return responses, t_np

def run_batch_wave_propagation_analysis():
    """
    Run wave propagation analysis for all cases, structuring the output
    in a row-based format with a dedicated column for response points.
    """

    # --- Configuration ---
    train_file = "/home/user2/Music/abhi2/datagen/lfsm_train_dataset.csv"
    test_file = "/home/user2/Music/abhi2/datagen/lfsm_test_dataset.csv"

    response_points = [1.85, 1.87, 1.9, 1.92, 1.95, 1.97, 2.0, 2.02, 2.05, 2.07, 2.1]  # Shifted +1m to the right
    TARGET_NUM_POINTS = 1500

    def process_dataset(csv_file, output_file, dataset_name):
        """Process a single dataset (train or test)"""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} dataset: {csv_file}")
        print(f"{'='*60}")

        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} cases from {dataset_name} dataset")
        except FileNotFoundError:
            print(f"Error: The file {csv_file} was not found. Skipping.")
            return None
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            return None

        all_rows = []
        failed_cases = []
        num_time_steps = TARGET_NUM_POINTS

        for idx, row in df.iterrows():
            try:
                original_case_id = int(row['case_id'])
                print(f"\nProcessing Case ID: {original_case_id} ({idx+1}/{len(df)})")

                L = float(row['length'])
                E = float(row['youngs_modulus'])
                rho = float(row['density'])
                notch_center = float(row['notch_x'])
                notch_width = float(row['notch_width'])
                notch_depth = float(row['notch_depth'])

                print(f"  Parameters: L={L}m, E={E/1e9:.1f}GPa, rho={rho}kg/m³")

                start_time = time.time()
                responses, time_array = run_wave_propagation_analysis_free_free(
                    L, E, rho, notch_center, notch_width, notch_depth,
                    response_points=response_points, # Correctly pass the list of points
                    coarse_elements_on_L=6000 # Reduced from 6000 for faster processing
                )
                end_time = time.time()
                print(f"  Analysis completed in {end_time - start_time:.2f} seconds")

                original_time_steps = len(time_array)
                print(f"  Original time steps: {original_time_steps}. Resampling to {TARGET_NUM_POINTS} steps.")

                if original_time_steps > 1:
                    target_time_array = np.linspace(time_array[0], time_array[-1], TARGET_NUM_POINTS)
                else:
                    target_time_array = np.full(TARGET_NUM_POINTS, time_array[0] if original_time_steps == 1 else 0)

                for x_point in response_points:
                    if x_point not in responses:
                        print(f"  Warning: Response for x={x_point} not found in case {original_case_id}. Skipping.")
                        continue

                    displacement_series = responses[x_point]['displacement']

                    if original_time_steps > 1:
                        resampled_series = np.interp(target_time_array, time_array, displacement_series)
                    else:
                        resampled_series = np.full(TARGET_NUM_POINTS, displacement_series[0] if len(displacement_series) > 0 else 0)

                    max_abs_val = np.max(np.abs(resampled_series))
                    normalized_series = resampled_series / max_abs_val if max_abs_val > 1e-12 else resampled_series
                    rounded_series = np.round(normalized_series, 5)

                    # --- MODIFICATION: Construct the new row format ---
                    # The row now contains the original case_id and the response_point in a new column.
                    new_row = [
                        original_case_id,
                        x_point,  # New column for the response point location
                        row['notch_x'],
                        row['notch_depth'],
                        row['notch_width'],
                        row['length'],
                        row['density'],
                        row['youngs_modulus'],
                    ]
                    new_row.extend(rounded_series.tolist())
                    all_rows.append(new_row)

                del responses, time_array
                gc.collect()

            except Exception as e:
                print(f"  ERROR processing case ID {original_case_id}: {e}")
                failed_cases.append({
                    'case_id': original_case_id, 'error': str(e), 'parameters': dict(row)
                })
                continue

        if all_rows:
            # --- MODIFICATION: Define the new column headers ---
            base_columns = [
                'case_id', 'response_point', 'notch_x', 'notch_depth', 'notch_width',
                'length', 'density', 'youngs_modulus'
            ]
            response_columns = [f't_{i+1}' for i in range(num_time_steps)]
            final_columns = base_columns + response_columns

            results_df = pd.DataFrame(all_rows, columns=final_columns)
            results_df.to_csv(output_file, index=False, float_format='%.5f')
            print(f"\n✅ Successfully saved {len(results_df)} rows to {output_file}")
            return results_df
        else:
            print(f"\n❌ No successful results to save for {dataset_name}")

        if failed_cases:
            failed_file = output_file.replace('.csv', '_failed.csv')
            pd.DataFrame(failed_cases).to_csv(failed_file, index=False)
            print(f"⚠️  Saved {len(failed_cases)} failed cases to {failed_file}")

        return None

    # --- Main Execution ---
    print("Starting batch wave propagation analysis...")
    print(f"Target number of points per response: {TARGET_NUM_POINTS}")

    train_output = "LFSM6000train.csv"
    train_results = process_dataset(train_file, train_output, "TRAIN")

    test_output = "LFSM6000test.csv"
    test_results = process_dataset(test_file, test_output, "TEST")

    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETED!")
    print(f"{'='*60}")

    if train_results is not None:
        print(f"✅ Train results saved to: {train_output}")
        print(f"   - Shape: {train_results.shape}")

    if test_results is not None:
        print(f"✅ Test results saved to: {test_output}")
        print(f"   - Shape: {test_results.shape}")

    return train_results, test_results

def plot_initial_responses(train_df, test_df, num_samples=5, response_point=1.85):
    """
    Plots the displacement responses for the first few samples from the train and test sets.
    """
    if train_df is None and test_df is None:
        print("\nNo data available to plot.")
        return

    print(f"\n📊 Plotting responses for the first {num_samples} samples at x = {response_point} m...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Displacement Response at x = {response_point} m', fontsize=16)

    time_cols = [col for col in train_df.columns if col.startswith('t_')]
    time_axis = np.linspace(0, 300, len(time_cols)) # Time in microseconds

    # --- Plot Training Data ---
    
    if train_df is not None:
        ax = axes[0]
        train_subset = train_df[train_df['response_point'] == response_point]
        
        # Get unique case_ids and take the first num_samples
        unique_cases = train_subset['case_id'].unique()[:num_samples]

        for case_id in unique_cases:
            case_data = train_subset[train_subset['case_id'] == case_id]
            if not case_data.empty:
                response = case_data[time_cols].iloc[0].values
                ax.plot(time_axis, response, label=f'Train Case {case_id}')
        
        ax.set_title('Training Set Responses')
        ax.set_ylabel('Normalized Displacement')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # --- Plot Testing Data ---
    if test_df is not None:
        ax = axes[1]
        test_subset = test_df[test_df['response_point'] == response_point]
        
        # Get unique case_ids and take the first num_samples
        unique_cases = test_subset['case_id'].unique()[:num_samples]
        
        for case_id in unique_cases:
            case_data = test_subset[test_subset['case_id'] == case_id]
            if not case_data.empty:
                response = case_data[time_cols].iloc[0].values
                ax.plot(time_axis, response, label=f'Test Case {case_id}')

        ax.set_title('Test Set Responses')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Normalized Displacement')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    train_df, test_df = run_batch_wave_propagation_analysis()
    plot_initial_responses(train_df, test_df)