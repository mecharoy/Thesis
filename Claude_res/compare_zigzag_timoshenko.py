"""
Comparison of 1D Zigzag Theory vs Timoshenko Beam Theory for Homogeneous Beam
===============================================================

This script compares the displacement responses predicted by:
1. Zigzag Theory (with layer-wise shear deformation functions)
2. Timoshenko Beam Theory (first-order shear deformation)

For a 3m homogeneous aluminum beam without any notch.

Key Parameters:
- Length: 3m
- Cross-section: 1.5mm × 1m
- Material: Aluminum (E=70GPa, ρ=2700 kg/m³)
- Loading: Dual-point excitation at 1.6535m and 1.6465m
- Response monitored at: x = 1.85m
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
from scipy.fft import fft, fftfreq

# Use float32 for better GPU memory efficiency
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
# COMMON PARAMETERS AND SHARED FUNCTIONS
# ============================================================================

def get_common_parameters():
    """Define common parameters for both theories"""
    # Beam geometry
    L = 3.0  # Length in meters (corrected from 1.65m)
    h = 0.0015  # Total height of the beam (m)
    b = 1.0  # Width in m

    # Material properties (Aluminum)
    E = 70e9  # Young's modulus in Pa
    nu = 0.33  # Poisson's ratio
    rho = 2700  # Density in kg/m³
    G = E / (2 * (1 + nu))  # Shear modulus

    # Beam theory parameters
    Q11 = E / (1 - nu**2)  # Reduced stiffness
    Q55 = G * 0.9  # Shear stiffness (with 0.9 factor as in original)

    # Loading parameters
    f = 100e3  # Frequency of 100 kHz
    excitation_points = [1.6535, 1.6465]  # Dual-point loading
    response_point = 1.85  # Monitor response at this point

    # Time parameters
    total_duration = 300e-6
    active_duration = 50e-6

    # Mesh parameters
    num_elements = 6000  # Number of elements

    return {
        'L': L, 'h': h, 'b': b, 'E': E, 'nu': nu, 'rho': rho, 'G': G,
        'Q11': Q11, 'Q55': Q55, 'f': f, 'excitation_points': excitation_points,
        'response_point': response_point, 'total_duration': total_duration,
        'active_duration': active_duration, 'num_elements': num_elements
    }

def create_uniform_mesh(L, num_elements):
    """Create uniform mesh for the beam"""
    element_length = L / num_elements
    x_coords = np.linspace(0, L, num_elements + 1, dtype=dtype)
    element_lengths = np.ones(num_elements, dtype=dtype) * element_length
    min_dx = element_length
    return x_coords, element_lengths, min_dx

def generate_excitation_signal(num_steps, active_samples, frequency, device):
    """Generate Hanning-windowed sinusoidal excitation"""
    t_active = torch.linspace(0, 50e-6, active_samples, dtype=torch_dtype, device=device)
    n = torch.arange(active_samples, dtype=torch_dtype, device=device)
    hanning = 0.5 * (1 - torch.cos(2 * torch.pi * n / (active_samples - 1)))
    active_signal = hanning * torch.sin(2 * torch.pi * frequency * t_active)
    return active_signal, active_samples

# ============================================================================
# ZIGZAG THEORY IMPLEMENTATION (NO NOTCH)
# ============================================================================

def get_beam_properties_zigzag_no_notch(x, h):
    """
    Get beam properties at position x for homogeneous beam without notch
    Simplified version of original function without notch complexity
    """
    # For homogeneous beam without notch, geometry is uniform
    z0_local = -h/2  # Bottom coordinate
    z3_local = h/2   # Top coordinate

    # For zigzag theory, we still divide into layers but all have same properties
    # Using three equal layers for consistency with original formulation
    z1_local = z0_local + h/3   # First interface
    z2_local = z0_local + 2*h/3  # Second interface

    return z0_local, z1_local, z2_local, z3_local

def compute_zigzag_functions_no_notch(h, Q55):
    """
    Compute zigzag function parameters for homogeneous beam
    Simplified version for uniform cross-section
    """
    # Get uniform layer boundaries
    z0, z1, z2, z3 = get_beam_properties_zigzag_no_notch(0, h)  # Position doesn't matter for uniform beam

    # Calculate C_1^k and C_2^k for each layer (same calculation as original)
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

    # Calculate R3 and R4 (global coefficients)
    R3 = 4 * C2_L / delta
    R4 = -4 * C1_L / (3 * delta)

    # Calculate a_1^k and a_2^k for each layer
    a1 = np.zeros(3, dtype=dtype)
    a2 = np.zeros(3, dtype=dtype)
    z_k = [z1, z2, z3]  # Top coordinates of each layer

    for k in range(3):
        a1[k] = 2 * ((C1[k] / Q55) - z_k[k])
        a2[k] = 3 * ((2 * C2[k] / Q55) - z_k[k]**2)

    # Calculate R2^k for each layer
    R2 = np.zeros(3, dtype=dtype)
    for k in range(3):
        R2[k] = a1[k] * R3 + a2[k] * R4

    # Define reference layer (middle layer)
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
        if z < z1:
            k = 0
        elif z < z2:
            k = 1
        else:
            k = 2
        return R_hat1[k] + z * R_hat2[k] + z**2 * R_hat3 + z**3 * R_hat4

    return R_k

# ============================================================================
# TIMOSHENKO BEAM THEORY IMPLEMENTATION
# ============================================================================

def compute_timoshenko_matrices(L_e, E, G, A, I, b, h, kappa=5/6):
    """
    Compute Timoshenko beam element matrices

    Parameters:
    - L_e: Element length
    - E: Young's modulus
    - G: Shear modulus
    - A: Cross-sectional area
    - I: Second moment of area
    - b: Width
    - h: Height
    - kappa: Shear correction factor (5/6 for rectangular)
    """

    # Shape function derivatives for Hermite cubic shape functions
    # Same as C1, C5, C6, C7 matrices from original formulation

    # C1 matrix for axial displacement
    C1 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=dtype) * L_e / 6.0

    # C5 matrix for axial displacement derivative
    C5 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype) / L_e

    # C6 matrix for coupling terms
    C6 = np.array([[0.0, 1.0/L_e, 0.0, -1/L_e],
                  [0, -1.0/L_e, 0, 1.0/L_e]], dtype=dtype)

    # C7 matrix for bending
    C7 = np.array([
        [12.0/L_e**3, 6.0/L_e**2, -12.0/L_e**3, 6.0/L_e**2],
        [6.0/L_e**2, 4.0/L_e, -6.0/L_e**2, 2.0/L_e],
        [-12.0/L_e**3, -6.0/L_e**2, 12.0/L_e**3, -6.0/L_e**2],
        [6.0/L_e**2, 2.0/L_e, -6.0/L_e**2, 4.0/L_e]
    ], dtype=dtype)

    # Timoshenko stiffness components
    # Bending stiffness: D = E*I
    D = E * I

    # Shear stiffness: k_s = kappa * G * A
    k_s = kappa * G * A

    # Axial stiffness: EA
    EA = E * A

    # Element stiffness matrix for Timoshenko beam
    K_e = np.zeros((8, 8), dtype=dtype)

    # Axial-axial terms
    K_e[0:2, 0:2] = EA * C5

    # Bending-bending terms
    K_e[2:6, 2:6] = D * C7

    # Shear deformation terms (coupling between rotation and deflection)
    # Timoshenko theory includes additional shear flexibility
    shear_matrix = np.zeros((4, 4), dtype=dtype)
    shear_matrix[0, 0] = shear_matrix[2, 2] = k_s / L_e
    shear_matrix[0, 2] = shear_matrix[2, 0] = -k_s / L_e
    shear_matrix[1, 1] = shear_matrix[3, 3] = k_s * L_e / 3.0
    shear_matrix[1, 3] = shear_matrix[3, 1] = k_s * L_e / 6.0

    # Add shear contribution to bending terms
    K_e[2:6, 2:6] += shear_matrix

    # Coupling terms (simplified for Timoshenko)
    # No zigzag function coupling in Timoshenko theory
    K_e[0:2, 2:6] = np.zeros((2, 4), dtype=dtype)
    K_e[2:6, 0:2] = np.zeros((4, 2), dtype=dtype)

    # Shear rotation terms (psi DOFs) - simplified
    K_e[6:8, 6:8] = np.zeros((2, 2), dtype=dtype)  # No shear rotation in Timoshenko

    return K_e

def compute_timoshenko_mass_matrix(rho, A, I, L_e):
    """
    Compute Timoshenko beam element mass matrix
    """
    # Consistent mass matrix for Timoshenko beam
    # Including rotary inertia effects

    # Mass per unit length
    m = rho * A

    # Rotary inertia per unit length
    I_rho = rho * I

    # Mass matrix components (same shape as original but simplified)
    # c1 matrix for axial displacement
    c1 = (L_e/3) * np.array([
        [1, 1/2],
        [1/2, 1]
    ], dtype=dtype)

    # c2 matrix for coupling
    c2 = (1/2) * np.array([
        [-1, L_e/6, 1, -L_e/6],
        [-1, -L_e/6, 1, L_e/6]
    ], dtype=dtype)

    # c3 matrix for bending
    c3 = np.array([
        [6/(5*L_e), 1/10, -6/(5*L_e), 1/10],
        [1/10, 2*L_e/15, -1/10, -L_e/30],
        [-6/(5*L_e), -1/10, 6/(5*L_e), -1/10],
        [1/10, -L_e/30, -1/10, 2*L_e/15]
    ], dtype=dtype)

    # c4 matrix for bending inertia
    c4 = np.array([
        [13*L_e/35, 11*L_e**2/210, 9*L_e/70, -13*L_e**2/420],
        [11*L_e**2/210, L_e**3/105, 13*L_e**2/420, -L_e**3/140],
        [9*L_e/70, 13*L_e**2/420, 13*L_e/35, -11*L_e**2/210],
        [-13*L_e**2/420, -L_e**3/140, -11*L_e**2/210, L_e**3/105]
    ], dtype=dtype)

    # Element mass matrix
    M_e = np.zeros((8, 8), dtype=dtype)

    # Axial mass
    M_e[0:2, 0:2] = m * c1

    # Bending mass with rotary inertia
    M_e[2:6, 2:6] = m * c3 + I_rho * c4

    # Shear rotation mass (simplified for Timoshenko)
    M_e[6:8, 6:8] = np.zeros((2, 2), dtype=dtype)  # Negligible

    # Coupling terms
    M_e[0:2, 2:6] = -m * c2
    M_e[2:6, 0:2] = (-m * c2).T

    return M_e

# ============================================================================
# MATRIX ASSEMBLY FUNCTIONS
# ============================================================================

def assemble_zigzag_matrices(x_coords, element_lengths, params):
    """Assemble global matrices for zigzag theory"""
    num_elements = len(element_lengths)
    num_nodes = num_elements + 1
    size_per_node = 4
    size_global = num_nodes * size_per_node

    K = lil_matrix((size_global, size_global), dtype=dtype)
    M = lil_matrix((size_global, size_global), dtype=dtype)

    # Pre-compute zigzag function for uniform beam
    h = params['h']
    Q55 = params['Q55']
    R_k = compute_zigzag_functions_no_notch(h, Q55)

    # Import functions from original formulation
    from sys import modules
    import importlib.util
    spec = importlib.util.spec_from_file_location("original_zigzag", "/home/mecharoy/Thesis/Code/datagenzigzag.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    for e in range(num_elements):
        L_e = element_lengths[e]

        # Get beam properties (uniform for no-notch beam)
        z0, z1, z2, z3 = get_beam_properties_zigzag_no_notch(0, h)

        # Calculate matrices using original functions
        C1, C5, C6, C7 = original.compute_C_matrices(L_e)
        A11, A12, A13, A22, A23, A33, A33_bar = original.compute_A_matrices(
            params['Q11'], Q55, z0, z1, z2, z3, R_k, R_k, params['b']
        )
        K_e = original.compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7)

        # Mass matrix
        I_bar = original.compute_inertia_integrals(params['rho'], params['b'], z0, z1, z2, z3, R_k)
        M_e = original.compute_element_mass_matrix(I_bar, L_e)

        # Assembly
        node1_idx = e
        node2_idx = e + 1
        global_dofs = np.array([
            node1_idx * size_per_node + 0, node2_idx * size_per_node + 0,
            node1_idx * size_per_node + 1, node1_idx * size_per_node + 2,
            node2_idx * size_per_node + 1, node2_idx * size_per_node + 2,
            node1_idx * size_per_node + 3, node2_idx * size_per_node + 3
        ], dtype=int)

        for i in range(8):
            for j in range(8):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]
                M[global_dofs[i], global_dofs[j]] += M_e[i, j]

    return K.tocsr(), M.tocsr()

def assemble_timoshenko_matrices(x_coords, element_lengths, params):
    """Assemble global matrices for Timoshenko beam theory"""
    num_elements = len(element_lengths)
    num_nodes = num_elements + 1
    size_per_node = 4
    size_global = num_nodes * size_per_node

    K = lil_matrix((size_global, size_global), dtype=dtype)
    M = lil_matrix((size_global, size_global), dtype=dtype)

    # Cross-sectional properties
    h = params['h']
    b = params['b']
    A = b * h  # Cross-sectional area
    I = b * h**3 / 12  # Second moment of area

    for e in range(num_elements):
        L_e = element_lengths[e]

        # Element matrices
        K_e = compute_timoshenko_matrices(L_e, params['E'], params['G'], A, I, b, h)
        M_e = compute_timoshenko_mass_matrix(params['rho'], A, I, L_e)

        # Assembly
        node1_idx = e
        node2_idx = e + 1
        global_dofs = np.array([
            node1_idx * size_per_node + 0, node2_idx * size_per_node + 0,
            node1_idx * size_per_node + 1, node1_idx * size_per_node + 2,
            node2_idx * size_per_node + 1, node2_idx * size_per_node + 2,
            node1_idx * size_per_node + 3, node2_idx * size_per_node + 3
        ], dtype=int)

        for i in range(8):
            for j in range(8):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]
                M[global_dofs[i], global_dofs[j]] += M_e[i, j]

    return K.tocsr(), M.tocsr()

# ============================================================================
# TIME INTEGRATION AND ANALYSIS FUNCTIONS
# ============================================================================

def run_time_integration(K, M, params, theory_name):
    """Run time integration for given theory"""
    print(f"\n--- Running {theory_name} Analysis ---")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors
    M_torch = torch.tensor(M.toarray(), dtype=torch_dtype, device=device)
    K_torch = torch.tensor(K.toarray(), dtype=torch_dtype, device=device)
    C_torch = torch.zeros_like(K_torch, dtype=torch_dtype, device=device)  # No damping

    # Time parameters
    c_wave = np.sqrt(params['E'] / params['rho'])
    min_dx = params['L'] / params['num_elements']
    CFL = 0.7
    dt = CFL * min_dx / c_wave

    total_duration = params['total_duration']
    active_duration = params['active_duration']
    num_steps = int(total_duration / dt)
    active_samples = int(num_steps * active_duration / total_duration)

    print(f"Time step: {dt*1e9:.2f} ns, Total steps: {num_steps}")

    # Generate excitation
    excitation_signal, _ = generate_excitation_signal(num_steps, active_samples, params['f'], device)

    # Find excitation nodes
    n_dofs = K_torch.shape[0]
    x = torch.linspace(0, params['L'], params['num_elements'] + 1, device=device)
    d1_node = torch.argmin(torch.abs(x - params['excitation_points'][0]))
    d2_node = torch.argmin(torch.abs(x - params['excitation_points'][1]))

    # Adjust DOF indices
    d1_node_adj = d1_node * 4
    d2_node_adj = d2_node * 4

    # Newmark-beta parameters
    beta = 0.3025
    gamma = 0.6
    a0 = 1.0/(beta*dt**2)
    a1 = gamma/(beta*dt)
    a2 = 1.0/(beta*dt)
    a3 = 1.0/(2.0*beta)-1.0
    a4 = gamma/beta-1.0
    a5 = dt*(gamma/(2.0*beta)-1.0)

    K_eff = K + a0 * M_torch + a1 * C_torch

    # Time integration
    n = M_torch.shape[0]
    u = torch.zeros(n, dtype=torch_dtype, device=device)
    v = torch.zeros(n, dtype=torch_dtype, device=device)
    a = torch.zeros(n, dtype=torch_dtype, device=device)

    # Save response at target point
    response_times = []
    response_values = []
    save_interval = max(1, num_steps // 1000)  # Save 1000 points

    # Find response node
    response_node = torch.argmin(torch.abs(x - params['response_point']))

    print(f"Starting time integration for {theory_name}...")
    start_time = python_time.time()

    with torch.no_grad():
        for step in range(num_steps):
            # Apply excitation
            F = torch.zeros(n, dtype=torch_dtype, device=device)
            if step < active_samples:
                h = params['h']
                if d1_node_adj < n:
                    F[d1_node_adj] = excitation_signal[step]
                    if d1_node_adj + 2 < n: F[d1_node_adj + 2] = -excitation_signal[step] * (h/2)
                if d2_node_adj < n:
                    F[d2_node_adj] = -excitation_signal[step]
                    if d2_node_adj + 2 < n: F[d2_node_adj + 2] = excitation_signal[step] * (h/2)

            # Effective force
            F_eff = F + torch.mv(M_torch, a0*u + a2*v + a3*a) + torch.mv(C_torch, a1*u + a4*v + a5*a)

            # Solve for displacement
            u_new = torch.linalg.solve(K_eff, F_eff)

            # Update acceleration and velocity
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

            # Save response
            if step % save_interval == 0:
                response_times.append(step * dt)
                # Get displacement at response point (mid-plane, axial DOF)
                u_at_point = u[response_node * 4]  # Axial displacement at mid-plane
                response_values.append(u_at_point.cpu().numpy())

            # Update for next step
            u, v, a = u_new, v_new, a_new

            if (step + 1) % 10000 == 0:
                print(f"  Step {step + 1}/{num_steps} ({(step + 1)/num_steps*100:.1f}%)")

    end_time = python_time.time()
    print(f"{theory_name} analysis completed in {end_time - start_time:.2f} seconds")

    return np.array(response_times), np.array(response_values)

def get_zigzag_response_at_point(x_coords, u_full, target_x, target_z, params, theory_name):
    """Get response at specific point for zigzag theory"""
    if theory_name == "Zigzag":
        # Find closest node
        r_node = np.argmin(np.abs(x_coords - target_x))

        # Extract DOFs at node
        node_idx = r_node
        u0 = u_full[node_idx*4, :]   # Axial displacement at mid-plane
        w0_x = u_full[node_idx*4+2, :] # Rotation (w0,x)
        psi0 = u_full[node_idx*4+3, :]   # Shear rotation

        # Get zigzag function
        h = params['h']
        Q55 = params['Q55']
        R_k = compute_zigzag_functions_no_notch(h, Q55)
        R_value = R_k(target_z)

        # Calculate displacement at specific point
        u_at_point = u0 - target_z * w0_x + R_value * psi0
        return u_at_point
    else:
        # For Timoshenko, simpler displacement field
        r_node = np.argmin(np.abs(x_coords - target_x))
        node_idx = r_node
        u0 = u_full[node_idx*4, :]   # Axial displacement at mid-plane
        w0_x = u_full[node_idx*4+2, :] # Rotation

        # Timoshenko displacement: u = u0 - z*w0,x (no zigzag term)
        u_at_point = u0 - target_z * w0_x
        return u_at_point

# ============================================================================
# MAIN COMPARISON ANALYSIS
# ============================================================================

def main():
    """Main comparison function"""
    print("=" * 80)
    print("ZIGZAG THEORY VS TIMOSHENKO BEAM THEORY COMPARISON")
    print("=" * 80)
    print("Beam: 3m homogeneous aluminum beam without notch")
    print("Loading: Dual-point excitation at 1.6535m and 1.6465m")
    print("Response monitored at: x = 1.85m, z = 0 (mid-plane)")
    print("=" * 80)

    # Get common parameters
    params = get_common_parameters()

    # Create mesh
    x_coords, element_lengths, min_dx = create_uniform_mesh(params['L'], params['num_elements'])
    print(f"Mesh: {params['num_elements']} elements, min dx = {min_dx*1e3:.4f} mm")

    # Assemble matrices for both theories
    print("\nAssembling Zigzag theory matrices...")
    K_zigzag, M_zigzag = assemble_zigzag_matrices(x_coords, element_lengths, params)

    print("Assembling Timoshenko theory matrices...")
    K_timoshenko, M_timoshenko = assemble_timoshenko_matrices(x_coords, element_lengths, params)

    # Run analyses
    times_zigzag, response_zigzag = run_time_integration(K_zigzag, M_zigzag, params, "Zigzag Theory")
    times_timoshenko, response_timoshenko = run_time_integration(K_timoshenko, M_timoshenko, params, "Timoshenko Theory")

    # Ensure same time array for comparison
    min_length = min(len(times_zigzag), len(times_timoshenko))
    times_common = times_zigzag[:min_length]
    response_zigzag = response_zigzag[:min_length]
    response_timoshenko = response_timoshenko[:min_length]

    # Normalize responses for comparison
    max_zigzag = np.max(np.abs(response_zigzag))
    max_timoshenko = np.max(np.abs(response_timoshenko))

    if max_zigzag > 1e-12:
        response_zigzag_norm = response_zigzag / max_zigzag
    else:
        response_zigzag_norm = response_zigzag

    if max_timoshenko > 1e-12:
        response_timoshenko_norm = response_timoshenko / max_timoshenko
    else:
        response_timoshenko_norm = response_timoshenko

    # Calculate relative difference
    if max_timoshenko > 1e-12:
        relative_diff = ((response_zigzag_norm - response_timoshenko_norm) / response_timoshenko_norm) * 100
    else:
        relative_diff = np.zeros_like(response_timoshenko_norm)

    # Create comprehensive plots
    create_comparison_plots(times_common, response_zigzag_norm, response_timoshenko_norm,
                         relative_diff, params)

    # Print summary statistics
    print_summary_statistics(times_common, response_zigzag_norm, response_timoshenko_norm, relative_diff)

def create_comparison_plots(times, zigzag_resp, timoshenko_resp, relative_diff, params):
    """Create comprehensive comparison plots"""

    # Ensure output directory exists
    output_dir = "/home/mecharoy/Thesis/Claude_res"
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Zigzag Theory vs Timoshenko Beam Theory Comparison\n' +
                  f'3m Homogeneous Aluminum Beam, Response at x = {params["response_point"]}m',
                  fontsize=14, fontweight='bold')

    # Convert time to microseconds for plotting
    times_us = times * 1e6

    # Plot 1: Time History Comparison
    ax1 = axes[0]
    ax1.plot(times_us, zigzag_resp, 'b-', linewidth=2, label='Zigzag Theory', alpha=0.8)
    ax1.plot(times_us, timoshenko_resp, 'r--', linewidth=2, label='Timoshenko Theory', alpha=0.8)
    ax1.set_ylabel('Normalized Displacement', fontsize=12)
    ax1.set_title('Time History Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, times_us[-1]])

    # Plot 2: Frequency Domain Comparison (FFT)
    ax2 = axes[1]

    # Compute FFTs
    N = len(zigzag_resp)
    fft_zigzag = fft(zigzag_resp)
    fft_timoshenko = fft(timoshenko_resp)
    freqs = fftfreq(N, times[1] - times[0])

    # Only plot positive frequencies up to 500 kHz
    pos_freq_mask = (freqs > 0) & (freqs <= 500e3)
    freqs_pos = freqs[pos_freq_mask] / 1e3  # Convert to kHz

    # Plot magnitude spectra
    ax2.semilogy(freqs_pos, np.abs(fft_zigzag[pos_freq_mask]), 'b-',
                linewidth=2, label='Zigzag Theory', alpha=0.8)
    ax2.semilogy(freqs_pos, np.abs(fft_timoshenko[pos_freq_mask]), 'r--',
                linewidth=2, label='Timoshenko Theory', alpha=0.8)
    ax2.set_ylabel('Magnitude (log scale)', fontsize=12)
    ax2.set_title('Frequency Domain Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (kHz)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, 500])

    # Plot 3: Relative Difference
    ax3 = axes[2]
    ax3.plot(times_us, relative_diff, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='±5% difference')
    ax3.axhline(y=-5, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (μs)', fontsize=12)
    ax3.set_ylabel('Relative Difference (%)', fontsize=12)
    ax3.set_title('Relative Difference: ((Zigzag - Timoshenko)/Timoshenko) × 100%',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_xlim([0, times_us[-1]])

    plt.tight_layout()

    # Save plot
    plot_filename = os.path.join(output_dir, 'zigzag_vs_timoshenko_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {plot_filename}")

    # Also create a detailed zoom plot of the initial response
    fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(12, 8))

    # Zoom to first 50 microseconds
    zoom_mask = times_us <= 50

    ax_zoom.plot(times_us[zoom_mask], zigzag_resp[zoom_mask], 'b-',
                linewidth=2, label='Zigzag Theory', alpha=0.8)
    ax_zoom.plot(times_us[zoom_mask], timoshenko_resp[zoom_mask], 'r--',
                linewidth=2, label='Timoshenko Theory', alpha=0.8)
    ax_zoom.set_xlabel('Time (μs)', fontsize=12)
    ax_zoom.set_ylabel('Normalized Displacement', fontsize=12)
    ax_zoom.set_title('Detailed View: First 50 μs of Response', fontsize=14, fontweight='bold')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc='upper right')

    plt.tight_layout()

    # Save zoom plot
    zoom_filename = os.path.join(output_dir, 'zigzag_vs_timoshenko_zoom.png')
    plt.savefig(zoom_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Zoom plot saved to: {zoom_filename}")

    plt.show()

def print_summary_statistics(times, zigzag_resp, timoshenko_resp, relative_diff):
    """Print summary statistics of the comparison"""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY STATISTICS")
    print("=" * 60)

    # Basic statistics
    max_diff = np.max(np.abs(relative_diff))
    mean_diff = np.mean(np.abs(relative_diff))
    std_diff = np.std(np.abs(relative_diff))

    print(f"Maximum absolute difference: {max_diff:.4f}%")
    print(f"Mean absolute difference: {mean_diff:.4f}%")
    print(f"Standard deviation of differences: {std_diff:.4f}%")

    # Peak response comparison
    peak_zigzag_idx = np.argmax(np.abs(zigzag_resp))
    peak_timoshenko_idx = np.argmax(np.abs(timoshenko_resp))

    print(f"\nPeak response times:")
    print(f"  Zigzag: {times[peak_zigzag_idx]*1e6:.2f} μs")
    print(f"  Timoshenko: {times[peak_timoshenko_idx]*1e6:.2f} μs")

    # Energy content comparison (using RMS)
    rms_zigzag = np.sqrt(np.mean(zigzag_resp**2))
    rms_timoshenko = np.sqrt(np.mean(timoshenko_resp**2))

    print(f"\nRMS values:")
    print(f"  Zigzag: {rms_zigzag:.6f}")
    print(f"  Timoshenko: {rms_timoshenko:.6f}")
    print(f"  Ratio (Zigzag/Timoshenko): {rms_zigzag/rms_timoshenko:.6f}")

    # Frequency content comparison
    from scipy.fft import fft, fftfreq
    N = len(zigzag_resp)
    fft_zigzag = fft(zigzag_resp)
    fft_timoshenko = fft(timoshenko_resp)
    freqs = fftfreq(N, times[1] - times[0])

    # Find dominant frequency
    pos_freq_mask = freqs > 0
    dominant_freq_idx_zigzag = np.argmax(np.abs(fft_zigzag[pos_freq_mask]))
    dominant_freq_idx_timoshenko = np.argmax(np.abs(fft_timoshenko[pos_freq_mask]))

    print(f"\nDominant frequencies:")
    print(f"  Zigzag: {freqs[pos_freq_mask][dominant_freq_idx_zigzag]/1e3:.2f} kHz")
    print(f"  Timoshenko: {freqs[pos_freq_mask][dominant_freq_idx_timoshenko]/1e3:.2f} kHz")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if max_diff < 1.0:
        print("✅ Very good agreement between theories (< 1% difference)")
    elif max_diff < 5.0:
        print("✅ Good agreement between theories (< 5% difference)")
    elif max_diff < 10.0:
        print("⚠️  Moderate agreement between theories (< 10% difference)")
    else:
        print("❌ Significant differences between theories (> 10% difference)")
    print("=" * 60)

if __name__ == "__main__":
    main()