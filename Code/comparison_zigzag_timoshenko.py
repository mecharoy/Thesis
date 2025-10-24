"""
Comparison of Zigzag Theory vs FSDT (Timoshenko) for Uniform Beam
==================================================================

This code compares wave propagation responses between:
1. Full zigzag theory (with R^k(z) function)
2. First-Order Shear Deformation Theory / Timoshenko (R^k(z) = 0)

Both use the same finite element formulation for fair comparison.
Uses NumPy/SciPy for efficient computation without GPU dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu
import gc
import time as python_time
import os
import pandas as pd

# Use float64 for better accuracy
dtype = np.float64

def compute_zigzag_functions(z0, z1, z2, z3, Q55, use_zigzag=True):
    """
    Compute the zigzag function parameters according to the exact theory.

    Parameters:
    -----------
    z0, z1, z2, z3 : float
        Layer interface z-coordinates
    Q55 : float
        Shear stiffness
    use_zigzag : bool
        If False, returns zero zigzag functions (FSDT/Timoshenko)

    Returns:
    --------
    R_k : function
        Zigzag function R^k(z)
    dR_k_dz : function
        Derivative of zigzag function
    """

    if not use_zigzag:
        # Return zero functions for FSDT/Timoshenko comparison
        def R_k(z):
            return 0.0
        def dR_k_dz(z):
            return 0.0
        return R_k, dR_k_dz

    # Full zigzag theory implementation
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

    # Calculate R3 and R4
    R3 = 4 * C2_L / delta
    R4 = -4 * C1_L / (3 * delta)

    # Calculate a_1^k and a_2^k
    a1 = np.zeros(3, dtype=dtype)
    a2 = np.zeros(3, dtype=dtype)
    z_k = [z1, z2, z3]

    for k in range(3):
        a1[k] = 2 * ((C1[k] / Q55) - z_k[k])
        a2[k] = 3 * ((2 * C2[k] / Q55) - z_k[k]**2)

    # Calculate R2^k
    R2 = np.zeros(3, dtype=dtype)
    for k in range(3):
        R2[k] = a1[k] * R3 + a2[k] * R4

    # Reference layer
    k0 = 1
    R2k0 = abs(R2[k0])

    # Calculate R_bar_2^k
    R_bar2 = np.zeros(3, dtype=dtype)
    z_interfaces = [z0, z1, z2, z3]

    for k in range(3):
        for i in range(1, k+1):
            R_bar2[k] += z_interfaces[i-1] * (R2[i-1] - R2[i])

    # Calculate R1^k
    R1 = np.zeros(3, dtype=dtype)
    for k in range(3):
        R1[k] = R_bar2[k] - R_bar2[k0]

    # Normalize coefficients
    R_hat1 = R1 / R2k0
    R_hat2 = R2 / R2k0
    R_hat3 = R3 / R2k0
    R_hat4 = R4 / R2k0

    def R_k(z):
        """Calculate the zigzag function R^k(z)"""
        if z < z1:
            k = 0
        elif z < z2:
            k = 1
        else:
            k = 2
        return R_hat1[k] + z * R_hat2[k] + z**2 * R_hat3 + z**3 * R_hat4

    def dR_k_dz(z):
        """Calculate the derivative of the zigzag function"""
        if z < z1:
            k = 0
        elif z < z2:
            k = 1
        else:
            k = 2
        return R_hat2[k] + 2 * R_hat3 * z + 3 * R_hat4 * z**2

    return R_k, dR_k_dz

def compute_C_matrices(L):
    """Compute element C matrices for FEM"""
    C1 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=dtype) * L / 6.0
    C5 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype) / L
    C6 = np.array([[0.0, 1.0/L, 0.0, -1/L], [0, -1.0/L, 0, 1.0/L]], dtype=dtype)
    C7 = np.array([
        [12.0/L**3, 6.0/L**2, -12.0/L**3, 6.0/L**2],
        [6.0/L**2, 4.0/L, -6.0/L**2, 2.0/L],
        [-12.0/L**3, -6.0/L**2, 12.0/L**3, -6.0/L**2],
        [6.0/L**2, 2.0/L, -6.0/L**2, 4.0/L]
    ], dtype=dtype)

    return C1, C5, C6, C7

def compute_A_matrices(Q11, Q55, z0, z1, z2, z3, R_k, dRz, b):
    """Compute stiffness integrals through the thickness"""
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):
        xi, w = np.polynomial.legendre.leggauss(n_points)
        z_mapped = 0.5 * (z_upper - z_lower) * xi + 0.5 * (z_upper + z_lower)
        f_values = np.array([f(z_val) for z_val in z_mapped], dtype=dtype)
        integral = np.sum(w * f_values)
        integral *= 0.5 * (z_upper - z_lower)
        return integral

    def integrand_A11(z): return Q11
    def integrand_A12(z): return Q11 * z
    def integrand_A13(z): return Q11 * R_k(z)
    def integrand_A22(z): return Q11 * z**2
    def integrand_A23(z): return Q11 * z * R_k(z)
    def integrand_A33(z): return Q11 * R_k(z)**2
    def integrand_A33_bar(z): return Q55 * dRz(z)**2

    A_values = np.zeros(7, dtype=dtype)
    integrands = [integrand_A11, integrand_A12, integrand_A13, integrand_A22,
                  integrand_A23, integrand_A33, integrand_A33_bar]

    layer_bounds = [(z0, z1), (z1, z2), (z2, z3)]

    for z_lower, z_upper in layer_bounds:
        for i, integrand in enumerate(integrands):
            A_values[i] += integrate_over_thickness(integrand, z_lower, z_upper)

    A_values *= b
    return tuple(A_values)

def compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7):
    """Compute the element stiffness matrix"""
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

def assemble_global_stiffness_matrix(num_elements, element_length, params, use_zigzag=True):
    """Assemble global stiffness matrix for uniform beam"""
    b = params['b']
    Q11 = params['Q11']
    Q55 = params['Q55']
    h = params['h']

    size_per_node = 4
    num_nodes = num_elements + 1
    size_global = size_per_node * num_nodes

    K = lil_matrix((size_global, size_global), dtype=dtype)

    # Uniform beam properties
    z0 = -h/2
    z1 = z0 + h/3
    z2 = z0 + 2*h/3
    z3 = h/2

    # Calculate zigzag functions
    R_k, R_k_z = compute_zigzag_functions(z0, z1, z2, z3, Q55, use_zigzag=use_zigzag)

    # Calculate C matrices
    C1, C5, C6, C7 = compute_C_matrices(element_length)

    # Calculate A matrices
    A11, A12, A13, A22, A23, A33, A33_bar = compute_A_matrices(Q11, Q55, z0, z1, z2, z3, R_k, R_k_z, b)

    # Calculate element stiffness matrix
    K_e = compute_element_stiffness(A11, A12, A13, A22, A23, A33, A33_bar, C1, C5, C6, C7)

    for e in range(num_elements):
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

        for i in range(len(global_dofs)):
            for j in range(len(global_dofs)):
                K[global_dofs[i], global_dofs[j]] += K_e[i, j]

    return K.tocsr()

def compute_inertia_integrals(rho, b, z0, z1, z2, z3, R_k_func):
    """Compute inertia integrals for mass matrix"""
    def integrate_over_thickness(f, z_lower, z_upper, n_points=6):
        xi, w = np.polynomial.legendre.leggauss(n_points)
        z_mapped = 0.5 * (z_upper - z_lower) * xi + 0.5 * (z_upper + z_lower)
        jacobian = 0.5 * (z_upper - z_lower)
        f_values = np.array([f(z_val) for z_val in z_mapped], dtype=dtype)
        integral = np.sum(w * f_values)
        return integral * jacobian

    interfaces = [z0, z1, z2, z3]
    I = np.zeros((4, 4), dtype=dtype)

    rho_layer = [rho, rho, rho]

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

    I[0, 0] = I_values[0]
    I[0, 2] = I[2, 0] = I_values[1]
    I[0, 3] = I[3, 0] = I_values[2]
    I[2, 2] = I_values[3]
    I[2, 3] = I[3, 2] = I_values[4]
    I[3, 3] = I_values[5]
    I[1, 1] = I_values[0]

    I *= b
    return I

def compute_element_mass_matrix(I_bar, L_e):
    """Compute the element mass matrix"""
    a = L_e

    c1 = (a/3) * np.array([[1, 1/2], [1/2, 1]], dtype=dtype)

    c2 = (1/2) * np.array([
        [-1, a/6, 1, -a/6],
        [-1, -a/6, 1, a/6]
    ], dtype=dtype)

    c3 = np.array([
        [6/(5*a), 1/10, -6/(5*a), 1/10],
        [1/10, 2*a/15, -1/10, -a/30],
        [-6/(5*a), -1/10, 6/(5*a), -1/10],
        [1/10, -a/30, -1/10, 2*a/15]
    ], dtype=dtype)

    c4 = np.array([
        [13*a/35, 11*a**2/210, 9*a/70, -13*a**2/420],
        [11*a**2/210, a**3/105, 13*a**2/420, -a**3/140],
        [9*a/70, 13*a**2/420, 13*a/35, -11*a**2/210],
        [-13*a**2/420, -a**3/140, -11*a**2/210, a**3/105]
    ], dtype=dtype)

    I11 = I_bar[0, 0]
    I12 = I_bar[0, 2]
    I13 = I_bar[0, 3]
    I22 = I_bar[2, 2]
    I23 = I_bar[2, 3]
    I33 = I_bar[3, 3]

    M_e = np.zeros((8, 8), dtype=dtype)

    u_indices = np.array([0, 1])
    M_e[np.ix_(u_indices, u_indices)] = I11 * c1

    w_indices = np.array([2, 3, 4, 5])
    M_e[np.ix_(u_indices, w_indices)] = -I12 * c2
    M_e[np.ix_(w_indices, u_indices)] = (-I12 * c2).T

    psi_indices = np.array([6, 7])
    M_e[np.ix_(u_indices, psi_indices)] = I13 * c1
    M_e[np.ix_(psi_indices, u_indices)] = (I13 * c1).T

    M_e[np.ix_(w_indices, w_indices)] = I22 * c3 + I11 * c4

    M_e[np.ix_(w_indices, psi_indices)] = -I23 * c2.T
    M_e[np.ix_(psi_indices, w_indices)] = (-I23 * c2.T).T

    M_e[np.ix_(psi_indices, psi_indices)] = I33 * c1

    return M_e

def assemble_global_mass_matrix(num_elements, element_length, params, use_zigzag=True):
    """Assemble global mass matrix for uniform beam"""
    rho = params['rho']
    b = params['b']
    h = params['h']
    Q55 = params['Q55']

    num_nodes = num_elements + 1
    size_per_node = 4
    size_global = num_nodes * size_per_node

    M = lil_matrix((size_global, size_global), dtype=dtype)

    # Uniform beam properties
    z0 = -h/2
    z1 = z0 + h/3
    z2 = z0 + 2*h/3
    z3 = h/2

    # Calculate zigzag functions
    R_k, dR_k_dz = compute_zigzag_functions(z0, z1, z2, z3, Q55, use_zigzag=use_zigzag)

    # Calculate inertia integrals
    I_bar = compute_inertia_integrals(rho, b, z0, z1, z2, z3, R_k)

    # Calculate element mass matrix
    M_e = compute_element_mass_matrix(I_bar, element_length)

    for e in range(num_elements):
        node1 = e
        node2 = e + 1

        global_dofs = np.array([
            node1 * size_per_node + 0,
            node2 * size_per_node + 0,
            node1 * size_per_node + 1,
            node1 * size_per_node + 2,
            node2 * size_per_node + 1,
            node2 * size_per_node + 2,
            node1 * size_per_node + 3,
            node2 * size_per_node + 3
        ], dtype=int)

        for i in range(8):
            for j in range(8):
                M[global_dofs[i], global_dofs[j]] += M_e[i, j]

    return M.tocsr()

def get_response_at_specific_point(x_coords, u_full, target_x, target_z, params, use_zigzag=True):
    """Extract displacement response at a specific point"""
    r_node = np.argmin(np.abs(x_coords - target_x))

    node_idx = r_node
    u0 = u_full[node_idx*4, :]
    w0_x = u_full[node_idx*4+2, :]
    psi0 = u_full[node_idx*4+3, :]

    h = params['h']
    Q55 = params['Q55']

    z0 = -h/2
    z1 = z0 + h/3
    z2 = z0 + 2*h/3
    z3 = h/2

    R_k, _ = compute_zigzag_functions(z0, z1, z2, z3, Q55, use_zigzag=use_zigzag)

    R_value = R_k(target_z)

    u_at_point = u0 - target_z * w0_x + R_value * psi0

    return u_at_point, x_coords[r_node]

def run_wave_propagation_analysis(L, E, rho, num_elements=1000, use_zigzag=True, theory_name=""):
    """
    Run wave propagation analysis with specified theory

    Parameters:
    -----------
    L : float
        Beam length (m)
    E : float
        Young's modulus (Pa)
    rho : float
        Density (kg/m³)
    num_elements : int
        Number of finite elements
    use_zigzag : bool
        True for zigzag theory, False for FSDT
    theory_name : str
        Name of theory for printing
    """
    print(f"\n{'='*70}")
    print(f"Running {theory_name} Analysis")
    print(f"{'='*70}")

    # Material properties
    nu = 0.33
    G = E / (2 * (1 + nu))

    # Beam geometry
    h = 0.0015  # 1.5 mm
    b = 1  # 1 m width

    # Reduced stiffness matrix components
    Q11 = E / (1 - nu**2)
    Q55 = G * 0.9

    params = {
        'b': b, 'h': h, 'Q11': Q11, 'Q55': Q55, 'rho': rho
    }

    # Uniform mesh
    element_length = L / num_elements
    x_coords = np.linspace(0, L, num_elements + 1, dtype=dtype)

    print(f"Mesh: {num_elements} uniform elements, element size: {element_length*1e3:.4f} mm")

    # Calculate time step
    c_wave = np.sqrt(E / rho)
    CFL = 0.7
    dt = CFL * element_length / c_wave

    total_duration = 300e-6
    active_duration = 50e-6
    f = 100e3
    num_steps = int(total_duration / dt)

    print(f"Wave speed: {c_wave:.2f} m/s")
    print(f"Time step: {dt*1e9:.2f} ns")
    print(f"Total steps: {num_steps}")

    # Excitation nodes
    d1_node = np.argmin(np.abs(x_coords - 1.6535))
    d2_node = np.argmin(np.abs(x_coords - 1.6465))

    # Assemble matrices
    print("\nAssembling global matrices...")
    K_sparse = assemble_global_stiffness_matrix(num_elements, element_length, params, use_zigzag=use_zigzag)
    gc.collect()
    M_sparse = assemble_global_mass_matrix(num_elements, element_length, params, use_zigzag=use_zigzag)
    gc.collect()

    n_dofs = K_sparse.shape[0]

    print("Converting sparse matrices to dense for time integration...")
    M_dense = M_sparse.toarray()
    K_dense = K_sparse.toarray()
    C_dense = np.zeros_like(K_dense, dtype=dtype)

    del K_sparse, M_sparse
    gc.collect()

    d1_node_adj = d1_node * 4
    d2_node_adj = d2_node * 4

    def generate_excitation(active_samples, frequency):
        t_active = np.linspace(0, active_duration, active_samples, dtype=dtype)
        n = np.arange(active_samples, dtype=dtype)
        hanning = 0.5 * (1 - np.cos(2 * np.pi * n / (active_samples - 1)))
        active_signal = hanning * np.sin(2 * np.pi * frequency * t_active)
        return active_signal

    def time_integration(M, K, C, dt, num_steps, excitation_signal, active_samples,
                        d1_node, d2_node, h, save_interval=500):
        n = M.shape[0]

        save_indices = list(range(0, num_steps, save_interval))
        if num_steps - 1 not in save_indices:
            save_indices.append(num_steps - 1)
        save_steps = len(save_indices)
        u_saved = np.zeros((n, save_steps), dtype=dtype)

        u = np.zeros(n, dtype=dtype)
        v = np.zeros(n, dtype=dtype)
        a = np.zeros(n, dtype=dtype)

        beta = 0.3025
        gamma = 0.6
        a0 = 1.0/(beta*dt**2)
        a1 = gamma/(beta*dt)
        a2 = 1.0/(beta*dt)
        a3 = 1.0/(2.0*beta) - 1.0
        a4 = gamma/beta - 1.0
        a5 = dt*(gamma/(2.0*beta) - 1.0)

        K_eff = K + a0 * M + a1 * C

        print("Computing LU decomposition of effective stiffness matrix...")
        K_eff_sparse = csr_matrix(K_eff)
        K_eff_LU = splu(K_eff_sparse)

        F = np.zeros(n, dtype=dtype)
        save_counter = 0

        print("\nStarting time integration...")
        for step in range(num_steps):
            if step < active_samples:
                F.fill(0.0)
                if d1_node < n and d1_node >= 0:
                    F[d1_node] = excitation_signal[step]
                    if d1_node + 2 < n:
                        F[d1_node + 2] = -excitation_signal[step] * (h/2)
                if d2_node < n and d2_node >= 0:
                    F[d2_node] = -excitation_signal[step]
                    if d2_node + 2 < n:
                        F[d2_node + 2] = excitation_signal[step] * (h/2)
            else:
                F.fill(0.0)

            temp_vec1 = M @ (a0*u + a2*v + a3*a)
            temp_vec2 = C @ (a1*u + a4*v + a5*a)
            F_eff = F + temp_vec1 + temp_vec2

            u_new = K_eff_LU.solve(F_eff)

            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

            if step in save_indices:
                u_saved[:, save_counter] = u
                save_counter += 1

            u, v, a = u_new, v_new, a_new

            if (step + 1) % 10000 == 0:
                print(f"  Step {step + 1}/{num_steps} ({(step + 1)/num_steps*100:.1f}%)")

        return u_saved, save_indices

    # Generate excitation
    active_samples = int(num_steps * active_duration / total_duration)
    excitation_signal = generate_excitation(active_samples, f)

    # Run time integration
    start_time = python_time.time()
    u_saved, save_indices = time_integration(
        M_dense, K_dense, C_dense, dt, num_steps, excitation_signal, active_samples,
        d1_node_adj, d2_node_adj, h, save_interval=max(1, num_steps // 500)
    )
    end_time = python_time.time()
    print(f"\nTime integration completed in {end_time - start_time:.2f} seconds")

    u_full_saved = u_saved

    del M_dense, K_dense, C_dense, u_saved, excitation_signal
    gc.collect()

    # Extract response at target point
    target_x = 2.0  # meters
    target_z = 0.00075  # mid-thickness

    print(f"\nExtracting response at x={target_x}m, z={target_z}m...")
    u_specific_saved, actual_x = get_response_at_specific_point(
        x_coords, u_full_saved, target_x, target_z, params, use_zigzag=use_zigzag
    )

    # Interpolate to full time array
    from scipy.interpolate import interp1d
    t_np = np.linspace(0, total_duration, num_steps)
    saved_times = np.array(save_indices) * dt

    if len(u_specific_saved.shape) > 1:
        u_specific_saved = u_specific_saved.flatten()

    interp_func = interp1d(saved_times, u_specific_saved, kind='linear', bounds_error=False, fill_value=0)
    u_specific_full = interp_func(t_np)

    return u_specific_full, t_np, actual_x

def main():
    """Main comparison function"""

    print("\n" + "="*70)
    print("ZIGZAG vs FSDT (TIMOSHENKO) COMPARISON FOR UNIFORM BEAM")
    print("="*70)

    # Parameters
    L = 3.0  # m
    E = 70e9  # Pa (70 GPa - Aluminum)
    rho = 2700  # kg/m³
    num_elements = 1000

    print(f"\nBeam Parameters:")
    print(f"  Length: {L} m")
    print(f"  Young's Modulus: {E/1e9:.1f} GPa")
    print(f"  Density: {rho} kg/m³")
    print(f"  Thickness: 1.5 mm")
    print(f"  Width: 1.0 m")

    # Run zigzag analysis
    u_zigzag, time_array, actual_x = run_wave_propagation_analysis(
        L, E, rho, num_elements=num_elements, use_zigzag=True, theory_name="ZIGZAG THEORY"
    )

    # Run FSDT analysis
    u_fsdt, _, _ = run_wave_propagation_analysis(
        L, E, rho, num_elements=num_elements, use_zigzag=False, theory_name="FSDT (TIMOSHENKO)"
    )

    # Create output directory
    output_dir = "/home/mecharoy/Thesis/Claude_res"
    os.makedirs(output_dir, exist_ok=True)

    # Save numerical data
    data_file = os.path.join(output_dir, "comparison_data.csv")
    df = pd.DataFrame({
        'time_us': time_array * 1e6,
        'zigzag_displacement': u_zigzag,
        'fsdt_displacement': u_fsdt,
        'difference': u_zigzag - u_fsdt
    })
    df.to_csv(data_file, index=False)
    print(f"\n✓ Numerical data saved to: {data_file}")

    # Create comparison plot
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOT")
    print("="*70)

    plt.figure(figsize=(14, 8))

    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(time_array * 1e6, u_zigzag, 'b-', linewidth=1.5, label='Zigzag Theory', alpha=0.8)
    plt.plot(time_array * 1e6, u_fsdt, 'r--', linewidth=1.5, label='FSDT (Timoshenko)', alpha=0.8)
    plt.xlabel('Time (μs)', fontsize=12)
    plt.ylabel('Displacement (m)', fontsize=12)
    plt.title(f'Wave Propagation Response at x = {actual_x:.3f} m, z = 0.75 mm (mid-thickness)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Difference plot
    plt.subplot(2, 1, 2)
    difference = u_zigzag - u_fsdt
    plt.plot(time_array * 1e6, difference, 'g-', linewidth=1.2, label='Zigzag - FSDT')
    plt.xlabel('Time (μs)', fontsize=12)
    plt.ylabel('Difference (m)', fontsize=12)
    plt.title('Difference Between Theories', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add statistics text
    max_diff = np.max(np.abs(difference))
    rms_diff = np.sqrt(np.mean(difference**2))
    relative_error = rms_diff / np.max(np.abs(u_zigzag)) * 100 if np.max(np.abs(u_zigzag)) > 0 else 0

    stats_text = f'Max |Diff|: {max_diff:.2e} m\nRMS Diff: {rms_diff:.2e} m\nRelative Error: {relative_error:.3f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, "zigzag_vs_timoshenko_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_file}")

    # Print summary statistics
    print("\n" + "="*70)
    print("COMPARISON STATISTICS")
    print("="*70)
    print(f"Maximum absolute difference: {max_diff:.6e} m")
    print(f"RMS difference: {rms_diff:.6e} m")
    print(f"Relative error: {relative_error:.4f}%")
    print(f"Max zigzag displacement: {np.max(np.abs(u_zigzag)):.6e} m")
    print(f"Max FSDT displacement: {np.max(np.abs(u_fsdt)):.6e} m")

    if relative_error < 0.1:
        print("\n🔍 CONCLUSION: Zigzag and FSDT theories produce nearly IDENTICAL results")
        print("   → The zigzag function provides minimal benefit for uniform homogeneous beams")
    elif relative_error < 1.0:
        print("\n🔍 CONCLUSION: Zigzag and FSDT theories show MINOR differences")
        print("   → The zigzag function has small but measurable effect")
    else:
        print("\n🔍 CONCLUSION: Zigzag and FSDT theories show SIGNIFICANT differences")
        print("   → The zigzag function captures important physics")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
