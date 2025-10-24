#part 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
import pandas as pd
import logging
import torch
import os
from typing import Tuple, List, Dict
import json
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def apply_pytorch_fix():
    """Apply fixes for PyTorch memory allocator issues"""
    # Remove problematic environment variables
    problematic_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_LAUNCH_BLOCKING'
    ]
    
    for var in problematic_vars:
        if var in os.environ:
            del os.environ[var]
    
    # Set safe CUDA memory fraction if using GPU
    if torch.cuda.is_available():
        try:
            # Enable TF32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        except Exception as e:
            logging.warning(f"Could not set CUDA memory fraction: {e}")
    
    logging.info("Applied PyTorch memory allocator fixes")

def get_cache_path(L, E, rho, nx, ny, element_type):
    """Generate a unique cache filename based on mesh parameters"""
    # Create a hash of the parameters
    param_str = f"{L}_{E}_{rho}_{nx}_{ny}_{element_type}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    cache_dir = "./fem_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    return {
        'K_path': f"{cache_dir}/K_matrix_{param_hash}.npz",
        'M_path': f"{cache_dir}/M_matrix_{param_hash}.npz",
        'mesh_path': f"{cache_dir}/mesh_data_{param_hash}.npz",
        'params_path': f"{cache_dir}/params_{param_hash}.json"
    }

def save_cache_params(params_path, L, E, rho, nx, ny, element_type, b, h, nu):
    """Save parameters to JSON file"""
    params = {
        'L': L, 'E': E, 'rho': rho, 'nx': nx, 'ny': ny,
        'element_type': element_type, 'b': b, 'h': h, 'nu': nu
    }
    with open(params_path, 'w') as f:
        json.dump(params, f)


def get_safe_device():
    """Get a safe device for computation, defaulting to CPU if GPU has issues"""
    try:
        if torch.cuda.is_available():
            # Test if GPU works properly
            test_tensor = torch.tensor([1.0], device='cuda')
            return torch.device('cuda')
    except RuntimeError as e:
        logging.warning(f"GPU initialization failed: {e}")
    
    logging.info("Using CPU for computation")
    return torch.device('cpu')


def check_cache_validity(params_path, L, E, rho, nx, ny, element_type):
    """Check if cached parameters match current parameters"""
    if not os.path.exists(params_path):
        return False
    
    with open(params_path, 'r') as f:
        cached_params = json.load(f)
    
    return (cached_params['L'] == L and 
            cached_params['E'] == E and 
            cached_params['rho'] == rho and
            cached_params['nx'] == nx and
            cached_params['ny'] == ny and
            cached_params['element_type'] == element_type)

# [KEEP UNCHANGED] - Put gauss_quadrature() function here
def gauss_quadrature(n):
    """Generate Gauss quadrature points and weights on [-1, 1]"""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        points = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    elif n == 4:
        p1 = np.sqrt(3/7 - 2/7 * np.sqrt(6/5))
        p2 = np.sqrt(3/7 + 2/7 * np.sqrt(6/5))
        w1 = (18 + np.sqrt(30)) / 36
        w2 = (18 - np.sqrt(30)) / 36
        points = np.array([-p2, -p1, p1, p2])
        weights = np.array([w2, w1, w1, w2])
    else:
        # For higher orders, use numpy's polynomial roots
        from numpy.polynomial.legendre import leggauss
        points, weights = leggauss(n)
    
    return points, weights


class Mesh2D:
    def __init__(self, domain, nx, ny, element_type='bilinear'):
        """Create 2D FEM mesh - Modified to NOT exclude notch elements"""
        self.domain = domain
        self.nx = nx
        self.ny = ny
        self.element_type = element_type
        self.nodes = None
        self.elements = None
        self.node_numbers = None
        
        # Set nodes per element based on element type
        if element_type == 'bilinear':
            self.nodes_per_element = 4
            self.order = 1
        elif element_type == 'biquadratic':
            self.nodes_per_element = 9
            self.order = 2
        elif element_type == 'bicubic':
            self.nodes_per_element = 16
            self.order = 3
        else:
            raise ValueError("Unsupported element type")

    def generate_mesh(self):
        """Generate complete FEM mesh WITHOUT excluding any elements"""
        xmin, xmax, ymin, ymax = self.domain
        
        # Initialize lists
        self.nodes = []
        self.elements = []
        self.node_numbers = {}
        node_count = 0
        
        # Generate element edge coordinates for reference
        x_edges = np.linspace(xmin, xmax, self.nx + 1)
        y_edges = np.linspace(ymin, ymax, self.ny + 1)
        
        # Store element centers for later notch identification
        self.element_centers = []
        
        # Generate nodes and elements for ALL elements (no exclusion)
        for j in range(self.ny):
            for i in range(self.nx):
                # Calculate element boundaries and center
                x_left, x_right = x_edges[i], x_edges[i+1]
                y_bottom, y_top = y_edges[j], y_edges[j+1]
                x_center = 0.5 * (x_left + x_right)
                y_center = 0.5 * (y_bottom + y_top)
                
                # Store element center
                self.element_centers.append([x_center, y_center])
                
                # Generate nodes for this element
                element_nodes = []
                
                if self.element_type == 'bilinear':
                    # Corner coordinates
                    coords = [
                        [x_left, y_bottom],   # node 0
                        [x_right, y_bottom],  # node 1
                        [x_right, y_top],     # node 2
                        [x_left, y_top]       # node 3
                    ]
                    
                elif self.element_type == 'biquadratic':
                    # 9-node coordinates (3x3 grid)
                    x_mid = 0.5 * (x_left + x_right)
                    y_mid = 0.5 * (y_bottom + y_top)
                    coords = [
                        [x_left, y_bottom],   # 0
                        [x_right, y_bottom],  # 1
                        [x_right, y_top],     # 2
                        [x_left, y_top],      # 3
                        [x_mid, y_bottom],    # 4
                        [x_right, y_mid],     # 5
                        [x_mid, y_top],       # 6
                        [x_left, y_mid],      # 7
                        [x_mid, y_mid]        # 8
                    ]
                    
                elif self.element_type == 'bicubic':
                    # 16-node coordinates (4x4 grid)
                    x_coords_elem = np.linspace(x_left, x_right, 4)
                    y_coords_elem = np.linspace(y_bottom, y_top, 4)
                    coords = []
                    for yy in y_coords_elem:
                        for xx in x_coords_elem:
                            coords.append([xx, yy])
                
                # Process each node coordinate
                for coord in coords:
                    x, y = coord
                    # Use rounded coordinates as key to handle floating point precision
                    node_key = tuple(np.round([x, y], decimals=10))
                    
                    if node_key not in self.node_numbers:
                        # New node
                        self.nodes.append([x, y])
                        self.node_numbers[node_key] = node_count
                        element_nodes.append(node_count)
                        node_count += 1
                    else:
                        # Existing node (shared between elements)
                        element_nodes.append(self.node_numbers[node_key])
                
                # Add element to list
                self.elements.append(element_nodes)
        
        # Convert to numpy arrays
        self.nodes = np.array(self.nodes)
        self.element_centers = np.array(self.element_centers)


def standard_shape_functions(xi, eta, element_type):
    if element_type == 'bilinear':
        return np.array([
            0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta), 0.25 * (1 - xi) * (1 + eta)
        ])
    elif element_type == 'biquadratic':
        N = np.zeros(9)
        N[0] = 0.25 * xi * (xi - 1) * eta * (eta - 1)
        N[1] = 0.25 * xi * (xi + 1) * eta * (eta - 1)
        N[2] = 0.25 * xi * (xi + 1) * eta * (eta + 1)
        N[3] = 0.25 * xi * (xi - 1) * eta * (eta + 1)
        N[4] = 0.5 * (1 - xi**2) * eta * (eta - 1)
        N[5] = 0.5 * xi * (xi + 1) * (1 - eta**2)
        N[6] = 0.5 * (1 - xi**2) * eta * (eta + 1)
        N[7] = 0.5 * xi * (xi - 1) * (1 - eta**2)
        N[8] = (1 - xi**2) * (1 - eta**2)
        return N
    elif element_type == 'bicubic':
        xi_nodes = np.array([-1, -1/3, 1/3, 1])
        eta_nodes = np.array([-1, -1/3, 1/3, 1])
        N = np.zeros(16)
        node_idx = 0
        for j in range(4):
            for i in range(4):
                Lxi = np.prod([(xi - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k]) for k in range(4) if k != i])
                Leta = np.prod([(eta - eta_nodes[k]) / (eta_nodes[j] - eta_nodes[k]) for k in range(4) if k != j])
                N[node_idx] = Lxi * Leta
                node_idx += 1
        return N
    return None


def standard_shape_function_derivatives(xi, eta, element_type):
    if element_type == 'bilinear':
        dN_dxi = np.array([-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)])
        dN_deta = np.array([-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)])
        return dN_dxi, dN_deta
    elif element_type == 'biquadratic':
        dN_dxi = np.zeros(9)
        dN_deta = np.zeros(9)
        dN_dxi[0] = 0.25 * (2*xi - 1) * eta * (eta - 1); dN_deta[0] = 0.25 * xi * (xi - 1) * (2*eta - 1)
        dN_dxi[1] = 0.25 * (2*xi + 1) * eta * (eta - 1); dN_deta[1] = 0.25 * xi * (xi + 1) * (2*eta - 1)
        dN_dxi[2] = 0.25 * (2*xi + 1) * eta * (eta + 1); dN_deta[2] = 0.25 * xi * (xi + 1) * (2*eta + 1)
        dN_dxi[3] = 0.25 * (2*xi - 1) * eta * (eta + 1); dN_deta[3] = 0.25 * xi * (xi - 1) * (2*eta + 1)
        dN_dxi[4] = -xi * eta * (eta - 1); dN_deta[4] = 0.5 * (1 - xi**2) * (2*eta - 1)
        dN_dxi[5] = 0.5 * (2*xi + 1) * (1 - eta**2); dN_deta[5] = -xi * (xi + 1) * eta
        dN_dxi[6] = -xi * eta * (eta + 1); dN_deta[6] = 0.5 * (1 - xi**2) * (2*eta + 1)
        dN_dxi[7] = 0.5 * (2*xi - 1) * (1 - eta**2); dN_deta[7] = -xi * (xi - 1) * eta
        dN_dxi[8] = -2*xi * (1 - eta**2); dN_deta[8] = -2*eta * (1 - xi**2)
        return dN_dxi, dN_deta
    elif element_type == 'bicubic':
        xi_nodes = np.array([-1, -1/3, 1/3, 1])
        eta_nodes = np.array([-1, -1/3, 1/3, 1])
        dN_dxi, dN_deta = np.zeros(16), np.zeros(16)
        node_idx = 0
        for j in range(4):
            for i in range(4):
                dLxi_dxi = 0.0
                Leta = 1.0
                for k in range(4):
                    if k != i:
                        temp = 1.0 / (xi_nodes[i] - xi_nodes[k])
                        for m in range(4):
                            if m != i and m != k:
                                temp *= (xi - xi_nodes[m]) / (xi_nodes[i] - xi_nodes[m])
                        dLxi_dxi += temp
                for k in range(4):
                    if k != j:
                        Leta *= (eta - eta_nodes[k]) / (eta_nodes[j] - eta_nodes[k])
                dN_dxi[node_idx] = dLxi_dxi * Leta
                
                Lxi = 1.0
                dLeta_deta = 0.0
                for k in range(4):
                    if k != i:
                        Lxi *= (xi - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k])
                for k in range(4):
                    if k != j:
                        temp = 1.0 / (eta_nodes[j] - eta_nodes[k])
                        for m in range(4):
                            if m != j and m != k:
                                temp *= (eta - eta_nodes[m]) / (eta_nodes[j] - eta_nodes[m])
                        dLeta_deta += temp
                dN_deta[node_idx] = Lxi * dLeta_deta
                node_idx += 1
        return dN_dxi, dN_deta
    return None, None


class BeamProperties:
    """Class to store beam material and geometric properties"""
    def __init__(self, L: float, b: float, h: float, E: float, rho: float, nu: float):
        self.L, self.b, self.h, self.E, self.rho, self.nu = L, b, h, E, rho, nu

    def get_D_matrix(self) -> np.ndarray:
        factor = self.E / ((1 + self.nu) * (1 - 2*self.nu))
        return factor * np.array([
            [1 - self.nu, self.nu, 0], 
            [self.nu, 1 - self.nu, 0], 
            [0, 0, ((1 - 2*self.nu) / 2)*(5/6)]
        ], dtype=np.float32)


def elemental_matrices(element: np.ndarray, mesh, beam: BeamProperties) -> Tuple[np.ndarray, np.ndarray]:
    nodes = mesh.nodes[element]
    n_nodes = len(element)
    
    quad_order = {'bilinear': 2, 'biquadratic': 3, 'bicubic': 4}.get(mesh.element_type, 4)
    xi_points, xi_weights = gauss_quadrature(quad_order)
    eta_points, eta_weights = gauss_quadrature(quad_order)
    
    # MEMORY FIX: Use float32 for smaller memory footprint.
    K_e = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=np.float32)
    M_e = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=np.float32)
    D = beam.get_D_matrix()
    
    for i, xi_i in enumerate(xi_points):
        for j, eta_j in enumerate(eta_points):
            N = standard_shape_functions(xi_i, eta_j, mesh.element_type)
            dN_dxi, dN_deta = standard_shape_function_derivatives(xi_i, eta_j, mesh.element_type)
            
            J = np.array([[np.dot(dN_dxi, nodes[:, 0]), np.dot(dN_dxi, nodes[:, 1])],
                          [np.dot(dN_deta, nodes[:, 0]), np.dot(dN_deta, nodes[:, 1])]])
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-10: raise ValueError("Singular Jacobian detected")
            J_inv = np.linalg.inv(J)
            
            dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
            dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
            
            B = np.zeros((3, 2 * n_nodes), dtype=np.float32)
            B[0, 0::2] = dN_dx; B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy; B[2, 1::2] = dN_dx
            
            N_matrix = np.zeros((2, 2 * n_nodes), dtype=np.float32)
            N_matrix[0, 0::2] = N; N_matrix[1, 1::2] = N
            
            weight = beam.b * det_J * xi_weights[i] * eta_weights[j]
            
            K_e += weight * (B.T @ D @ B)
            M_e += weight * beam.rho * (N_matrix.T @ N_matrix)
    
    return K_e, M_e










def identify_notch_elements(mesh, notch):
    """Identify which elements are affected by the notch"""
    notch_xmin, notch_xmax, notch_ymin, notch_ymax = notch
    notch_elements = []
    
    for idx, center in enumerate(mesh.element_centers):
        x_center, y_center = center
        if (notch_xmin <= x_center <= notch_xmax and
            notch_ymin <= y_center <= notch_ymax):
            notch_elements.append(idx)
    
    return notch_elements

def assemble_global_matrices_with_cache(mesh, beam: BeamProperties, cache_paths, force_recompute=False):
    """Assembles global matrices with memory-efficient caching."""
    
    # Check if valid cache exists
    if not force_recompute and all(os.path.exists(p) for p in [cache_paths['K_path'], cache_paths['M_path']]):
        logging.info("Loading cached matrices...")
        
        # Load K from its sparse components
        loader = np.load(cache_paths['K_path'])
        K_sparse = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        
        M_diag = np.load(cache_paths['M_path'])['data']
        
        return K_sparse, M_diag
    
    logging.info("Computing matrices from scratch...")
    n_dofs = 2 * len(mesh.nodes)
    K = lil_matrix((n_dofs, n_dofs), dtype=np.float32)
    M = lil_matrix((n_dofs, n_dofs), dtype=np.float32)
    
    for e_idx, element in enumerate(mesh.elements):
        try:
            K_e, M_e = elemental_matrices(element, mesh, beam)
            
            # Assemble into global matrices
            idx = np.array([[2*node, 2*node+1] for node in element]).flatten()
            K[np.ix_(idx, idx)] += K_e
            M[np.ix_(idx, idx)] += M_e
            
        except Exception as e:
            logging.error(f"Error in element {e_idx}: {str(e)}")
            raise
    
    K_csr = K.tocsr()
    M_csr = M.tocsr()
    
    # Lumped mass
    M_lumped_diag = np.array(M_csr.sum(axis=1)).flatten()
    
    # Save to cache
    logging.info("Saving matrices to cache...")
    
    # Save K as sparse components
    np.savez_compressed(cache_paths['K_path'], 
                        data=K_csr.data,
                        indices=K_csr.indices,
                        indptr=K_csr.indptr,
                        shape=K_csr.shape)
                        
    np.savez_compressed(cache_paths['M_path'], data=M_lumped_diag)
    
    return K_csr, M_lumped_diag

def modify_matrices_for_notch(K_base, M_lumped_base, mesh, beam, notch_elements, reduction_factor=1e-6):
    """
    Modify existing matrices to account for notch by re-computing
    matrices only for the specific notch elements, avoiding large storage.
    """
    
    K_modified = K_base.tolil()
    M_lumped_modified = M_lumped_base.copy()
    
    for e_idx in notch_elements:
        element = mesh.elements[e_idx]
        idx = np.array([[2*node, 2*node+1] for node in element]).flatten()
        
        # Re-compute elemental matrices on-the-fly to save memory
        K_e_orig, M_e_orig = elemental_matrices(element, mesh, beam)
        
        # Subtract original contribution
        K_modified[np.ix_(idx, idx)] -= K_e_orig
        
        # Add reduced stiffness (to avoid singularity)
        K_modified[np.ix_(idx, idx)] += reduction_factor * K_e_orig
        
        # Adjust the lumped mass matrix
        M_row_sums_orig = M_e_orig.sum(axis=1)
        M_row_sums_reduced = reduction_factor * M_row_sums_orig
        
        for local_idx, global_idx in enumerate(idx):
            M_lumped_modified[global_idx] -= M_row_sums_orig[local_idx]
            M_lumped_modified[global_idx] += M_row_sums_reduced[local_idx]
    
    return K_modified.tocsr(), M_lumped_modified

# [KEEP UNCHANGED] - Put scipy_csr_to_torch_sparse_safe() function here
def scipy_csr_to_torch_sparse_safe(csr_matrix, device):
    """Safely convert scipy CSR matrix to PyTorch sparse tensor"""
    try:
        # Method 1: Use COO format (more compatible)
        coo = csr_matrix.tocoo()
        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col])
        ).long()
        values = torch.from_numpy(coo.data).float()
        size = coo.shape
        
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, size, 
            dtype=torch.float32, device=device
        )
        return sparse_tensor.coalesce()
        
    except Exception as e:
        logging.warning(f"COO conversion failed: {e}")
        raise RuntimeError("Matrix is too large for dense conversion and sparse conversion failed")

def compute_stable_time_step(K, M_lumped_diag):
    """Compute stable time step for central difference method"""
    # Estimate maximum eigenvalue using max diagonal ratio
    max_ratio = np.max(K.diagonal() / M_lumped_diag)
    omega_max = np.sqrt(max_ratio)
    dt_stable = 2.0 / omega_max
    return dt_stable * 0.8  # Apply safety factor

def central_difference_solve_memory_efficient(
    K, M_lumped_diag, excitation_signal, dt, num_steps, 
    excitation_dofs, response_dofs_global
):
    """
    Solve 2D dynamic problem using central difference with memory optimization,
    lumped mass matrix. Free-free boundary conditions.
    """
    device = get_safe_device()
    logging.info(f"Using device: {device}")
    
    # Check time step stability
    dt_stable = compute_stable_time_step(K, M_lumped_diag)
    if dt > dt_stable:
        logging.warning(f"Time step {dt:.2e} exceeds stable limit {dt_stable:.2e}. Using stable time step.")
        dt = dt_stable
        # Adjust number of steps to maintain same total time
        total_time = dt * num_steps
        num_steps = int(total_time / dt)
        logging.info(f"Adjusted to {num_steps} steps with dt={dt:.2e}")
    
    n_dofs = K.shape[0]
    
    # Convert to GPU Tensors
    try:
        K_gpu = scipy_csr_to_torch_sparse_safe(K, device)
        M_inv_diag_gpu = torch.from_numpy(1.0 / M_lumped_diag).to(device=device, dtype=torch.float32)

    except Exception as e:
        logging.error(f"GPU tensor conversion failed: {e}")
        raise

    # Memory Efficient Storage for multiple response points
    num_response_points = len(response_dofs_global)
    u_hist = np.zeros((num_steps, num_response_points))
    v_hist = np.zeros((num_steps, num_response_points))
    a_hist = np.zeros((num_steps, num_response_points))
    
    u_prev = torch.zeros(n_dofs, dtype=torch.float32, device=device)
    u_curr = torch.zeros(n_dofs, dtype=torch.float32, device=device)

    # Initial conditions
    F0 = torch.zeros(n_dofs, dtype=torch.float32, device=device)
    F0[excitation_dofs[0]] = -excitation_signal[0]
    F0[excitation_dofs[1]] = excitation_signal[0]
    
    a_curr = M_inv_diag_gpu * F0
    u_curr = u_prev + 0.5 * dt**2 * a_curr
    
    # Store initial values for all response points
    for idx, resp_dof in enumerate(response_dofs_global):
        u_hist[0, idx] = u_prev[resp_dof].item()
        v_hist[0, idx] = 0.0
        a_hist[0, idx] = a_curr[resp_dof].item()
    
    # Time Integration Loop (no damping for free-free boundaries)
    for i in range(1, num_steps):
        F_t = torch.zeros(n_dofs, dtype=torch.float32, device=device)
        F_t[excitation_dofs[0]] = -excitation_signal[i]
        F_t[excitation_dofs[1]] = excitation_signal[i]
        
        # Effective force (only stiffness, no damping)
        F_eff = F_t - torch.sparse.mm(K_gpu, u_curr.unsqueeze(1)).squeeze(1)
            
        accel_term = M_inv_diag_gpu * F_eff
        
        u_next = dt**2 * accel_term + 2 * u_curr - u_prev
        v_curr = (u_next - u_prev) / (2 * dt)
        a_curr = (u_next - 2 * u_curr + u_prev) / (dt**2)
        
        # Store values for all response points
        for idx, resp_dof in enumerate(response_dofs_global):
            u_hist[i, idx] = u_curr[resp_dof].item()
            v_hist[i, idx] = v_curr[resp_dof].item()
            a_hist[i, idx] = a_curr[resp_dof].item()
        
        u_prev, u_curr = u_curr, u_next
        
        if i % 500 == 0:
            logging.info(f"Completed {i}/{num_steps} time steps")

    return u_hist, v_hist, a_hist

# [KEEP UNCHANGED] - Put generate_excitation() function here
def generate_excitation(t, active_duration, total_duration, frequency):
    """Generate excitation signal with Hanning window"""
    N = len(t)
    active_samples = int(N * active_duration / total_duration)
    
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(active_samples) / (active_samples - 1)))
    active_signal = window * np.sin(2 * np.pi * frequency * t[:active_samples])
    
    full_signal = np.zeros(N)
    full_signal[:active_samples] = active_signal
    return full_signal

def run_fem_analysis(L, E, rho, notch_location, notch_width, notch_depth, response_x_positions):
    """
    Main function with memory-efficient caching and free-free boundaries
    """
    
    # Apply PyTorch fixes
    apply_pytorch_fix()
    
    # Fixed beam properties
    b = 1.0  # width
    h = 0.0015  # height
    nu = 0.33  # Poisson's ratio
    
    # Base parameters for caching
    L_base, E_base, rho_base = 2.0, 7e10, 2700
    
    # Mesh parameters
    nx, ny = 6000, 10
    element_type = 'biquadratic'
    
    # Check if we need to recompute matrices
    need_recompute = (L != L_base or E != E_base or rho != rho_base)
    
    # Get cache paths
    cache_paths = get_cache_path(L, E, rho, nx, ny, element_type)
    
    # Define beam properties
    beam = BeamProperties(L=L, b=b, h=h, E=E, rho=rho, nu=nu)
    
    # Create mesh (always without notch initially)
    mesh = Mesh2D([0, L, 0, h], nx=nx, ny=ny, element_type=element_type)
    mesh.generate_mesh()
    
    # Check cache validity
    cache_valid = check_cache_validity(cache_paths['params_path'], L, E, rho, nx, ny, element_type)
    
    if need_recompute or not cache_valid:
        logging.info(f"Parameters changed from base values or cache invalid. Recomputing matrices...")
        # Save new parameters
        save_cache_params(cache_paths['params_path'], L, E, rho, nx, ny, element_type, b, h, nu)
    
    # Load/assemble matrices
    K_base, M_lumped_base = assemble_global_matrices_with_cache(
        mesh, beam, cache_paths,
        force_recompute=(need_recompute or not cache_valid)
    )
    
    # Define notch
    notch_xmin = notch_location - notch_width / 2
    notch_xmax = notch_location + notch_width / 2
    notch_ymin = h - notch_depth
    notch_ymax = h
    notch = [notch_xmin, notch_xmax, notch_ymin, notch_ymax]
    
    # Identify notch elements
    notch_elements = identify_notch_elements(mesh, notch)
    logging.info(f"Found {len(notch_elements)} elements in notch region")
    
    # Modify matrices for the notch
    K, M_lumped_diag = modify_matrices_for_notch(
        K_base, M_lumped_base, mesh, beam, notch_elements
    )
    
    # Time parameters
    N = 150000
    active_duration = 50e-6
    total_duration = 300e-6
    dt = total_duration/N
    t = np.linspace(0, total_duration, N, endpoint=False)
    
    # Generate excitation signal
    f = 100e3
    excitation_signal = generate_excitation(t, active_duration, total_duration, f)
    
    # Find nodes for excitation (shifted 1m to the right)
    excitation_y = h
    excitation_x1, excitation_x2 = 0.6465 + 1.0, 0.6535 + 1.0
    
    node_coords = mesh.nodes
    dist1 = np.linalg.norm(node_coords - np.array([excitation_x1, excitation_y]), axis=1)
    dist2 = np.linalg.norm(node_coords - np.array([excitation_x2, excitation_y]), axis=1)
    
    excitation_node1, excitation_node2 = np.argmin(dist1), np.argmin(dist2)
    
    # Define excitation DOFs
    excitation_dofs = [2 * excitation_node1, 2 * excitation_node2]
    
    # Find response nodes for all requested positions
    response_nodes = []
    response_dofs_global = []
    
    for response_x in response_x_positions:
        dist_resp = np.linalg.norm(node_coords - np.array([response_x, excitation_y]), axis=1)
        response_node = np.argmin(dist_resp)
        response_nodes.append(response_node)
        response_dofs_global.append(2 * response_node)  # Y-direction DOF
    
    logging.info(f"Found {len(response_nodes)} response nodes")
    
    # Free-free boundary conditions
    logging.info("Starting memory-efficient central difference solution with free-free boundaries...")
    u_hist, __, __ = central_difference_solve_memory_efficient(
        K, M_lumped_diag, excitation_signal, dt, N, excitation_dofs, response_dofs_global
    )
    
    # Prepare results dictionary
    results = {
        'time': t,
        'response_positions': response_x_positions,
        'displacement': {}
    }
    
    # Store results for each response position
    for idx, x_pos in enumerate(response_x_positions):
        results['displacement'][x_pos] = u_hist[:, idx]
    
    logging.info("FEM analysis completed successfully with free-free boundaries")
    return results

def plot_response(results, case_name="", save_plots=False):
    """Plot the displacement response and optionally save to file"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    time = results['time']
    
    for x_pos in results['response_positions']:
        displacement = results['displacement'][x_pos]
        plt.plot(time * 1e6, displacement * 1e6, label=f'x = {x_pos} m')
    
    plt.xlabel('Time (μs)')
    plt.ylabel('Displacement (μm)')
    plt.title(f'Displacement Response at Different Positions (Free-Free Boundaries) - {case_name}')
    plt.grid(True)
    plt.legend()
    
    if save_plots:
        filename = f'displacement_response_{case_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved as {filename}")
    
    plt.show()


# Example usage for testing the free-free system:
if __name__ == "__main__":

    
    # Test parameters
    L = 3.0 # Updated length 
    E = 7.029509165300941e10 
    rho = 2696.054692379139
    
    # Response positions (shifted 1m to the right)
    response_x_positions = [0.85 + 1.0]
    
    # First run - will compute and cache
    print("First run - computing matrices...")
    """
    results1 = run_fem_analysis(L, E, rho, 
                                notch_location=1.78871, # Shifted by 1m 
                                notch_width=0.00083, 
                                notch_depth=0.00066,
                                response_x_positions=response_x_positions)
    plot_response(results1, "Case 1", save_plots=True)
        
    # Second run - will use cache and only modify for notch
    print("\nSecond run - using cached matrices...")
    results2 = run_fem_analysis(L, E, rho, 
                                notch_location=0.80 + 1.0,   # Shifted 1m right
                                notch_width=0.0010,   # Different width
                                notch_depth=0.0007,  # Different depth
                                response_x_positions=response_x_positions)
    
    plot_response(results2, "Case 2", save_plots=True)
    
    # Third run - different notch parameters
    print("\nThird run - different notch parameters...")
    results3 = run_fem_analysis(L, E, rho, 
                                notch_location=0.90 + 1.0,  # Shifted 1m right
                                notch_width=0.0012, 
                                notch_depth=0.0008,
                                response_x_positions=response_x_positions)
    
    plot_response(results3, "Case 3", save_plots=True)
    """

#part 2
# for mass generation of responses using parameters from /home/user2/Music/abhi2/datagen/hfsm_train_dataset.csv and /home/user2/Music/abhi2/datagen/hfsm_test_dataset.csv

import pandas as pd
import numpy as np
import time
import os
from scipy.interpolate import interp1d

# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================
def scale_and_subsample(signal_data, target_points=1500):
    """
    Scale signal between -1 to 1 and subsample/interpolate to target_points.
    """
    # Scale to the [-1, 1] range based on the maximum absolute value
    max_abs = np.max(np.abs(signal_data))
    if max_abs > 0:
        signal_data = signal_data / max_abs

    # Resample
    current_points = len(signal_data)
    if current_points > target_points:
        indices = np.linspace(0, current_points - 1, target_points, dtype=int)
        signal_data = signal_data[indices]
    elif current_points < target_points:
        old_indices = np.arange(current_points)
        new_indices = np.linspace(0, current_points - 1, target_points)
        f = interp1d(old_indices, signal_data, kind='linear', fill_value="extrapolate")
        signal_data = f(new_indices)

    return signal_data


def process_dataset(df, dataset_name, response_positions, target_points, output_dir, start_index=0):
    """
    Process dataset and save results in the new format.
    Each response from each sensor is stored as a separate row.
    """
    print(f"\nProcessing {dataset_name} dataset...")

    output_filename = os.path.join(output_dir, f'{dataset_name}_responses.csv')
    temp_filename = os.path.join(output_dir, f'{dataset_name}_responses_temp.csv')

    results_data = []
    total_cases = len(df)

    for idx, row in df.iloc[start_index:].iterrows():
        try:
            case_id = row['case_id']
            L = row['length']
            E = row['youngs_modulus']
            rho = row['density']
            damage_present = row['damage_present']

            if damage_present == 'y':
                notch_location = row['notch_x']
                notch_width = row['notch_width']
                notch_depth = row['notch_depth']
            else:
                notch_location = 0.0
                notch_width = 1e-8
                notch_depth = 1e-8

            print(f"\nRunning case {idx+1}/{total_cases} - {case_id}: L={L:.3f}, E={E:.2e}, rho={rho:.1f}")
            print(f"  Damage: {damage_present}, Notch: x={notch_location:.5f}, w={notch_width:.5f}, d={notch_depth:.5f}")

            # FEM analysis must be implemented elsewhere
            start_time = time.time()
            results = run_fem_analysis(
                L=L,
                E=E, 
                rho=rho,
                notch_location=notch_location,
                notch_width=notch_width,
                notch_depth=notch_depth,
                response_x_positions=response_positions
            )
            analysis_time = time.time() - start_time
            print(f"  Analysis completed in {analysis_time:.2f} seconds")

            for pos in response_positions:
                displacement = results['displacement'][pos]
                processed_response = scale_and_subsample(displacement, target_points)

                case_result = {
                    'case_id': case_id,
                    'response_point': pos,
                    'notch_x': f"{notch_location:.5f}",
                    'notch_depth': f"{notch_depth:.5f}",
                    'notch_width': f"{notch_width:.5f}",
                    'length': L,
                    'density': rho,
                    'youngs_modulus': E
                }
                for i, val in enumerate(processed_response):
                    case_result[f'r{i}'] = f"{val:.5f}"

                results_data.append(case_result)

            if (idx + 1) % 5 == 0:
                temp_df = pd.DataFrame(results_data)
                temp_df.to_csv(temp_filename, index=False)
                print(f"  Saved intermediate results for {len(results_data)} rows")

        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            with open(os.path.join(output_dir, 'error_log.txt'), 'a') as f:
                f.write(f"Case ID {case_id} failed with error: {str(e)}\n")
            continue

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_filename, index=False)

    print(f"\n{dataset_name} dataset analysis completed!")
    print(f"Results saved to: {output_filename}")

    if os.path.exists(output_filename):
        file_size = os.path.getsize(output_filename) / (1024*1024)
        print(f"Output file size: {file_size:.2f} MB")

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return results_df


def run_fem_on_dataset():
    """
    Main function to run FEM analysis on datasets and store responses.
    """
    train_file = 'hfsm_2d_train_dataset.csv'
    test_file = 'hfsm_2d_test_dataset.csv'
    output_dir = './'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Commented out train dataset loading
        train_df = pd.read_csv(train_file)
        #test_df = pd.read_csv(test_file)
        print(f"Loaded train dataset: {len(train_df)} cases from {train_file}")
        #print(f"Loaded test dataset: {len(test_df)} cases from {test_file}")
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. {e}")
        return

    response_positions = [1.85, 1.87, 1.9, 1.92, 1.95, 1.97, 2.0, 2.02, 2.05, 2.07, 2.1]
    target_points = 1500
    print(f"Response positions: {response_positions}")
    print(f"Target points per response: {target_points}")

    # Commented out train dataset processing
    print("\n" + "="*60)
    print("STARTING TRAIN DATASET ANALYSIS")
    print("="*60)
    train_results = process_dataset(train_df, 'train', response_positions, target_points, output_dir)

    #print("\n" + "="*60)
    #print("STARTING TEST DATASET ANALYSIS (from case 21 onwards)")
    #print("="*60)
    #test_results = process_dataset(test_df, 'test', response_positions, target_points, output_dir, start_index=0)

    #print("\n" + "="*60)
    #print("ANALYSIS SUMMARY")
    #print("="*60)
    #if 'test_results' in locals() and not test_results.empty:
        #print(f"Test dataset: {test_results['case_id'].nunique()} cases processed, resulting in {len(test_results)} rows.")
    #print(f"Output files are located in: {output_dir}")


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Starting FEM analysis on generated datasets...")
    print(f"Storing responses with {1500} points each.")
    print("This may create large CSV files. Progress is saved every 5 cases.")

    run_fem_on_dataset()

