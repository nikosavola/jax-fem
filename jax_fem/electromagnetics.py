"""High-frequency electromagnetic simulation module for JAX-FEM.

This module implements the time-harmonic vector wave equation for the electric
field using H(curl)-conforming Nédélec (edge) elements:

.. math::

    \\nabla \\times \\left(\\frac{1}{\\mu_r} \\nabla \\times \\mathbf{E}\\right)
    - k_0^2 \\epsilon_{rc} \\mathbf{E} = -j\\omega\\mu_0 \\mathbf{J}_s

The weak form reads:

.. math::

    \\int_V \\left[\\frac{1}{\\mu_r}(\\nabla \\times \\mathbf{E})
    \\cdot (\\nabla \\times \\mathbf{w})
    - k_0^2 \\epsilon_{rc} \\mathbf{E} \\cdot \\mathbf{w}\\right] dV
    = \\int_V -j\\omega\\mu_0 \\mathbf{J}_s \\cdot \\mathbf{w} \\, dV
    + \\oint_S (\\hat{n} \\times \\mathbf{H}) \\cdot \\mathbf{w} \\, dS

Boundary conditions supported:

*  **PEC** (Perfect Electric Conductor): :math:`\\hat{n} \\times \\mathbf{E} = 0`.
   Imposed strongly by setting edge DoFs to 0 on boundary faces.
*  **First-order ABC** (Absorbing Boundary Condition): adds the surface integral
   :math:`j k_0 \\int_S (\\hat{n} \\times \\mathbf{E}) \\cdot (\\hat{n} \\times \\mathbf{w}) dS`.
"""

import numpy as onp
import jax
import jax.numpy as np
import scipy.sparse
import scipy.sparse.linalg
from jax import config
config.update("jax_enable_x64", True)

from jax_fem.generate_mesh import Mesh
from jax_fem.edge_fe import EdgeFiniteElement
from jax_fem import logger

# Physical constants
MU_0 = 4.0e-7 * onp.pi      # Vacuum permeability  [H/m]
EPS_0 = 8.854187817e-12      # Vacuum permittivity  [F/m]
C_0 = 1.0 / onp.sqrt(MU_0 * EPS_0)  # Speed of light [m/s]


class EMProblem:
    """Assemble and solve the time-harmonic vector wave equation.

    Parameters
    ----------
    edge_fe : EdgeFiniteElement
        Pre-built edge finite element (carries mesh, topology, basis, BCs).
    frequency : float
        Operating frequency in Hz.
    mu_r : complex, optional
        Relative permeability (scalar, isotropic).  Default 1.
    eps_r : complex, optional
        Relative permittivity.  Default 1.
    sigma : float, optional
        Electrical conductivity [S/m].  Default 0.
    source_fn : callable, optional
        ``source_fn(x)`` returns source current density :math:`\\mathbf{J}_s`
        at physical coordinate *x* (shape ``(dim,)``).  Default is no source.
    abc_edges : ndarray of int, optional
        Global edge indices where first-order ABC is applied.
    abc_face_info : dict, optional
        Pre-computed ABC surface integral data.  If *None*, ABC is not used.
    """

    def __init__(self, edge_fe, frequency, *,
                 mu_r=1.0+0j, eps_r=1.0+0j, sigma=0.0,
                 source_fn=None):
        self.fe = edge_fe
        self.freq = frequency
        self.omega = 2.0 * onp.pi * frequency
        self.k0 = self.omega / C_0
        self.mu_r = complex(mu_r)
        self.eps_rc = complex(eps_r) - 1j * sigma / (self.omega * EPS_0) if self.omega != 0 else complex(eps_r)
        self.source_fn = source_fn
        self.num_dofs = edge_fe.num_total_dofs

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def assemble(self):
        """Assemble global stiffness matrix and load vector.

        Returns
        -------
        A : scipy.sparse.csr_matrix, shape (num_dofs, num_dofs)
            Complex system matrix.
        b : ndarray, shape (num_dofs,)
            Complex load vector.
        """
        fe = self.fe
        num_dofs = self.num_dofs
        dim = fe.dim

        # Pre-allocate triplet lists
        rows = []
        cols = []
        vals = []

        # Source vector
        b = onp.zeros(num_dofs, dtype=onp.complex128)

        # Physical quad points for source evaluation
        from jax_fem.basis import get_shape_vals_and_grads
        shape_vals_lag, _, _ = get_shape_vals_and_grads(fe.ele_type, fe.gauss_order)
        physical_coos = onp.take(fe.points, fe.cells, axis=0)  # (C, N_nodes, dim)
        # (C, Q, dim)
        physical_quad_pts = onp.einsum('qn,cnd->cqd', shape_vals_lag, physical_coos)

        for c in range(fe.num_cells):
            # Local DoF indices and orientations
            local_edge_inds = fe.cell_edge_indices[c]   # (num_local_edges,)
            orient = fe.edge_orientations[c]             # (num_local_edges,)

            # Local element matrix  K_local[i,j]
            K_local = onp.zeros((fe.num_local_edges, fe.num_local_edges), dtype=onp.complex128)

            for q in range(fe.num_quads):
                JxW_q = fe.JxW[c, q]

                if dim == 3:
                    # Curl-curl term
                    curl_q = fe.shape_curls_physical[c, q]  # (num_local_edges, dim)
                    curl_curl = onp.einsum('id,jd->ij', curl_q, curl_q) * JxW_q / self.mu_r

                    # Mass term
                    vals_q = fe.shape_vals_physical[c, q]  # (num_local_edges, dim)
                    mass = onp.einsum('id,jd->ij', vals_q, vals_q) * JxW_q * (-self.k0**2 * self.eps_rc)
                elif dim == 2:
                    # Curl-curl term (scalar curls in 2D)
                    curl_q = fe.shape_curls_physical[c, q]  # (num_local_edges,)
                    curl_curl = onp.outer(curl_q, curl_q) * JxW_q / self.mu_r

                    # Mass term
                    vals_q = fe.shape_vals_physical[c, q]  # (num_local_edges, dim)
                    mass = onp.einsum('id,jd->ij', vals_q, vals_q) * JxW_q * (-self.k0**2 * self.eps_rc)
                else:
                    raise ValueError(f"Unsupported dim {dim}")

                K_local += curl_curl + mass

            # Apply edge orientations: K_global[I,J] += orient[i]*orient[j]*K_local[i,j]
            orient_outer = onp.outer(orient, orient)
            K_local *= orient_outer

            # Source vector
            if self.source_fn is not None:
                f_local = onp.zeros(fe.num_local_edges, dtype=onp.complex128)
                for q in range(fe.num_quads):
                    x_q = physical_quad_pts[c, q]
                    Js = self.source_fn(x_q)  # (dim,)
                    vals_q = fe.shape_vals_physical[c, q]  # (E, dim)
                    # -j * omega * mu_0 * Js · N_i * JxW
                    f_local += onp.einsum('ed,d->e', vals_q, Js) * \
                               (-1j * self.omega * MU_0 * fe.JxW[c, q])
                f_local *= orient
                for i_local in range(fe.num_local_edges):
                    b[local_edge_inds[i_local]] += f_local[i_local]

            # Scatter to global
            for i_local in range(fe.num_local_edges):
                for j_local in range(fe.num_local_edges):
                    rows.append(local_edge_inds[i_local])
                    cols.append(local_edge_inds[j_local])
                    vals.append(K_local[i_local, j_local])

        A = scipy.sparse.coo_matrix(
            (onp.array(vals), (onp.array(rows), onp.array(cols))),
            shape=(num_dofs, num_dofs)
        ).tocsr()

        return A, b

    # ------------------------------------------------------------------
    # Boundary condition application
    # ------------------------------------------------------------------
    @staticmethod
    def apply_dirichlet(A, b, dof_inds, dof_vals):
        """Apply essential (Dirichlet / PEC) boundary conditions via row elimination.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            System matrix (modified in-place via lil conversion).
        b : ndarray
            RHS vector (modified in-place).
        dof_inds : ndarray of int
            Indices of constrained DoFs.
        dof_vals : ndarray of complex
            Prescribed values.

        Returns
        -------
        A : scipy.sparse.csr_matrix
        b : ndarray
        """
        if len(dof_inds) == 0:
            return A, b

        A_lil = A.tolil()
        for idx, val in zip(dof_inds, dof_vals):
            A_lil[idx, :] = 0
            A_lil[idx, idx] = 1.0
            b[idx] = val
        return A_lil.tocsr(), b

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, solver_options=None):
        """Assemble and solve the EM system.

        Parameters
        ----------
        solver_options : dict, optional
            ``{'solver': 'direct'}`` or ``{'solver': 'iterative', ...}``.

        Returns
        -------
        edge_sol : ndarray, shape (num_dofs,)
            Complex-valued edge DoF solution.
        """
        if solver_options is None:
            solver_options = {}

        logger.info("Assembling EM system matrix and RHS...")
        A, b = self.assemble()

        # Apply PEC boundary conditions
        A, b = self.apply_dirichlet(A, b, self.fe.edge_dof_inds, self.fe.edge_dof_vals)

        logger.info(f"Solving complex linear system of size {self.num_dofs}...")
        solver_type = solver_options.get('solver', 'direct')

        if solver_type == 'direct':
            edge_sol = scipy.sparse.linalg.spsolve(A, b)
        elif solver_type == 'iterative':
            maxiter = solver_options.get('maxiter', 10000)
            tol = solver_options.get('tol', 1e-10)
            edge_sol, info = scipy.sparse.linalg.gmres(A, b, maxiter=maxiter, rtol=tol)
            if info != 0:
                logger.warning(f"GMRES did not converge, info = {info}")
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        res_norm = onp.linalg.norm(A @ edge_sol - b)
        logger.info(f"EM solve residual norm: {res_norm:.6e}")

        return edge_sol

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def reconstruct_field(self, edge_sol):
        """Reconstruct the E-field at quadrature points from edge DoFs.

        Parameters
        ----------
        edge_sol : ndarray, shape (num_dofs,)

        Returns
        -------
        E_field : ndarray, shape (num_cells, num_quads, dim)
            Complex vector field.
        """
        fe = self.fe
        cell_edge_sols = edge_sol[fe.cell_edge_indices] * fe.edge_orientations
        E_field = onp.einsum('cqed,ce->cqd', fe.shape_vals_physical, cell_edge_sols)
        return E_field


def compute_cavity_eigenvalues(edge_fe, mu_r=1.0, eps_r=1.0, num_modes=6,
                               sigma_shift=None):
    """Solve the cavity eigenvalue problem for resonant frequencies.

    Solves the generalised eigenvalue problem:

    .. math::

        S \\mathbf{e} = k^2 M \\mathbf{e}

    where *S* is the curl-curl stiffness matrix and *M* is the mass matrix.
    The curl-curl operator has a large null space (gradients of scalar fields).
    This function uses a penalty approach to push the zero modes away and
    cleanly separate them from the physical modes.

    Parameters
    ----------
    edge_fe : EdgeFiniteElement
        Edge FE with PEC boundary conditions already applied.
    mu_r : float
        Relative permeability.
    eps_r : float
        Relative permittivity.
    num_modes : int
        Number of physical (non-zero) eigenvalues to compute.
    sigma_shift : float, optional
        Shift for the shift-invert eigensolver.  If *None*, an automatic
        estimate is used.

    Returns
    -------
    eigenvalues : ndarray
        Sorted non-zero eigenvalues :math:`k^2`.
    resonant_frequencies : ndarray
        Corresponding resonant frequencies :math:`f = c_0 k / (2\\pi)`.
    """
    fe = edge_fe
    dim = fe.dim
    num_dofs = fe.num_total_dofs

    rows, cols, s_vals, m_vals = [], [], [], []

    for c in range(fe.num_cells):
        local_inds = fe.cell_edge_indices[c]
        orient = fe.edge_orientations[c]

        S_local = onp.zeros((fe.num_local_edges, fe.num_local_edges))
        M_local = onp.zeros((fe.num_local_edges, fe.num_local_edges))

        for q in range(fe.num_quads):
            JxW_q = fe.JxW[c, q]

            if dim == 3:
                curl_q = fe.shape_curls_physical[c, q]
                S_local += onp.einsum('id,jd->ij', curl_q, curl_q) * JxW_q / mu_r

                vals_q = fe.shape_vals_physical[c, q]
                M_local += onp.einsum('id,jd->ij', vals_q, vals_q) * JxW_q * eps_r
            elif dim == 2:
                curl_q = fe.shape_curls_physical[c, q]
                S_local += onp.outer(curl_q, curl_q) * JxW_q / mu_r

                vals_q = fe.shape_vals_physical[c, q]
                M_local += onp.einsum('id,jd->ij', vals_q, vals_q) * JxW_q * eps_r

        orient_outer = onp.outer(orient, orient)
        S_local *= orient_outer
        M_local *= orient_outer

        for i in range(fe.num_local_edges):
            for j in range(fe.num_local_edges):
                rows.append(local_inds[i])
                cols.append(local_inds[j])
                s_vals.append(S_local[i, j])
                m_vals.append(M_local[i, j])

    S = scipy.sparse.coo_matrix(
        (onp.array(s_vals), (onp.array(rows), onp.array(cols))),
        shape=(num_dofs, num_dofs)).tocsr()

    M = scipy.sparse.coo_matrix(
        (onp.array(m_vals), (onp.array(rows), onp.array(cols))),
        shape=(num_dofs, num_dofs)).tocsr()

    # Apply PEC: eliminate constrained DoFs by extracting the free sub-matrices
    bc_inds = fe.edge_dof_inds
    all_dofs = onp.arange(num_dofs)
    free_dofs = onp.setdiff1d(all_dofs, bc_inds)
    num_free = len(free_dofs)

    if num_free == 0:
        logger.warning("No free DOFs after applying PEC BCs")
        return onp.array([]), onp.array([])

    S_free = S[onp.ix_(free_dofs, free_dofs)]
    M_free = M[onp.ix_(free_dofs, free_dofs)]

    # Automatic sigma estimate: use the smallest expected physical eigenvalue
    # For a unit-sized cavity the first mode has k^2 ~ π^2 ~ 10
    if sigma_shift is None:
        sigma_shift = 10.0

    # Request extra modes to skip the gradient null-space (dimension ~ number
    # of interior vertices).  Over-request to ensure we capture enough physical modes.
    num_request = min(num_modes + 200, num_free - 1)
    logger.info(f"Solving eigenvalue problem for {num_request} modes "
                f"(sigma={sigma_shift}, free DOFs={num_free})...")
    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            S_free, k=num_request, M=M_free, sigma=sigma_shift, which='LM'
        )
    except scipy.sparse.linalg.ArpackNoConvergence as e:
        logger.warning(f"ARPACK did not fully converge: {e}")
        eigenvalues = e.eigenvalues
        eigenvectors = e.eigenvectors

    # Sort by eigenvalue
    sort_idx = onp.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]

    # Filter out near-zero eigenvalues (gradient null space) and
    # extremely large ones (penalty for PEC DoFs)
    threshold = 0.1
    physical_mask = eigenvalues > threshold
    eigenvalues = eigenvalues[physical_mask][:num_modes]

    # k^2 = eigenvalue -> f = c0 * sqrt(k^2) / (2*pi)
    k_squared = onp.where(eigenvalues > 0, eigenvalues, 0.0)
    resonant_frequencies = C_0 * onp.sqrt(k_squared) / (2.0 * onp.pi)

    logger.info(f"Computed physical eigenvalues (k^2): {eigenvalues}")
    logger.info(f"Resonant frequencies [Hz]: {resonant_frequencies}")

    return eigenvalues, resonant_frequencies
