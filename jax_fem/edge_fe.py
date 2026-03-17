"""Edge (H(curl)) finite element for Nédélec elements.

This module provides :class:`EdgeFiniteElement`, which maps degrees of freedom to
mesh edges rather than nodes.  It is the fundamental building block for
high-frequency electromagnetic simulations.
"""

import numpy as onp
import jax
import jax.numpy as np
import time
from dataclasses import dataclass

from jax_fem.generate_mesh import Mesh
from jax_fem.basis import get_edge_shape_vals_and_curls, get_nedelec_elements
from jax_fem import logger


def compute_edge_topology(cells, edge_vertices_ref):
    """Build the global edge table and local-to-global edge mapping.

    Parameters
    ----------
    cells : ndarray, shape (num_cells, num_nodes)
        Global node indices for each element.
    edge_vertices_ref : ndarray, shape (num_local_edges, 2)
        Reference-element local vertex indices defining each edge.

    Returns
    -------
    unique_edges : ndarray, shape (num_unique_edges, 2)
        Sorted global vertex pairs for every unique edge.
    cell_edge_indices : ndarray, shape (num_cells, num_local_edges)
        Maps each cell's local edge to its global edge index.
    edge_orientations : ndarray, shape (num_cells, num_local_edges)
        +1 if the local edge orientation matches the global orientation, -1 otherwise.
    """
    num_cells = cells.shape[0]
    num_local_edges = edge_vertices_ref.shape[0]

    # Build all cell edges with global vertex indices
    # cell_edges_v0[c, e] = cells[c, edge_vertices_ref[e, 0]]
    cell_edges_v0 = cells[:, edge_vertices_ref[:, 0]]  # (num_cells, num_local_edges)
    cell_edges_v1 = cells[:, edge_vertices_ref[:, 1]]  # (num_cells, num_local_edges)

    # Stack into (num_cells * num_local_edges, 2)
    all_edges = onp.stack([cell_edges_v0.ravel(), cell_edges_v1.ravel()], axis=1)

    # Define global orientation: sort so that smaller vertex index comes first
    sorted_edges = onp.sort(all_edges, axis=1)

    # Find unique edges
    unique_edges, inverse_indices = onp.unique(sorted_edges, axis=0, return_inverse=True)

    cell_edge_indices = inverse_indices.reshape(num_cells, num_local_edges)

    # Compute orientation: +1 if local edge direction matches global (sorted), -1 otherwise
    # Local direction: v0 -> v1.  Global direction: smaller -> larger vertex index.
    local_matches_global = (all_edges[:, 0] < all_edges[:, 1]).astype(onp.float64)
    # If local v0 < v1, orientation = +1, else -1
    edge_orientations = (2.0 * local_matches_global - 1.0).reshape(num_cells, num_local_edges)

    return unique_edges, cell_edge_indices, edge_orientations


@dataclass
class EdgeFiniteElement:
    """Finite element with edge-based (H(curl)) degrees of freedom.

    This element uses Nédélec basis functions where DoFs live on edges
    rather than nodes.  It is essential for electromagnetic simulations
    to avoid spurious modes.

    Attributes
    ----------
    mesh : Mesh
        Stores points (coordinates) and cells (connectivity).
    dim : int
        Spatial dimension (2 or 3).
    ele_type : str
        Element type (e.g. ``'TET4'``, ``'HEX8'``).
    gauss_order : int, optional
        Override for quadrature order.
    dirichlet_edge_info : list, optional
        ``[location_fns, value_fns]`` for essential (PEC) boundary conditions
        on edges.  ``location_fns`` are callables that take a point and return
        True if the point is on the boundary.  ``value_fns`` return the
        prescribed tangential value (typically 0 for PEC).
    """
    mesh: Mesh
    dim: int
    ele_type: str
    gauss_order: int = None
    dirichlet_edge_info: list = None

    def __post_init__(self):
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)

        start = time.time()
        logger.debug("Computing Nédélec shape functions, curls, and edge topology...")

        # Nédélec basis on reference element
        (self.shape_vals, self.shape_curls_ref,
         self.quad_weights, self.edge_vertices_ref) = get_edge_shape_vals_and_curls(
            self.ele_type, self.gauss_order)

        self.num_quads = self.shape_vals.shape[0]
        self.num_local_edges = self.shape_vals.shape[1]

        # Build global edge topology
        (self.unique_edges, self.cell_edge_indices,
         self.edge_orientations) = compute_edge_topology(self.cells, self.edge_vertices_ref)

        self.num_total_edges = len(self.unique_edges)
        # Each edge carries exactly one DoF for lowest-order Nédélec
        self.num_total_dofs = self.num_total_edges
        self.vec = 1  # scalar DoF per edge (tangential component)

        # Compute Jacobians and physical shape function values / curls
        self.shape_curls_physical, self.shape_vals_physical, self.JxW = \
            self._compute_physical_quantities()

        # PEC (Dirichlet) boundary conditions
        self.edge_dof_inds, self.edge_dof_vals = self._dirichlet_edge_bc()

        end = time.time()
        logger.debug(f"Edge FE pre-computation took {end - start:.3f} [s]")
        logger.info(f"Edge FE: {self.num_cells} cells, {self.num_total_edges} edges "
                     f"({self.num_total_dofs} DoFs), {self.num_quads} quads/cell")

    # ------------------------------------------------------------------
    # Physical shape functions and curls via Piola mapping
    # ------------------------------------------------------------------
    def _compute_physical_quantities(self):
        """Map reference Nédélec basis to physical coordinates.

        For H(curl) elements the covariant Piola mapping gives:
            N_phys = J^{-T} N_ref
            curl(N_phys) = (1 / det(J)) J curl(N_ref)

        Returns
        -------
        shape_curls_physical : ndarray
            (num_cells, num_quads, num_local_edges, dim) in 3-D or
            (num_cells, num_quads, num_local_edges) in 2-D.
        shape_vals_physical : ndarray
            (num_cells, num_quads, num_local_edges, dim).
        JxW : ndarray
            (num_cells, num_quads).
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)

        # Reference shape grads for Jacobian – use a simple linear mapping
        # We need the Jacobian dx/dξ.  For Nédélec elements we still compute it from
        # the Lagrange (geometry) element.
        from jax_fem.basis import get_shape_vals_and_grads
        _, shape_grads_ref_lag, _ = get_shape_vals_and_grads(self.ele_type, self.gauss_order)

        # Jacobian: (num_cells, num_quads, dim, dim)
        # J_ij = sum_a x_a_i * dN^lag_a / dξ_j
        jacobian = onp.einsum('cai,qaj->cqij',
                              physical_coos, shape_grads_ref_lag)  # (num_cells, num_quads, dim, dim)

        det_J = onp.linalg.det(jacobian)  # (num_cells, num_quads)
        inv_J = onp.linalg.inv(jacobian)  # (num_cells, num_quads, dim, dim)

        JxW = det_J * self.quad_weights[None, :]

        # Covariant Piola: N_phys = J^{-T} N_ref
        # shape_vals: (num_quads, num_local_edges, dim)
        # inv_J^T: (num_cells, num_quads, dim, dim)
        inv_JT = onp.transpose(inv_J, axes=(0, 1, 3, 2))  # (C, Q, dim, dim)
        shape_vals_physical = onp.einsum('cqij,qej->cqei',
                                         inv_JT, self.shape_vals)  # (C, Q, E, dim)

        if self.dim == 3:
            # curl_phys = (1/det(J)) * J * curl_ref
            shape_curls_physical = onp.einsum('cqij,qej->cqei',
                                              jacobian, self.shape_curls_ref) / det_J[:, :, None, None]
        elif self.dim == 2:
            # In 2D, curl_ref is scalar: (Q, E)
            # curl_phys = curl_ref / det(J)
            shape_curls_physical = self.shape_curls_ref[None, :, :] / det_J[:, :, None]
        else:
            raise ValueError(f"Unsupported dimension {self.dim}")

        return shape_curls_physical, shape_vals_physical, JxW

    # ------------------------------------------------------------------
    # Dirichlet (PEC) boundary conditions on edges
    # ------------------------------------------------------------------
    def _dirichlet_edge_bc(self):
        """Identify boundary edges and set their DoF values (typically 0 for PEC).

        Returns
        -------
        edge_dof_inds : ndarray
            Global edge indices on the boundary.
        edge_dof_vals : ndarray
            Prescribed DoF values (complex).
        """
        edge_dof_inds = onp.array([], dtype=onp.int32)
        edge_dof_vals = onp.array([], dtype=onp.complex128)

        if self.dirichlet_edge_info is None:
            return edge_dof_inds, edge_dof_vals

        location_fns, value_fns = self.dirichlet_edge_info

        for loc_fn, val_fn in zip(location_fns, value_fns):
            # An edge is on the boundary if both its vertices satisfy the location function
            midpoints = 0.5 * (self.points[self.unique_edges[:, 0]] +
                               self.points[self.unique_edges[:, 1]])
            on_boundary = onp.array([
                bool(loc_fn(self.points[v0])) and bool(loc_fn(self.points[v1]))
                for v0, v1 in self.unique_edges
            ])
            inds = onp.where(on_boundary)[0].astype(onp.int32)
            vals = onp.array([complex(val_fn(midpoints[i])) for i in inds],
                             dtype=onp.complex128)
            edge_dof_inds = onp.concatenate([edge_dof_inds, inds])
            edge_dof_vals = onp.concatenate([edge_dof_vals, vals])

        return edge_dof_inds, edge_dof_vals

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------
    def convert_edge_dofs_to_quad(self, edge_sol):
        """Interpolate edge DoF values to quadrature points as a vector field.

        Parameters
        ----------
        edge_sol : ndarray, shape (num_total_edges,)
            One scalar DoF per global edge.

        Returns
        -------
        u : ndarray, shape (num_cells, num_quads, dim)
            Reconstructed vector field at quadrature points.
        """
        # Gather cell-local edge solutions  (num_cells, num_local_edges)
        cell_edge_sols = edge_sol[self.cell_edge_indices] * self.edge_orientations

        # shape_vals_physical: (num_cells, num_quads, num_local_edges, dim)
        # cell_edge_sols: (num_cells, num_local_edges)
        u = onp.einsum('cqed,ce->cqd', self.shape_vals_physical, cell_edge_sols)
        return u
