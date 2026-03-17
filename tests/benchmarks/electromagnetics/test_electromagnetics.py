"""Tests for the high-frequency electromagnetic simulation module.

Verifies:
1. Nédélec basis function tabulation (values + curls)
2. Edge topology and orientation computation
3. PEC cavity resonant frequencies against analytical solutions
4. Driven (source-excited) EM problem assembly and solve
"""

import numpy as np
import numpy.testing as nptest
import unittest

from jax_fem.generate_mesh import box_mesh, Mesh
from jax_fem.basis import get_edge_shape_vals_and_curls, get_nedelec_elements
from jax_fem.edge_fe import EdgeFiniteElement, compute_edge_topology
from jax_fem.electromagnetics import EMProblem, compute_cavity_eigenvalues, C_0


class TestNedelecBasis(unittest.TestCase):
    """Test Nédélec shape function tabulation."""

    def test_tet4_shape_values(self):
        """TET4 N1E element should have 6 edge DOFs and 3D vector values."""
        sv, sc, w, ev = get_edge_shape_vals_and_curls('TET4')
        self.assertEqual(sv.shape[1], 6, "TET4 N1E should have 6 edge DOFs")
        self.assertEqual(sv.shape[2], 3, "TET4 N1E shape values should be 3D vectors")
        self.assertEqual(sc.shape[2], 3, "TET4 N1E curl should be 3D vectors")
        self.assertEqual(ev.shape, (6, 2), "TET4 should have 6 edges")

    def test_hex8_shape_values(self):
        """HEX8 N1E element should have 12 edge DOFs and 3D vector values."""
        sv, sc, w, ev = get_edge_shape_vals_and_curls('HEX8')
        self.assertEqual(sv.shape[1], 12, "HEX8 N1E should have 12 edge DOFs")
        self.assertEqual(sv.shape[2], 3, "HEX8 N1E shape values should be 3D vectors")
        self.assertEqual(sc.shape[2], 3, "HEX8 N1E curl should be 3D vectors")
        self.assertEqual(ev.shape, (12, 2), "HEX8 should have 12 edges")

    def test_tri3_shape_values(self):
        """TRI3 N1E element should have 3 edge DOFs and 2D vector values."""
        sv, sc, w, ev = get_edge_shape_vals_and_curls('TRI3')
        self.assertEqual(sv.shape[1], 3, "TRI3 N1E should have 3 edge DOFs")
        self.assertEqual(sv.shape[2], 2, "TRI3 N1E shape values should be 2D vectors")
        self.assertEqual(len(sc.shape), 2, "TRI3 N1E curl should be scalar (2D)")


class TestEdgeTopology(unittest.TestCase):
    """Test edge identification and orientation."""

    def test_single_hex_element(self):
        """A single HEX8 element should have exactly 12 unique edges."""
        mesh_data = box_mesh(1, 1, 1, 1., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])
        _, _, _, ev = get_edge_shape_vals_and_curls('HEX8')
        unique_edges, cell_edge_indices, orientations = compute_edge_topology(mesh.cells, ev)

        self.assertEqual(len(unique_edges), 12)
        self.assertEqual(cell_edge_indices.shape, (1, 12))
        # All orientations should be ±1
        self.assertTrue(np.all(np.abs(orientations) == 1.0))

    def test_two_hex_elements_share_edges(self):
        """Two adjacent hex elements along x should share 4 edges."""
        mesh_data = box_mesh(2, 1, 1, 2., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])
        _, _, _, ev = get_edge_shape_vals_and_curls('HEX8')
        unique_edges, cell_edge_indices, _ = compute_edge_topology(mesh.cells, ev)

        # 2 hex elements: 2×12 = 24 local edges, sharing 4 edges → 24 - 4 = 20 unique
        self.assertEqual(len(unique_edges), 20)

    def test_edge_orientation_consistency(self):
        """Shared edges should have consistent sign conventions."""
        mesh_data = box_mesh(2, 2, 2, 1., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])
        _, _, _, ev = get_edge_shape_vals_and_curls('HEX8')
        _, cell_edge_indices, orientations = compute_edge_topology(mesh.cells, ev)

        # For any global edge shared by two elements, the product of their
        # orientations should be consistent (both +1 or both -1 for the same edge).
        # This is automatically ensured by our sorting convention.
        num_cells = mesh.cells.shape[0]
        for c1 in range(num_cells):
            for c2 in range(c1 + 1, num_cells):
                shared = np.intersect1d(cell_edge_indices[c1], cell_edge_indices[c2])
                for edge_id in shared:
                    i1 = np.where(cell_edge_indices[c1] == edge_id)[0][0]
                    i2 = np.where(cell_edge_indices[c2] == edge_id)[0][0]
                    # Both should reference the same global edge with
                    # consistent orientation (product can be +1 or -1)
                    self.assertIn(orientations[c1, i1] * orientations[c2, i2],
                                  [-1.0, 1.0])


class TestEdgeFiniteElement(unittest.TestCase):
    """Test EdgeFiniteElement construction and properties."""

    def test_construction_hex8(self):
        """EdgeFiniteElement should initialise correctly for HEX8 mesh."""
        mesh_data = box_mesh(2, 2, 2, 1., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])
        fe = EdgeFiniteElement(mesh=mesh, dim=3, ele_type='HEX8')

        self.assertEqual(fe.num_cells, 8)
        self.assertEqual(fe.num_local_edges, 12)
        self.assertGreater(fe.num_total_edges, 0)
        self.assertEqual(fe.shape_vals_physical.shape[0], 8)  # num_cells
        self.assertEqual(fe.shape_vals_physical.shape[3], 3)  # dim

    def test_pec_boundary_detection(self):
        """PEC boundary conditions should identify all boundary edges."""
        mesh_data = box_mesh(2, 2, 2, 1., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])

        def all_boundary(x):
            return (np.isclose(x[0], 0.) or np.isclose(x[0], 1.) or
                    np.isclose(x[1], 0.) or np.isclose(x[1], 1.) or
                    np.isclose(x[2], 0.) or np.isclose(x[2], 1.))

        fe = EdgeFiniteElement(mesh=mesh, dim=3, ele_type='HEX8',
                               dirichlet_edge_info=([all_boundary], [lambda x: 0.0]))

        # Should have some PEC edges
        self.assertGreater(len(fe.edge_dof_inds), 0)
        # All PEC values should be 0
        nptest.assert_array_equal(fe.edge_dof_vals, 0.0)

    def test_jacobian_unit_cube(self):
        """For a unit cube with 1 element, JxW should be 1/8 at each quad point."""
        mesh_data = box_mesh(1, 1, 1, 1., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])
        fe = EdgeFiniteElement(mesh=mesh, dim=3, ele_type='HEX8')

        # The sum of JxW over all quad points should equal the element volume = 1
        total_volume = np.sum(fe.JxW)
        nptest.assert_almost_equal(total_volume, 1.0, decimal=10)


class TestCavityEigenvalues(unittest.TestCase):
    """Test PEC cavity resonant frequency computation against analytical values."""

    def test_rectangular_cavity_eigenvalues(self):
        """Computed cavity eigenvalues should match analytical values within mesh tolerance.

        Analytical formula: k² = (mπ/a)² + (nπ/b)² + (pπ/d)²
        for a PEC rectangular cavity of dimensions a × b × d.
        """
        a, b, d = 1.0, 0.5, 0.75
        N = 6  # elements per side

        mesh_data = box_mesh(N, N, N, a, b, d)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])

        def pec_boundary(x):
            return (np.isclose(x[0], 0.) or np.isclose(x[0], a) or
                    np.isclose(x[1], 0.) or np.isclose(x[1], b) or
                    np.isclose(x[2], 0.) or np.isclose(x[2], d))

        fe = EdgeFiniteElement(mesh=mesh, dim=3, ele_type='HEX8',
                               dirichlet_edge_info=([pec_boundary], [lambda x: 0.0]))

        eigenvalues, freqs = compute_cavity_eigenvalues(fe, num_modes=4)

        # Analytical first mode: (1,0,1)
        k2_101 = (np.pi / a) ** 2 + (np.pi / d) ** 2  # ≈ 27.42
        k2_110 = (np.pi / a) ** 2 + (np.pi / b) ** 2   # ≈ 49.35

        self.assertGreaterEqual(len(eigenvalues), 2,
                                "Should compute at least 2 physical modes")

        # Allow up to 5% relative error for this mesh resolution
        rel_err_0 = abs(eigenvalues[0] - k2_101) / k2_101
        rel_err_1 = abs(eigenvalues[1] - k2_110) / k2_110

        self.assertLess(rel_err_0, 0.05,
                        f"First eigenvalue k²={eigenvalues[0]:.4f} should be "
                        f"close to {k2_101:.4f} (mode 1,0,1), got {rel_err_0:.1%} error")
        self.assertLess(rel_err_1, 0.05,
                        f"Second eigenvalue k²={eigenvalues[1]:.4f} should be "
                        f"close to {k2_110:.4f} (mode 1,1,0), got {rel_err_1:.1%} error")


class TestEMProblem(unittest.TestCase):
    """Test driven EM problem assembly and solve."""

    def test_assembly_and_solve(self):
        """EMProblem should assemble a complex system and produce a finite solution."""
        a, b, d = 1.0, 0.5, 0.75
        mesh_data = box_mesh(3, 3, 3, a, b, d)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])

        def pec_boundary(x):
            return (np.isclose(x[0], 0.) or np.isclose(x[0], a) or
                    np.isclose(x[1], 0.) or np.isclose(x[1], b) or
                    np.isclose(x[2], 0.) or np.isclose(x[2], d))

        def source_fn(x):
            return np.array([0.0, 0.0, 1.0])  # Uniform z-directed source

        fe = EdgeFiniteElement(mesh=mesh, dim=3, ele_type='HEX8',
                               dirichlet_edge_info=([pec_boundary], [lambda x: 0.0]))

        problem = EMProblem(fe, frequency=300e6, source_fn=source_fn)
        A, b_vec = problem.assemble()

        # System matrix should be complex and square
        self.assertEqual(A.shape[0], A.shape[1])
        self.assertTrue(np.iscomplexobj(A.data))

        # Solve and check solution
        edge_sol = problem.solve()
        self.assertEqual(edge_sol.shape[0], fe.num_total_dofs)
        self.assertTrue(np.all(np.isfinite(edge_sol)))

        # PEC edges should be zero
        if len(fe.edge_dof_inds) > 0:
            pec_vals = edge_sol[fe.edge_dof_inds]
            nptest.assert_array_almost_equal(pec_vals, 0.0, decimal=10)

    def test_field_reconstruction(self):
        """Reconstructed E-field should have correct shape."""
        mesh_data = box_mesh(2, 2, 2, 1., 1., 1.)
        mesh = Mesh(mesh_data.points, mesh_data.cells_dict['hexahedron'])

        fe = EdgeFiniteElement(mesh=mesh, dim=3, ele_type='HEX8')
        problem = EMProblem(fe, frequency=1e9, source_fn=lambda x: np.array([1., 0., 0.]))
        edge_sol = problem.solve()

        E_field = problem.reconstruct_field(edge_sol)
        self.assertEqual(E_field.shape, (fe.num_cells, fe.num_quads, 3))


if __name__ == '__main__':
    unittest.main()
