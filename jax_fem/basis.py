import basix
import numpy as onp

from jax_fem import logger


# def get_full_integration_poly_degree(ele_type, lag_order, dim):
#     """Only works for weak forms of (grad_u, grad_v).
#     TODO: Is this correct?
#     Reference:
#     https://zhuanlan.zhihu.com/p/521630645
#     """
#     if ele_type == 'hexahedron' or ele_type == 'quadrilateral':
#         return 2 * (dim*lag_order - 1)

#     if ele_type == 'tetrahedron' or ele_type == 'triangle':
#         return 2 * (dim*(lag_order - 1) - 1)

def get_elements(ele_type):
    """Obtain element information useful for `basix <https://github.com/FEniCS/basix>`_ to handle. 

    Note that mesh node ordering is important.
    If the input mesh file is Gmsh .msh or Abaqus .inp, meshio would convert it to its own ordering.
    Our experience shows that meshio ordering is the same as Abaqus.
    For example, for a 10-node tetrahedron element, the ordering of meshio follows this 
    `instruction <https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html>`_.
    The troublesome thing is that basix has `a different ordering <https://defelement.com/elements/lagrange.html>`_. 
    The consequence is that we need to define the  **re_order** variable to make sure the ordering is correct.


    Parameters
    ----------
    ele_type: str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`

    Returns
    -------
    element_family : BasixObject
        `ElementFamily <https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.ElementFamily>`_
    basix_ele : BasixObject
        For element: `CellType <https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.CellType>`_
    basix_face_ele : BasixObject
        For element face: `CellType <https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.CellType>`_
    gauss_order : int
        :attr:`~jax_fem.fe.FiniteElement.gauss_order`
    degree : int
        Element degree, used in basix
    re_order : list
        Specifieds node re-ordering transformation. For example, [0, 1, 3, 2, 4, 5, 7, 6] for HEX8 element.
    """
    element_family = basix.ElementFamily.P
    if ele_type == 'HEX8':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 2x2x2, TODO: is this full integration?
        degree = 1
    elif ele_type == 'HEX27':
        print(f"Warning: 27-node hexahedron is rarely used in practice and not recommended.")
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                    17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 10 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'HEX20':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'TET4':
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TET10':
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2 # 4, full integration
        degree = 2
    # TODO: Check if this is correct.
    elif ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'QUAD8':
        re_order = [0, 1, 3, 2, 4, 6, 7, 5]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 2
    elif ele_type == 'TRI3':
        re_order = [0, 1, 2]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TRI6':
        re_order = [0, 1, 2, 5, 3, 4]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2 # 3, full integration
        degree = 2
    else:
        raise NotImplementedError

    return element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order


def get_nedelec_elements(ele_type):
    """Obtain Nédélec (edge) element information for H(curl)-conforming finite elements.

    These elements associate degrees of freedom with mesh edges (and faces for
    higher order) instead of nodes.  They are required for high-frequency
    electromagnetic simulations to avoid spurious modes.

    Parameters
    ----------
    ele_type : str
        Element type.  Currently supported: ``'TET4'``, ``'HEX8'``, ``'TRI3'``, ``'QUAD4'``.

    Returns
    -------
    basix_ele : BasixObject
        Basix ``CellType``.
    basix_face_ele : BasixObject
        Basix ``CellType`` for the element face.
    gauss_order : int
        Default Gauss quadrature order.
    degree : int
        Polynomial degree of the Nédélec element (always 1 for lowest order).
    """
    degree = 1
    if ele_type == 'TET4':
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2
    elif ele_type == 'HEX8':
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2
    elif ele_type == 'TRI3':
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
    elif ele_type == 'QUAD4':
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
    else:
        raise NotImplementedError(f"Nédélec elements not implemented for {ele_type}")

    return basix_ele, basix_face_ele, gauss_order, degree


def get_edge_shape_vals_and_curls(ele_type, gauss_order=None):
    """Compute Nédélec (edge) shape function values and their curls on the reference element.

    Parameters
    ----------
    ele_type : str
        Element type (e.g. ``'TET4'``, ``'HEX8'``).
    gauss_order : int, optional
        Override for the default quadrature order.

    Returns
    -------
    shape_values : ndarray
        Shape ``(num_quads, num_edge_dofs, dim)`` – vector-valued basis functions.
    shape_curls_ref : ndarray
        In 3-D, shape ``(num_quads, num_edge_dofs, dim)`` – curl of each basis function.
        In 2-D, shape ``(num_quads, num_edge_dofs)`` – scalar curl (z-component).
    weights : ndarray
        Quadrature weights, shape ``(num_quads,)``.
    edge_vertices : ndarray
        Vertex indices defining each edge, shape ``(num_edges, 2)``.
    """
    basix_ele, basix_face_ele, gauss_order_default, degree = get_nedelec_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
    element = basix.create_element(basix.ElementFamily.N1E, basix_ele, degree)

    dim = quad_points.shape[1]
    # tabulate(1, points) -> (nderivs, num_quads, num_dofs, value_size)
    # nderivs = 1 + dim  (values + partial derivatives w.r.t. each coordinate)
    vals_and_grads = element.tabulate(1, quad_points)

    # Index 0: shape function values  (num_quads, num_dofs, dim)
    shape_values = vals_and_grads[0]

    # Compute curl from partial derivatives
    if dim == 3:
        # vals_and_grads[1] = d/dx, vals_and_grads[2] = d/dy, vals_and_grads[3] = d/dz
        dNdx = vals_and_grads[1]  # (num_quads, num_dofs, 3)
        dNdy = vals_and_grads[2]
        dNdz = vals_and_grads[3]
        # curl_x = dNz/dy - dNy/dz
        # curl_y = dNx/dz - dNz/dx
        # curl_z = dNy/dx - dNx/dy
        curl_x = dNdy[:, :, 2] - dNdz[:, :, 1]
        curl_y = dNdz[:, :, 0] - dNdx[:, :, 2]
        curl_z = dNdx[:, :, 1] - dNdy[:, :, 0]
        shape_curls_ref = onp.stack([curl_x, curl_y, curl_z], axis=-1)
    elif dim == 2:
        # 2D curl is scalar: curl_z = dNy/dx - dNx/dy
        dNdx = vals_and_grads[1]  # (num_quads, num_dofs, 2)
        dNdy = vals_and_grads[2]
        shape_curls_ref = dNdx[:, :, 1] - dNdy[:, :, 0]  # (num_quads, num_dofs)
    else:
        raise ValueError(f"Unsupported dimension {dim}")

    # Edge vertex connectivity from basix topology (in basix vertex ordering).
    # Our mesh cells use Abaqus ordering (via the re_order mapping from get_elements).
    # We must map basix vertex indices → Abaqus positions so that
    # cells[:, edge_vertices[:, k]] returns the correct global node.
    edge_topology = basix.topology(basix_ele)[1]
    edge_vertices_basix = onp.array(edge_topology)

    _, _, _, _, _, re_order = get_elements(ele_type)
    re_order = onp.array(re_order)

    # inv_re_order[basix_vertex] = abaqus_position
    inv_re_order = onp.empty_like(re_order)
    inv_re_order[re_order] = onp.arange(len(re_order))

    edge_vertices = inv_re_order[edge_vertices_basix]

    logger.debug(f"Nédélec ele_type = {ele_type}, quad_points.shape = {quad_points.shape}, "
                 f"num_edge_dofs = {shape_values.shape[1]}")

    return shape_values, shape_curls_ref, weights, edge_vertices


def reorder_inds(inds, re_order):
    """Apply re-ordering transformation for node indices.
    """

    new_inds = []
    for ind in inds.reshape(-1):
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type, gauss_order=None):
    """Use `basix <https://github.com/FEniCS/basix>`_ to get shape function values and gradients for elements.

    Parameters
    ----------
    ele_type : str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`
    gauss_order : int
        :attr:`~jax_fem.fe.FiniteElement.gauss_order`

    Returns
    -------
    shape_values: NumpyArray
        Shape is (num_quads, num_nodes), e.g, (8, 8) for HEX8 element.
    shape_grads_ref: NumpyArray
        Shape is (num_quads, num_nodes, dim), e.g, (8, 8, 3) for HEX8 element.
    weights: NumpyArray
        Shape is (num_quads,), e.g, (8,) for HEX8 element.
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    logger.debug(f"ele_type = {ele_type}, quad_points.shape = (num_quads, dim) = {quad_points.shape}")
    return shape_values, shape_grads_ref, weights


def get_face_shape_vals_and_grads(ele_type, gauss_order=None):
    """Use `basix <https://github.com/FEniCS/basix>`_ to get shape function values and gradients for element faces.

    Parameters
    ----------
    ele_type : str
        :attr:`~jax_fem.fe.FiniteElement.ele_type`
    gauss_order : int
        :attr:`~jax_fem.fe.FiniteElement.gauss_order`

    Returns
    -------
    face_shape_vals: NumpyArray
        Shape is (num_faces, num_face_quads, num_nodes), e.g, (6, 4, 8) for HEX8 element.
    face_shape_grads_ref: NumpyArray
        Shape is(num_faces, num_face_quads, num_nodes, dim), e.g, (6, 4, 3) for HEX8 element.
    face_weights: NumpyArray
        Shape is (num_faces, num_face_quads), e.g, (6, 4) for HEX8 element.
    face_normals:NumpyArray
        Shape is (num_faces, dim), e.g, (6, 3) for HEX8 element.
    face_inds: NumpyArray
        Shape is (num_faces, num_face_vertices), e.g, (6, 4) for HEX8 element.
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    # TODO: Check if this is correct.
    # We should provide freedom for seperate gauss_order for volume integral and surface integral
    # Currently, they're using the same gauss_order!
    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
    # Map face points
    # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    logger.debug(f"face_quad_points.shape = (num_faces, num_face_quads, dim) = {face_quad_points.shape}")
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds
