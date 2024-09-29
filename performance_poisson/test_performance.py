# Copyright (C) 2024 Maximilian Brodbeck
# SPDX-License-Identifier:    LGPL-3.0-or-later

""" Performance test for the Poisson equation

Solves
        -div(grad(u)) = f with f = -grad(u_ext)
based on the exact solution
        u_ext = sin(2*pi * x) * cos(2*pi * y).
Dirichlet boundary conditions are applied on the entire boundary.
Performance is measured on a series on uniformly refined meshes.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

from dolfinx import fem, mesh
import ufl

from dolfinx_eqlb.cpp import local_solver_cholesky
from dolfinx_eqlb.eqlb import FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection


# --- The exact solution
def exact_solution(pkt):
    """Exact solution
    u_ext = sin(pi * x) * cos(pi * y)

    Args:
        pkt: The package

    Returns:
        The function handle oft the exact solution
    """
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


# --- Mesh generation
def create_unit_square_builtin(
    n_elmt: int,
) -> typing.Tuple[mesh.Mesh, mesh.MeshTagsMetaClass, ufl.Measure]:
    """Create a unit square using the build-in mesh generator

                    4
      -     |---------------|
      |     |               |
      |     |               |
    1 |   1 |               | 3
      |     |               |
      |     |               |
      -     |---------------|
                    2

            '-------1-------'

    Args:
        n_elmt: The number of elements in each direction

    Returns:
        The mesh,
        The facet tags,
        The tagged surface measure
    """

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [n_elmt, n_elmt],
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.crossed,
    )

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = mesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    return domain, facet_tag, ds


# --- The primal problem
def solve(
    order_prime: int,
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    pdegree_rhs: typing.Optional[int] = None,
) -> typing.Tuple[fem.Function, typing.List[float]]:
    """Solves the Poisson problem based on lagrangian finite elements

    Args:
        order_prime: The order of the FE space
        domain:      The mesh
        facet_tags:  The facet tags
        pdegree_rhs: The degree of the DG space into which the RHS
                     is projected into

    Returns:
        The solution
        The timings
    """

    # Set function space (primal problem)
    V_prime = fem.FunctionSpace(domain, ("CG", order_prime))
    uh = fem.Function(V_prime)

    # Set trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    if pdegree_rhs is None:
        rhs = f
    else:
        rhs = local_projection(fem.FunctionSpace(domain, ("DG", pdegree_rhs)), [f])[0]

    # Equation system
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    l = fem.form(rhs * v * ufl.dx)

    # Dirichlet boundary conditions
    uD = fem.Function(V_prime)
    uD.interpolate(exact_solution(np))

    dofs_essnt = fem.locate_dofs_topological(V_prime, 1, facet_tags.indices[:])
    bcs_essnt = [fem.dirichletbc(uD, dofs_essnt)]

    # --- Solve primal problem

    timing_assemble = -time.perf_counter()
    # Assemble system matrix
    A = fem.petsc.assemble_matrix(a, bcs=bcs_essnt)
    A.assemble()

    # Assemble right-hand side
    L = fem.petsc.create_vector(l)
    fem.petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [bcs_essnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bcs_essnt)
    timing_assemble += time.perf_counter()

    # Set solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    solver.setTolerances(rtol=1e-12, atol=1e-12, max_it=1000)

    # Solve primal problem
    timing_solve = -time.perf_counter()
    solver.solve(L, uh.vector)
    timing_solve += time.perf_counter()

    return uh, [timing_assemble, timing_solve]


# --- The equilibration
def projection(
    order_eqlb: int,
    domain: mesh.Mesh,
    uh: fem.Function,
) -> float:
    """Projects flux and RHS

    The RHS is assumed to be the divergence of the exact
    flux (manufactured solution).

    Args:
        Equilibrator:        The flux equilibrator
        order_eqlb:          The order of the RT space
        domain:              The mesh
        facet_tags:          The facet tags
        bc_type:             The type of BCs
        uh:                  The primal solution
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The timings
    """

    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    # Set the flux
    flux = -ufl.grad(uh)

    # Project flux and RHS into required DG space
    V_proj = fem.FunctionSpace(domain, ("DG", order_eqlb - 1))

    # The variational problem
    u = ufl.TrialFunction(V_proj)
    v = ufl.TestFunction(V_proj)

    # Bilinear form
    a = fem.form(ufl.inner(u, v) * ufl.dx)
    l1 = fem.form(ufl.inner(flux[0], v) * ufl.dx)
    l2 = fem.form(ufl.inner(flux[1], v) * ufl.dx)
    l3 = fem.form(ufl.inner(f, v) * ufl.dx)

    # Set up the projector
    list_u = [fem.Function(V_proj) for _ in range(3)]
    list_ucpp = [u._cpp_object for u in list_u]

    # Solve the projection
    timing_proj = -time.perf_counter()
    local_solver_cholesky(list_ucpp, a, [l1, l2, l3])
    timing_proj += time.perf_counter()

    return timing_proj


def equilibrate(
    order_eqlb: int,
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    uh: fem.Function,
) -> float:
    """Equilibrate the flux

    The RHS is assumed to be the divergence of the exact
    flux (manufactured solution).

    Args:
        Equilibrator:        The flux equilibrator
        order_eqlb:          The order of the RT space
        domain:              The mesh
        facet_tags:          The facet tags
        bc_type:             The type of BCs
        uh:                  The primal solution
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The timings
    """

    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    # Project flux and RHS into required DG space
    V_rhs_proj = fem.FunctionSpace(domain, ("DG", order_eqlb - 1))
    V_flux_proj = fem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))

    sigma_proj = local_projection(V_flux_proj, [-ufl.grad(uh)])
    rhs_proj = local_projection(V_rhs_proj, [f])

    # --- The equilibration
    # Initialise equilibrator
    equilibrator = FluxEqlbSE(order_eqlb, domain, rhs_proj, sigma_proj)
    equilibrator.set_boundary_conditions([facet_tags.indices], [[]])

    # Solve equilibration
    timing_eqlb = -time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing_eqlb += time.perf_counter()

    return timing_eqlb


#  --- Perform timing
def time_problem(order_prime: int, order_eqlb: int, nref: int, nretry: int):
    """Solution procedure with timing

    Args:
        order_prime:   The order of the fe-space for the primal problem
        order_eqlb:    The order of the RT space for the equilibration
        nref:          The number of refinements
        nretry:        The number of retries
    """

    # Storage results
    outname_base = "Timing_Poisson-P{}_RT{}".format(order_prime, order_eqlb)
    results = np.zeros((nref, 10))

    for n in range(nretry):
        print("Retry {}/{}".format(n + 1, nretry))
        for i in range(nref):
            # --- Create mesh
            print("Mesh {}/{}".format(i + 1, nref))
            # Set mesh resolution
            nelmt = 2**i

            # Create mesh
            domain, facet_tags, ds = create_unit_square_builtin(nelmt)

            # --- Solve problem
            # Solve primal problem
            degree_proj = 0 if (order_prime == 1) else None
            uh, timings_prime = solve(order_prime, domain, facet_tags, degree_proj)

            # Project flux and RHS
            timing_proj = projection(order_eqlb, domain, uh)

            # Equilibrate flux
            timing_eqlb = equilibrate(order_eqlb, domain, facet_tags, uh)

            # Store data
            results[i, 0] = domain.topology.index_map(2).size_local
            results[i, 1] = uh.function_space.dofmap.index_map.size_global

            if n == 0:
                results[i, 2] = timings_prime[0]
                results[i, 3] = timings_prime[1]
                results[i, 5] = timing_proj
                results[i, 6] = timing_eqlb
            else:
                results[i, 2] = min(results[i, 2], timings_prime[0])
                results[i, 3] = min(results[i, 3], timings_prime[1])
                results[i, 5] = min(results[i, 5], timing_proj)
                results[i, 6] = min(results[i, 6], timing_eqlb)

    # Post-process timings
    results[:, 4] = results[:, 2] + results[:, 3]
    results[:, 7] = results[:, 6] + results[:, 5]
    results[:, 8] = results[:, 6] / results[:, 4]
    results[:, 9] = results[:, 7] / results[:, 4]

    # Export results to csv
    header_protocol = "nelmt, ndofs, tassmbl, tsolve, tprime, tproj, teqlb, teqlbt, teqlbbytprime, teqlbtbytprime"
    np.savetxt(outname_base + ".csv", results, delimiter=",", header=header_protocol)


if __name__ == "__main__":
    # --- Parameters ---
    nretry = 5
    # ------------------

    # Solve based on P1 with equilibration in RT1
    time_problem(1, 1, 11, nretry)

    # Solve based on P1 with equilibration in RT2
    time_problem(1, 2, 11, nretry)

    # Solve based on P2 with equilibration in RT2
    time_problem(2, 2, 10, nretry)

    # Solve based on P2 with equilibration in RT3
    time_problem(2, 3, 10, nretry)
