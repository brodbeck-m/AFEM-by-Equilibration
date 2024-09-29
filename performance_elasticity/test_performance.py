# Copyright (C) 2024 Maximilian Brodbeck
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Performance test for linear elasticity

Solution of the quasi-static linear elasticity equation

     div(sigma) = 0  with sigma = 2 * eps + pi_1 * div(u) * I,

based on the exact solution

        u_ext = [ sin(pi * x) * cos(pi * y) + x²/(2*pi_1),
                 -cos(pi * x) * sin(pi * y) + y²/(2*pi_1)]

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
def exact_solution(x: typing.Any, pi_1: float) -> typing.Any:
    """Exact solution
    u_ext = [ sin(pi * x) * cos(pi * y) + x²/(2*pi_1),
             -cos(pi * x) * sin(pi * y) + y²/(2*pi_1)]

    Args:
        x:    The spatial position
        pi_1: The ratio of lambda and mu

    Returns:
        The exact function as ufl-expression
    """
    return ufl.as_vector(
        [
            ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
            + 0.5 * (x[0] * x[0] / pi_1),
            -ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            + 0.5 * (x[1] * x[1] / pi_1),
        ]
    )


def interpolate_ufl_to_function(f_ufl: typing.Any, f_fe: fem.Function):
    """Interpolates a UFL expression to a function

    Args:
        f_ufl: The function in UFL
        f_fe:  The function to interpolate into
    """

    # Create expression
    expr = fem.Expression(f_ufl, f_fe.function_space.element.interpolation_points())

    # Perform interpolation
    f_fe.interpolate(expr)


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
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    pi_1: float,
    degree: int,
    degree_rhs: typing.Optional[int] = None,
) -> typing.Tuple[
    typing.Union[fem.Function, typing.Any],
    typing.List[fem.Function],
    typing.Any,
    typing.Any,
]:
    """Solves the problem of linear elasticity based on lagrangian finite elements

    Args:
        domain:      The mesh
        facet_tags:  The facet tags
        pi_1:        The ratio of lambda and mu
        degree:      The degree of the FE space
        degree_rhs:  The degree of the DG space into which the RHS
                     is projected
    Returns:
        The right-hand-side,
        The exact stress tensor,
        The approximated solution,
        The approximated stress tensor
        The timings
    """

    # The exact solution
    u_ext = exact_solution(ufl.SpatialCoordinate(domain), pi_1)
    sigma_ext = 2 * ufl.sym(ufl.grad(u_ext)) + pi_1 * ufl.div(u_ext) * ufl.Identity(2)

    # The right-hand-side
    f = -ufl.div(sigma_ext)

    if degree_rhs is None:
        rhs = f
    else:
        V_rhs = fem.VectorFunctionSpace(domain, ("DG", degree_rhs))
        rhs = local_projection(V_rhs, [f])[0]

    # --- Set weak form and BCs
    # Check input
    if degree < 2:
        raise ValueError("Consistency condition for weak symmetry not fulfilled!")

    # The function space
    V = fem.VectorFunctionSpace(domain, ("CG", degree))
    uh = fem.Function(V)

    # Trial- and test-functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # The variational form
    sigma = 2 * ufl.sym(ufl.grad(u)) + pi_1 * ufl.div(u) * ufl.Identity(2)

    a = fem.form(ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx)
    l = fem.form(ufl.inner(rhs, v) * ufl.dx)

    # The Dirichlet BCs
    uD = fem.Function(V)
    interpolate_ufl_to_function(u_ext, uD)

    dofs = fem.locate_dofs_topological(V, 1, facet_tags.indices[:])
    bcs_esnt = [fem.dirichletbc(uD, dofs)]

    # --- Solve the equation system
    timing_assemble = -time.perf_counter()
    # The system matrix
    A = fem.petsc.assemble_matrix(a, bcs=bcs_esnt)
    A.assemble()

    # The right-hand-side
    L = fem.petsc.create_vector(l)
    fem.petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [bcs_esnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bcs_esnt)
    timing_assemble += time.perf_counter()

    # The solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    # solver.setType(PETSc.KSP.Type.PREONLY)
    # pc = solver.getPC()
    # pc.setType(PETSc.PC.Type.LU)
    # pc.setFactorSolverType("mumps")

    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")

    # Solve the system
    solver.setTolerances(rtol=1e-12, atol=1e-12, max_it=1000)

    timing_solve = -time.perf_counter()
    solver.solve(L, uh.vector)
    timing_solve += time.perf_counter()

    # The approximated stress tensor
    sigma_h = 2 * ufl.sym(ufl.grad(uh)) + pi_1 * ufl.div(uh) * ufl.Identity(2)

    return rhs, sigma_ext, uh, sigma_h, [timing_assemble, timing_solve]


# --- The equilibration
def projection(
    order_eqlb: int, domain: mesh.Mesh, f: typing.Any, sigma_h: typing.Any
) -> float:
    """Projects flux and RHS

    The RHS is assumed to be the divergence of the exact
    flux (manufactured solution).

    Args:
        order_eqlb: The order of the RT space
        domain:     The mesh
        f:          The RHS
        sigma_h:    The approximated stress tensor

    Returns:
        The timings
    """

    # Project flux and RHS into required DG space
    V_proj = fem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))

    # The variational problem
    u = ufl.TrialFunction(V_proj)
    v = ufl.TestFunction(V_proj)

    # Bilinear form
    a = fem.form(ufl.inner(u, v) * ufl.dx)
    l1 = fem.form(ufl.inner(-ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]), v) * ufl.dx)
    l2 = fem.form(ufl.inner(-ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]), v) * ufl.dx)
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
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    f: typing.Any,
    sigma_h: typing.Any,
    degree: int,
    weak_symmetry: typing.Optional[bool],
) -> typing.Tuple[typing.Any, typing.Any, fem.Function]:
    """Equilibrates the negative stress-tensor of linear elasticity

    The RHS is assumed to be the divergence of the exact stress
    tensor (manufactured solution).

    Args:
        domain:              The mesh
        facet_tags:          The facet tags
        f:                   The right-hand-side
        sigma_h:             The approximated stress tensor
        degree:              The degree of the RT space
        weak_symmetry:       Id if weak symmetry condition is enforced

    Returns:
        The timing
    """

    # Check input
    if degree < 2:
        raise ValueError("Stress equilibration only possible for k>1")

    # Projected flux
    # (degree - 1 would be sufficient but not implemented for semi-explicit eqlb.)
    V_flux_proj = fem.VectorFunctionSpace(domain, ("DG", degree - 1))
    sigma_proj = local_projection(
        V_flux_proj,
        [
            ufl.as_vector([-sigma_h[0, 0], -sigma_h[0, 1]]),
            ufl.as_vector([-sigma_h[1, 0], -sigma_h[1, 1]]),
        ],
    )

    # Project RHS
    V_rhs_proj = fem.FunctionSpace(domain, ("DG", degree - 1))
    rhs_proj = local_projection(V_rhs_proj, [f[0], f[1]])

    # Initialise equilibrator
    equilibrator = FluxEqlbSE(
        degree,
        domain,
        rhs_proj,
        sigma_proj,
        equilibrate_stress=weak_symmetry,
        estimate_korn_constant=True,
    )

    # Set boundary conditions
    equilibrator.set_boundary_conditions(
        [facet_tags.indices[:], facet_tags.indices[:]], [[], []]
    )

    # Solve equilibration
    timing = 0

    timing -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing += time.perf_counter()

    return timing


#  --- Perform timing
def time_problem(
    order_prime: int, order_eqlb: int, weak_sym: bool, nref: int, nretry: int
):
    """Solution procedure with timing

    Args:
        order_prime:   The order of the fe-space for the primal problem
        order_eqlb:    The order of the RT space for the equilibration
        weak_sym:      True, if the weak symmetry is enforced
        nref:          The number of refinements
        nretry:        The number of retries
    """

    # Storage results
    if weak_sym:
        outname_base = "Timing_Elasticity-P{}_RT{}_gEE".format(order_prime, order_eqlb)
    else:
        outname_base = "Timing_Elasticity-P{}_RT{}_hEI".format(order_prime, order_eqlb)
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
            degree_proj = 1 if (order_prime == 2) else None
            f, stress_ref, u_h, stress_h, timings_prime = solve(
                domain, facet_tags, 2.333, order_prime, degree_proj
            )

            # Project flux and RHS
            timing_proj = projection(order_eqlb, domain, f, stress_h)

            # Equilibrate flux
            timing_eqlb = equilibrate(
                domain, facet_tags, f, stress_h, order_eqlb, weak_sym
            )

            # Store data
            results[i, 0] = domain.topology.index_map(2).size_local
            results[i, 1] = 2 * u_h.function_space.dofmap.index_map.size_global

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

    for is_grntd in [False, True]:
        # Solve based on P2 with equilibration in RT2
        time_problem(2, 2, is_grntd, 10, nretry)

        # Solve based on P2 with equilibration in RT3
        time_problem(2, 3, is_grntd, 10, nretry)

        # Solve based on P3 with equilibration in RT3
        time_problem(3, 3, is_grntd, 9, nretry)

        # Solve based on P3 with equilibration in RT4
        time_problem(3, 4, is_grntd, 9, nretry)
