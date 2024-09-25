"""Solution of linear elasticity with manufactured solution.

Solves

        div(2 * eps + pi_1 * div(u) * I) = -f ,

on a unit-square with inhomogeneous Dirichlet BCs on a series
of uniformly refined meshes. Convergence is reported with res-
pect to the manufactured solution

    u_ext = [ sin(pi * x) * cos(pi * y) + x²/(2*pi_1),
             -cos(pi * x) * sin(pi * y) + y²/(2*pi_1)]
"""

from mpi4py import MPI
import numpy as np
from numpy.typing import NDArray
from petsc4py import PETSc
import time
import typing

from dolfinx import cpp, fem, mesh
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbSE
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
    check_weak_symmetry_condition,
)
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
    timing = 0

    timing -= time.perf_counter()
    # The system matrix
    A = fem.petsc.assemble_matrix(a, bcs=bcs_esnt)
    A.assemble()

    # The right-hand-side
    L = fem.petsc.create_vector(l)
    fem.petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [bcs_esnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bcs_esnt)

    # The solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    # solver.setType(PETSc.KSP.Type.CG)
    # pc = solver.getPC()
    # pc.setType(PETSc.PC.Type.HYPRE)
    # pc.setHYPREType("boomeramg")

    # Solve the system
    solver.setTolerances(rtol=1e-12, atol=1e-12, max_it=1000)
    solver.solve(L, uh.vector)

    timing += time.perf_counter()

    print(f"Primal problem solved in {timing:.4e} s")

    # The approximated stress tensor
    sigma_h = 2 * ufl.sym(ufl.grad(uh)) + pi_1 * ufl.div(uh) * ufl.Identity(2)

    return rhs, sigma_ext, [uh], sigma_h


# --- The equilibration
def equilibrate(
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    f: typing.Any,
    sigma_h: typing.Any,
    degree: int,
    weak_symmetry: typing.Optional[bool] = True,
    check_equilibration: typing.Optional[bool] = True,
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
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The projected stress tensor (ufl tensor),
        The equilibrated stress tensor (ufl tensor),
        The cells Korns constant
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

    print(f"Equilibration solved in {timing:.4e} s")

    # Cast stresses into ufl tensors
    stress_eqlb = ufl.as_matrix(
        [
            [-equilibrator.list_flux[0][0], -equilibrator.list_flux[0][1]],
            [-equilibrator.list_flux[1][0], -equilibrator.list_flux[1][1]],
        ]
    )

    stress_proj = ufl.as_matrix(
        [
            [-sigma_proj[0][0], -sigma_proj[0][1]],
            [-sigma_proj[1][0], -sigma_proj[1][1]],
        ]
    )

    # --- Check equilibration conditions ---
    if check_equilibration:
        V_rhs_proj = fem.VectorFunctionSpace(domain, ("DG", degree - 1))
        rhs_proj_vecval = local_projection(V_rhs_proj, [-f])[0]

        # Check divergence condition
        div_condition_fulfilled = check_divergence_condition(
            stress_eqlb,
            stress_proj,
            rhs_proj_vecval,
            mesh=domain,
            degree=degree,
            flux_is_dg=True,
        )

        if not div_condition_fulfilled:
            raise ValueError("Divergence conditions not fulfilled")

        # Check if flux is H(div)
        for i in range(domain.geometry.dim):
            jump_condition_fulfilled = check_jump_condition(
                equilibrator.list_flux[i], sigma_proj[i]
            )

            if not jump_condition_fulfilled:
                raise ValueError("Jump conditions not fulfilled")

        # Check weak symmetry condition
        wsym_condition = check_weak_symmetry_condition(equilibrator.list_flux)

        if not wsym_condition:
            raise ValueError("Weak symmetry conditions not fulfilled")

    return stress_proj, stress_eqlb, equilibrator.get_korn_constants()


# --- Estimate the error
def estimate(
    domain: mesh.Mesh,
    pi_1: float,
    f: typing.Union[fem.Function, typing.Any],
    sigma_h: typing.Any,
    delta_sigmaR: typing.Any,
    korns_constants: fem.Function,
    guarantied_upper_bound: typing.Optional[bool] = True,
) -> typing.Tuple[float, typing.List[float]]:
    """Estimates the error of a linear elastic problem

    Args:
        domain:                 The mesh
        pi_1:                   The ratio of lambda and mu
        f:                      The exact body forces
        sigma_h:                The projected stress tensor (UFL)
        delta_sigmaR:           The equilibrated stress tensor (UFL)
        korns_constants:        The cells Korn's constants
        guarantied_upper_bound: True, if the error estimate is a guarantied upper bound

    Returns:
        The total error estimate,
        The error components
    """

    # Higher order volume integrator
    dvol = ufl.dx(degree=10)

    # Initialize storage of error
    V_e = fem.FunctionSpace(domain, ufl.FiniteElement("DG", domain.ufl_cell(), 0))
    v = ufl.TestFunction(V_e)

    # Extract cell diameter
    h_cell = fem.Function(V_e)
    num_cells = (
        domain.topology.index_map(2).size_local
        + domain.topology.index_map(2).num_ghosts
    )
    h = cpp.mesh.h(domain, 2, range(num_cells))
    h_cell.x.array[:] = h

    # The error estimate
    a_delta_sigma = 0.5 * (
        delta_sigmaR - (pi_1 / (2 + 2 * pi_1)) * ufl.tr(delta_sigmaR) * ufl.Identity(2)
    )

    err_osc = (
        korns_constants * (h_cell / ufl.pi) * (f + ufl.div(sigma_h + delta_sigmaR))
    )

    err_wsym = 0.5 * korns_constants * (delta_sigmaR[0, 1] - delta_sigmaR[1, 0])

    forms_eta = []
    forms_eta.append(fem.form(ufl.inner(delta_sigmaR, a_delta_sigma) * v * ufl.dx))
    forms_eta.append(fem.form(ufl.inner(err_wsym, err_wsym) * v * ufl.dx))
    forms_eta.append(fem.form(ufl.inner(err_osc, err_osc) * v * dvol))

    # Assemble cell-wise errors
    Li_eta = []
    eta_i = []

    for form in forms_eta:
        Li_eta.append(fem.petsc.create_vector(form))
        fem.petsc.assemble_vector(Li_eta[-1], form)

        eta_i.append(np.sqrt(np.sum(Li_eta[-1].array)))

    # Evaluate error norms
    eta = np.sum(Li_eta[0].array)

    if guarantied_upper_bound:
        eta += np.sum(
            Li_eta[-1].array
            + Li_eta[-2].array
            + 2 * np.multiply(np.sqrt(Li_eta[-1].array), np.sqrt(Li_eta[-2].array))
        )
    else:
        eta += np.sum(Li_eta[-1].array)

    return np.sqrt(eta), eta_i


# --- The solution procedure
def post_process(
    pi_1: float,
    nelmt: int,
    u_h: typing.List[fem.Function],
    sigma_proj: typing.Any,
    delta_sigmaR: typing.Any,
    eta: float,
    eta_i: typing.List[float],
    ref_level: int,
    results: NDArray,
):
    """Postprocess the results

    Args:
        pi_1:           The ratio of lambda and mu
        nelmt:          The number of elements in each direction
        u_h:            The approximated solution
        sigma_proj:     The projected stress tensor
        delta_sigmaR:   The equilibrated stress tensor
        eta:            The total error estimate
        eta_i:          The components of the error estimate
        ref_level:      The current refinement level
        results:        The results array
    """

    # The domain
    domain = u_h[0].function_space.mesh

    # The Volume integrator
    dvol = ufl.dx(degree=10)

    # The exact solution
    u_ext = exact_solution(ufl.SpatialCoordinate(domain), pi_1)
    sigma_ext = 2 * ufl.sym(ufl.grad(u_ext)) + pi_1 * ufl.div(u_ext) * ufl.Identity(2)

    # Energy norm of the displacement
    diff_u = u_h[0] - u_ext
    err_ufl = (
        ufl.inner(ufl.sym(ufl.grad(diff_u)), ufl.sym(ufl.grad(diff_u)))
        + ufl.inner(ufl.div(diff_u), ufl.div(diff_u))
    ) * dvol

    err = np.sqrt(
        domain.comm.allreduce(fem.assemble_scalar(fem.form(err_ufl)), op=MPI.SUM)
    )

    # H(div) error stress
    diff = ufl.div(sigma_proj + delta_sigmaR - sigma_ext)
    err_sighdiv = np.sqrt(
        domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * dvol)),
            op=MPI.SUM,
        )
    )

    # Store results
    results[ref_level, 0] = 1 / nelmt
    results[ref_level, 1] = domain.topology.index_map(2).size_local
    results[ref_level, 2] = err
    results[ref_level, 4] = err_sighdiv
    results[ref_level, 6] = eta

    for i, val in enumerate(eta_i):
        results[ref_level, 7 + i] = val

    results[ref_level, -1] = eta / err


def adaptive_solver(order_prime: int, order_eqlb: int, nref: int, guarantied_ee: bool):
    """Solution procedure (uniform mesh refinement)

    Args:
        order_prime:   The order of the fe-space for the primal problem
        order_eqlb:    The order of the RT space for the equilibration
        nref:          The number of refinements
        guarantied_ee: True, if the error estimate is a guarantied upper bound
    """
    # Storage of results
    if guarantied_ee:
        outname_base = "ManSol_P{}_RT{}_gEE".format(order_prime, order_eqlb)
    else:
        outname_base = "ManSol_P{}_RT{}_hEI".format(order_prime, order_eqlb)

    results = np.zeros((nref, 13))

    for i in range(nref):
        # --- Create mesh
        # Set mesh resolution
        nelmt = 2**i

        # Create mesh
        domain, facet_tags, ds = create_unit_square_builtin(nelmt)

        # --- Solve problem
        # Solve primal problem
        degree_proj = 1 if (order_prime == 2) else None

        f, stress_ref, u_h, stress_h = solve(
            domain, facet_tags, 1.0, order_prime, degree_proj
        )

        # Solve equilibration
        if guarantied_ee:
            stress_p, stress_e, ckorn = equilibrate(
                domain, facet_tags, f, stress_h, order_eqlb, True, False
            )
        else:
            stress_p, stress_e, ckorn = equilibrate(
                domain, facet_tags, f, stress_h, order_eqlb, False, False
            )

        # --- Estimate error
        eta, eta_i = estimate(
            domain, 1.0, -ufl.div(stress_ref), stress_p, stress_e, ckorn, guarantied_ee
        )

        # --- Postprocessing
        post_process(1.0, nelmt, u_h, stress_p, stress_e, eta, eta_i, i, results)

    # Calculate convergence rates
    results[1:, 3] = np.log(results[1:, 2] / results[:-1, 2]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )
    results[1:, 5] = np.log(results[1:, 4] / results[:-1, 4]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )
    results[1:, -3] = np.log(results[1:, 6] / results[:-1, 6]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )
    results[1:, -2] = np.log(results[1:, 7] / results[:-1, 7]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )

    # Export results to csv
    header_protocol = (
        "hmin, nelmt, err, rateerr, errsigmahdiv, ratesigmahdiv, "
        "eetot, eedsigR, eeasym, eeosc, rateetot, rateedsigR, ieff"
    )
    np.savetxt(outname_base + ".csv", results, delimiter=",", header=header_protocol)


if __name__ == "__main__":

    for is_grntd in [True, False]:
        # Solve based on P2 with equilibration in RT2
        adaptive_solver(2, 2, 8, is_grntd)

        # # Solve based on P2 with equilibration in RT3
        # adaptive_solver(2, 3, 7, is_grntd)

        # # Solve based on P3 with equilibration in RT3
        # adaptive_solver(3, 3, 7, is_grntd)

        # # Solve based on P3 with equilibration in RT4
        # adaptive_solver(3, 4, 8, is_grntd)
