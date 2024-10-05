# Copyright (C) 2024 Maximilian Brodbeck
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Adaptive solution of the Cooks membrane

Solution of the quasi-static linear elasticity equation

     div(sigma) = 0  with sigma = 2 * eps + pi_1 * div(u) * I,

for the Cooks membrane (parameters from [1]) on a series of adaptively re-
fined meshes. The error estimate is evaluated using the equilibrated stress
with weak symmetry while the spatial refinement is based on a Dörfler marking 
strategy. Convergence is reported with respect to an numerical overkill solu-
tion on a refined mesh and an increased polynomial degree of the used finite 
element space.

[1] Schröder J. et al., https://doi.org/10.1007/s11831-020-09477-3, 2021
"""

from contextlib import ExitStack
from enum import Enum
import gmsh
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

from dolfinx import fem, io, la, mesh
import ufl

from dolfinx_eqlb.cpp import local_solver_cholesky
from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection


# --- Mesh generation
def create_cmembrane(h: float) -> mesh.Mesh:
    """Create Cooks membrane following [1]

    [1] Schröder J. et al., https://doi.org/10.1007/s11831-020-09477-3, 2021

    Args:
        h: The mesh size

    Returns:
        The mesh
    """

    # --- Create basic mesh
    # Initialise gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)

    # Name of the geometry
    gmsh.model.add("CooksMembrane")

    # Points
    list_pnts = [
        [0, 0],
        [48, 44],
        [48, 60],
        [0, 44],
    ]

    pnts = [gmsh.model.occ.add_point(pnt[0], pnt[1], 0.0) for pnt in list_pnts]

    # Bounding curves and 2D surface
    bfcts = [
        gmsh.model.occ.add_line(pnts[0], pnts[1]),
        gmsh.model.occ.add_line(pnts[1], pnts[2]),
        gmsh.model.occ.add_line(pnts[2], pnts[3]),
        gmsh.model.occ.add_line(pnts[3], pnts[0]),
    ]

    boundary = gmsh.model.occ.add_curve_loop(bfcts)
    surface = gmsh.model.occ.add_plane_surface([boundary])
    gmsh.model.occ.synchronize()

    # Set tag on boundaries and surface
    for i, bfct in enumerate(bfcts):
        gmsh.model.addPhysicalGroup(1, [bfct], i + 1)

    gmsh.model.addPhysicalGroup(2, [surface], 1)

    # Generate mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)

    initial_mesh, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    # --- Make mesh compatible with equilibration
    # List of refined cells
    refined_cells = []

    # Required connectivity's
    initial_mesh.topology.create_connectivity(0, 2)
    initial_mesh.topology.create_connectivity(1, 2)
    pnt_to_cell = initial_mesh.topology.connectivity(0, 2)

    # The boundary facets
    bfcts = mesh.exterior_facet_indices(initial_mesh.topology)

    # Get boundary nodes
    V = fem.FunctionSpace(initial_mesh, ("Lagrange", 1))
    bpnts = fem.locate_dofs_topological(V, 1, bfcts)

    # Check if point is linked with only on cell
    for pnt in bpnts:
        cells = pnt_to_cell.links(pnt)

        if len(cells) == 1:
            refined_cells.append(cells[0])

    # Refine mesh
    list_ref_cells = list(set(refined_cells))

    # Add central node into refined cells
    x_new = np.copy(initial_mesh.geometry.x[:, 0:2])
    cells_new = np.copy(initial_mesh.geometry.dofmap.array).reshape(-1, 3)
    cells_add = np.zeros((2, 3), dtype=np.int32)

    for i, c_init in enumerate(list_ref_cells):
        # The cell
        c = c_init + 2 * i

        # Nodes on cell
        cnodes = cells_new[c, :]
        x_cnodes = x_new[cnodes]

        # Coordinate of central node
        node_central = (1 / 3) * np.sum(x_cnodes, axis=0)

        # New node coordinates
        id_new = max(cnodes) + 1
        x_new = np.insert(x_new, id_new, node_central, axis=0)

        # Adjust definition of existing cells
        cells_new[cells_new >= id_new] += 1

        # Add new cells
        cells_add[0, :] = [cells_new[c, 1], cells_new[c, 2], id_new]
        cells_add[1, :] = [cells_new[c, 2], cells_new[c, 0], id_new]
        cells_new = np.insert(cells_new, c + 1, cells_add, axis=0)

        # Correct definition of cell c
        cells_new[c, 2] = id_new

    return mesh.create_mesh(
        MPI.COMM_WORLD,
        cells_new,
        x_new,
        ufl.Mesh(
            ufl.VectorElement(
                "Lagrange", ufl.Cell("triangle", geometric_dimension=2), 1
            )
        ),
    )


class AdaptiveCMembrane:
    """An adaptive Cooks membrane
    Create an initial mesh based on [1] and refines the mesh using a the Doerfler strategy.

    [1] Schröder J. et al., https://doi.org/10.1007/s11831-020-09477-3, 2021
    """

    def __init__(self, h: int):
        """Constructor

        Args:
            h: The initial mesh size
        """

        # --- Initialise storage
        # The mesh counter
        self.refinement_level = 0

        # --- Create the initial mesh
        self.mesh = create_cmembrane(h)

        # The boundary markers
        self.boundary_markers = [
            (1, lambda x: np.isclose(x[0], 0)),
            (2, lambda x: np.isclose(x[1], 44 + x[0] / 3, rtol=1e-10, atol=1e-10)),
            (3, lambda x: np.isclose(x[0], 48)),
            (4, lambda x: np.isclose(x[1], 11 * x[0] / 12, rtol=1e-10, atol=1e-10)),
        ]

        # Set facet function and facet integrator
        self.facet_functions = None
        self.ds = None

        self.mark_boundary()

    # --- Generate the mesh ---
    def mark_boundary(self):
        """Marks the boundary based on the initially defined boundary markers"""

        facet_indices, facet_markers = [], []

        for marker, locator in self.boundary_markers:
            facets = mesh.locate_entities(self.mesh, 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))

        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.facet_functions = mesh.meshtags(
            self.mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        self.ds = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=self.facet_functions
        )

    def refine(self, doerfler: float, eta_h: typing.Optional[fem.Function] = None):
        """Refine the mesh based on Doerflers marking strategy

        Args:
            doerfler: The Doerfler parameter
            eta_h:    The function of the cells error estimate
            outname:  The name of the output file for the mesh
                      (no output when not specified)
        """
        # The number of current mesh cells
        ncells = self.mesh.topology.index_map(2).size_global

        # Refine the mesh
        if np.isclose(doerfler, 1.0):
            refined_mesh = mesh.refine(self.mesh)

            if eta_h is not None:
                eta_total = np.sum(eta_h.array)
        else:
            # Check input
            if eta_h is None:
                raise ValueError("Error marker required for adaptive refinement")

            # The total error (squared!)
            eta_total = np.sum(eta_h.array)

            # Cut-off
            cutoff = doerfler * eta_total

            # Sort cell contributions
            sorted_cells = np.argsort(eta_h.array)[::-1]

            # Create list of refined cells
            rolling_sum = 0.0
            breakpoint = ncells

            for i, e in enumerate(eta_h.array[sorted_cells]):
                rolling_sum += e
                if rolling_sum > cutoff:
                    breakpoint = i
                    break

            # List of refined cells
            refine_cells = np.array(
                np.sort(sorted_cells[0 : breakpoint + 1]), dtype=np.int32
            )

            # Refine mesh
            edges = mesh.compute_incident_entities(self.mesh, refine_cells, 2, 1)
            refined_mesh = mesh.refine(self.mesh, edges)

        # Update the mesh
        self.mesh = refined_mesh
        self.mark_boundary()

        # Update counter
        self.refinement_level += 1


# --- The primal problem
class SolverType(Enum):
    mumps = 1
    cg_amg = 2


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(3)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x and y dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(2)]

        # Build the three translational rigid body modes
        for i in range(2):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1 = x[dofs_block, 0], x[dofs_block, 1]
        basis[2][dofs[0]] = -x1
        basis[2][dofs[1]] = x0

    # Orthonormalise the six vectors
    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)


def solve(
    domain: AdaptiveCMembrane,
    pi_1: float,
    p_0: float,
    degree: int,
    solver_type: SolverType,
) -> typing.Tuple[typing.List[fem.Function], int, typing.Any]:
    """Solver for linear elasticity based on Lagrangian finite elements

    Args:
        domain:      The domain
        pi_1:        The ratio of first and second Lamé parameter
        p_0:         The the traction in y direction on surface 3
        degree:      The degree of the FE space
        solver_type: The solver for the linear equation system

    Returns:
        The approximate solution
        The number of DOFs
        The approximated stress
        The timings
    """

    # Check input
    if degree < 2:
        raise ValueError("Lagrangian element for displacement requires k>=2")

    # Set function space (primal problem)
    V = fem.VectorFunctionSpace(domain.mesh, ("P", degree))
    uh = fem.Function(V)

    # Set trial and test functions
    u = ufl.TrialFunction(V)
    v_u = ufl.TestFunction(V)

    # The bilinear form
    sigma = 2 * ufl.sym(ufl.grad(u)) + pi_1 * ufl.div(u) * ufl.Identity(2)
    a = fem.form(ufl.inner(sigma, ufl.sym(ufl.grad(v_u))) * ufl.dx)

    # The linear form (traction on surface 3)
    l = fem.form(ufl.inner(ufl.as_vector([0, p_0]), v_u) * domain.ds(3))

    # Dirichlet BCs
    uD = fem.Function(V)

    fcts = domain.facet_functions.indices[domain.facet_functions.values == 1]
    dofs = fem.locate_dofs_topological(V, 1, fcts)
    bc_essnt = [fem.dirichletbc(uD, dofs)]

    # Solve
    timing_assembly = -time.perf_counter()
    A = fem.petsc.assemble_matrix(a, bcs=bc_essnt)
    A.assemble()

    L = fem.petsc.create_vector(l)
    fem.petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [bc_essnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bc_essnt)
    timing_assembly += time.perf_counter()

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    if solver_type == SolverType.mumps:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
    elif solver_type == SolverType.cg_amg:
        solver.setType(PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    timing_solver = -time.perf_counter()
    solver.solve(L, uh.vector)
    timing_solver += time.perf_counter()

    # The approximated stress
    sigma_h = 2 * ufl.sym(ufl.grad(uh)) + pi_1 * ufl.div(uh) * ufl.Identity(2)

    # The number of primal DOFs
    ndofs = 2 * uh.function_space.dofmap.index_map.size_global

    return uh, ndofs, sigma_h, [timing_assembly, timing_solver]


# --- The flux equilibration
def projection(order_eqlb: int, domain: mesh.Mesh, sigma_h: typing.Any) -> float:
    """Projects flux and RHS

    The RHS is assumed to be the divergence of the exact
    flux (manufactured solution).

    Args:
        order_eqlb: The order of the RT space
        domain:     The mesh
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

    # Set up the projector
    list_u = [fem.Function(V_proj) for _ in range(2)]
    list_ucpp = [u._cpp_object for u in list_u]

    # Solve the projection
    timing_proj = -time.perf_counter()
    local_solver_cholesky(list_ucpp, a, [l1, l2])
    timing_proj += time.perf_counter()

    return timing_proj


def equilibrate(
    domain: AdaptiveCMembrane,
    p_0: float,
    sigma_h: typing.Any,
    degree: int,
    weak_symmetry: typing.Optional[bool],
) -> typing.Tuple[typing.Any, fem.Function, typing.List[float]]:
    """Equilibrate the stress

    Args:
        domain:              The domain
        p_0:                 The the traction in y direction on surface 3
        sigma_h:             The stress calculated from the primal solution uh
        degree:              The order of RT elements used for equilibration
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The difference between equilibrated and projected stress (as ufl tensor)
        The cells Korn constant
        The timings
    """

    # Check input
    if degree < 2:
        raise ValueError("Stress equilibration only possible for k>1")

    # Project stress and RHS into required DG space
    V_rhs_proj = fem.FunctionSpace(domain.mesh, ("DG", degree - 1))
    V_flux_proj = fem.VectorFunctionSpace(domain.mesh, ("DG", degree - 1))

    sigma_proj = local_projection(
        V_flux_proj,
        [
            ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]),
            ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]),
        ],
    )

    rhs_proj = [fem.Function(V_rhs_proj), fem.Function(V_rhs_proj)]

    # Time (ideal) projection
    timing_proj = projection(degree, domain.mesh, sigma_h)

    # Initialise equilibrator
    equilibrator = FluxEqlbSE(
        degree, domain.mesh, rhs_proj, sigma_proj, weak_symmetry, True
    )

    # Set BCs
    fluxbcs = [[], []]

    fctfkt = domain.facet_functions
    V_flux = equilibrator.V_flux
    czero = fem.Constant(domain.mesh, PETSc.ScalarType(0.0))
    cp0 = fem.Constant(domain.mesh, PETSc.ScalarType(-p_0))

    ## (sigma x n)[0] = 0 on surfaces 2-4
    fluxbcs[0].append(
        fluxbc(czero, fctfkt.indices[np.isin(fctfkt.values, [2, 3, 4])], V_flux)
    )

    ## (sigma x n)[1] = 0 on surface 2 and 4
    fluxbcs[1].append(
        fluxbc(czero, fctfkt.indices[np.isin(fctfkt.values, [2, 4])], V_flux)
    )

    ## (sigma x n)[1] = p_0 on surface 3
    fluxbcs[1].append(fluxbc(cp0, fctfkt.indices[fctfkt.values == 3], V_flux))

    ## Set BCs
    equilibrator.set_boundary_conditions(
        [fctfkt.indices[fctfkt.values == 1], fctfkt.indices[fctfkt.values == 1]],
        fluxbcs,
    )

    # Solve equilibration
    timing_eqlb = -time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing_eqlb += time.perf_counter()

    # Stresses as ufl tensor
    stress_eqlb = ufl.as_matrix(
        [
            [-equilibrator.list_flux[0][0], -equilibrator.list_flux[0][1]],
            [-equilibrator.list_flux[1][0], -equilibrator.list_flux[1][1]],
        ]
    )

    return stress_eqlb, equilibrator.get_korn_constants(), [timing_proj, timing_eqlb]


# --- Estimate the error
def estimate(
    domain: AdaptiveCMembrane,
    pi_1: float,
    delta_sigmaR: typing.Any,
    korns_constants: fem.Function,
    guarantied_upper_bound: typing.Optional[bool] = True,
) -> typing.Tuple[fem.Function, typing.List[float]]:
    """Estimates the error for elasticity

    The estimate is calculated following [1]. For the given problem the
    error due to data oscitation is zero.

    [1] Bertrand, F. et al., https://doi.org/10.1002/num.22741, 2021

    Args:
        domain:                 The domain
        pi_1:                   The ratio of lambda and mu
        sdisc_type:             The type of the spatial discretisation
        delta_sigmaR:           The difference of equilibrated and projected flux
        korns_constants:        The Korn's constants
        guarantied_upper_bound: True, if a guarantied upper bound shall be calculated

    Returns:
        The cell-local error estimates
        The total error estimate
    """

    # Initialize storage of error
    V_e = fem.FunctionSpace(
        domain.mesh, ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
    )
    v = ufl.TestFunction(V_e)

    # Error from stress difference and assymetric part
    a_delta_sigma = 0.5 * (
        delta_sigmaR - (pi_1 / (2 + 2 * pi_1)) * ufl.tr(delta_sigmaR) * ufl.Identity(2)
    )
    err_wsym = 0.5 * korns_constants * (delta_sigmaR[0, 1] - delta_sigmaR[1, 0])

    # The error estimate
    if guarantied_upper_bound:
        eta = ufl.inner(delta_sigmaR, a_delta_sigma) + ufl.inner(err_wsym, err_wsym)
    else:
        eta = ufl.inner(delta_sigmaR, a_delta_sigma)

    # Assemble cell-local errors
    form_eta = fem.form(eta * v * ufl.dx)
    L_eta = fem.petsc.create_vector(form_eta)
    fem.petsc.assemble_vector(L_eta, form_eta)

    # The overall error (contributions)
    err_dsigmaR = domain.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(a_delta_sigma, delta_sigmaR) * ufl.dx)),
        op=MPI.SUM,
    )
    err_asym = domain.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(err_wsym, err_wsym) * ufl.dx)),
        op=MPI.SUM,
    )

    etai_tot = [np.sqrt(np.sum(L_eta.array)), np.sqrt(err_dsigmaR), np.sqrt(err_asym)]

    return L_eta, etai_tot


# --- The adaptive solution procedure
def adaptive_solver(
    order_prime: int,
    order_eqlb: int,
    nref: int,
    doerfler: int,
    guarantied_ee: bool,
    solver_type: SolverType,
) -> NDArray:
    """Adaptive solution procedure

    Args:
        order_prime:   The order of the fe-space for the primal problem
        order_eqlb:    The order of the RT space for the equilibration
        nref:          The number of refinements
        doerfler:      The Doerfler parameter
        guarantied_ee: True, if EE is a guarantied upper bound
        solver_type:   The solver for the linear equation system

    Returns:
        The timings
    """

    # The parameters
    pi_1 = 2.333
    p_0 = 0.03

    # The domain
    domain = AdaptiveCMembrane(10)

    # Storage of results
    timings = np.zeros((nref, 6))

    for n in range(0, nref):
        # Solve
        _, ndofs, sigma_h, timings_primal = solve(
            domain, pi_1, p_0, order_prime, solver_type
        )

        # Equilibrate the flux
        if guarantied_ee:
            delta_sigmaR, korns_constants, timings_eqlb = equilibrate(
                domain, p_0, -sigma_h, order_eqlb, True
            )
        else:
            delta_sigmaR, korns_constants, timings_eqlb = equilibrate(
                domain, p_0, -sigma_h, order_eqlb, False
            )

        # Mark
        eta_h, _ = estimate(domain, pi_1, delta_sigmaR, korns_constants, guarantied_ee)

        # Store results
        timings[n, 0] = domain.mesh.topology.index_map(2).size_global
        timings[n, 1] = ndofs
        timings[n, 2] = timings_primal[0]
        timings[n, 3] = timings_primal[1]
        timings[n, 4] = timings_eqlb[0]
        timings[n, 5] = timings_eqlb[1]

        # Refine
        domain.refine(doerfler, eta_h)

    return timings


# --- Time teh adaptive solver
def time_problem(
    order_prime: int,
    order_eqlb: int,
    nref: int,
    doerfler: int,
    guarantied_ee: bool,
    solver_type: SolverType,
    nretry: int,
):
    """Time an adaptive solution procedure

    Args:
        order_prime:   The order of the fe-space for the primal problem
        order_eqlb:    The order of the RT space for the equilibration
        nref:          The number of refinements
        doerfler:      The Doerfler parameter
        guarantied_ee: True, if EE is a guarantied upper bound
        nretry:        The number of retries
    """

    # Storage results
    if guarantied_ee:
        outname_base = "Timing_AdaptCook-P{}_RT{}_gEE".format(order_prime, order_eqlb)
    else:
        outname_base = "Timing_AdaptCook-P{}_RT{}_hEI".format(order_prime, order_eqlb)
    results = np.zeros((nref + 1, 10))

    for n in range(nretry):
        print("Retry {}/{}".format(n + 1, nretry))

        timings_n = adaptive_solver(
            order_prime, order_eqlb, nref, doerfler, guarantied_ee, solver_type
        )

        if n == 0:
            results[0:nref, :6] = timings_n[:, :]
        else:
            results[0:nref, 2:6] = np.minimum(results[0:nref, 2:6], timings_n[:, 2:])

    # Post-process timings
    for i in range(4):
        results[nref, 2 + i] = np.sum(results[0:nref, 2 + i])

    results[:, 6] = results[:, 2] + results[:, 3]
    results[:, 7] = results[:, 4] + results[:, 5]
    results[:, 8] = results[:, 5] / results[:, 6]
    results[:, 9] = results[:, 7] / results[:, 6]

    # Export results to csv
    header_protocol = "nelmt, ndofs, tpassmbl, tpsolve, teproj, tesolve, tprime, teqlb, tesolvebytprime, teqlbbytprime"
    np.savetxt(outname_base + ".csv", results, delimiter=",", header=header_protocol)


if __name__ == "__main__":
    # --- Parameters ---
    doerfler = 0.6
    nretry = 10
    # ------------------

    # Solve based on P2 with equilibration in RT2
    time_problem(2, 2, 14, doerfler, True, SolverType.mumps, nretry)
    time_problem(2, 2, 13, doerfler, False, SolverType.mumps, nretry)

    # Solve based on P2 with equilibration in RT3
    time_problem(2, 3, 15, doerfler, True, SolverType.mumps, nretry)
    time_problem(2, 3, 13, doerfler, False, SolverType.mumps, nretry)

    # Solve based on P3 with equilibration in RT3
    time_problem(3, 3, 12, doerfler, True, SolverType.mumps, nretry)
    time_problem(3, 3, 12, doerfler, False, SolverType.mumps, nretry)

    # Solve based on P3 with equilibration in RT4
    time_problem(3, 4, 14, doerfler, True, SolverType.mumps, nretry)
    time_problem(3, 4, 12, doerfler, False, SolverType.mumps, nretry)
