from enum import Enum
import gmsh
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

import dolfinx
from dolfinx.io import gmshio
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
    check_weak_symmetry_condition,
)


# --- Mesh generation
def create_cmembrane(h: float) -> dmesh.Mesh:
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

    initial_mesh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    # --- Make mesh compatible with equilibration
    # List of refined cells
    refined_cells = []

    # Required connectivity's
    initial_mesh.topology.create_connectivity(0, 2)
    initial_mesh.topology.create_connectivity(1, 2)
    pnt_to_cell = initial_mesh.topology.connectivity(0, 2)

    # The boundary facets
    bfcts = dmesh.exterior_facet_indices(initial_mesh.topology)

    # Get boundary nodes
    V = dfem.FunctionSpace(initial_mesh, ("Lagrange", 1))
    bpnts = dfem.locate_dofs_topological(V, 1, bfcts)

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

    return dmesh.create_mesh(
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
            facets = dmesh.locate_entities(self.mesh, 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))

        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.facet_functions = dmesh.meshtags(
            self.mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        self.ds = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=self.facet_functions
        )

    def refine(
        self,
        doerfler: float,
        eta_h: typing.Optional[dfem.Function] = None,
        outname: typing.Optional[str] = None,
    ):
        """Refine the mesh based on Doerflers marking strategy

        Args:
            doerfler: The Doerfler parameter
            eta_h:    The function of the cells error estimate
            outname:  The name of the output file for the mesh
                      (no output when not specified)
        """
        # The number of current mesh cells
        ncells = self.mesh.topology.index_map(2).size_global

        # Export marker to ParaView
        if outname is not None:
            V_out = dfem.FunctionSpace(self.mesh, ("DG", 0))
            eta_h_out = dfem.Function(V_out)
            eta_h_out.name = "eta_h"
            eta_h_out.x.array[:] = eta_h.array[:]

            outfile = dolfinx.io.XDMFFile(
                MPI.COMM_WORLD,
                outname + "-mesh" + str(self.refinement_level) + "_error.xdmf",
                "w",
            )
            outfile.write_mesh(self.mesh)
            outfile.write_function(eta_h_out, 0)
            outfile.close()

        # Refine the mesh
        if np.isclose(doerfler, 1.0):
            refined_mesh = dmesh.refine(self.mesh)
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
            edges = dmesh.compute_incident_entities(self.mesh, refine_cells, 2, 1)
            refined_mesh = dmesh.refine(self.mesh, edges)

        # Update the mesh
        self.mesh = refined_mesh
        self.mark_boundary()

        # Update counter
        self.refinement_level += 1

        # Print infos
        if eta_h is None:
            print(
                "Refinement {} - ncells: {}".format(
                    self.refinement_level, self.mesh.topology.index_map(2).size_global
                )
            )
        else:
            print(
                "Refinement {} - Total error: {} - ncells: {}".format(
                    self.refinement_level,
                    np.sqrt(eta_total),
                    self.mesh.topology.index_map(2).size_global,
                )
            )


# --- The primal problem
def solve(
    domain: AdaptiveCMembrane,
    pi_1: float,
    p_0: float,
    degree: int,
    use_up: typing.Optional[bool] = False,
) -> typing.Tuple[dfem.Function, int, typing.Any]:
    """Solves the Poisson problem based on lagrangian finite elements

    Args:
        domain:      The domain
        pi_1:        The ratio of first and second Lamé parameter
        p_0:         The the traction in y direction on surface 3
        order_prime: The order of the FE space
        use_up:      True, if a u-p formulation shall be applied

    Returns:
        The approximate solution
    """

    # Initialise timing
    timing = 0

    if use_up:
        raise NotImplementedError("UP not implemented")
    else:
        # Set function space (primal problem)
        V_prime = dfem.VectorFunctionSpace(domain.mesh, ("P", degree))
        uh = dfem.Function(V_prime)

        # Set trial and test functions
        u = ufl.TrialFunction(V_prime)
        v_u = ufl.TestFunction(V_prime)

        # Equation system
        sigma = 2 * ufl.sym(ufl.grad(u)) + pi_1 * ufl.div(u) * ufl.Identity(2)

        a = dfem.form(ufl.inner(sigma, ufl.sym(ufl.grad(v_u))) * ufl.dx)

        # The traction BC on surface 3
        l = dfem.form(ufl.inner(ufl.as_vector([0, p_0]), v_u) * domain.ds(3))

        # Dirichlet boundary conditions
        uD = dfem.Function(V_prime)

        fcts = domain.facet_functions.indices[domain.facet_functions.values == 1]
        dofs = dfem.locate_dofs_topological(V_prime, 1, fcts)
        bc_essnt = [dfem.dirichletbc(uD, dofs)]

        # Solve primal problem
        timing -= time.perf_counter()
        A = dfem.petsc.assemble_matrix(a, bcs=bc_essnt)
        A.assemble()

        L = dfem.petsc.create_vector(l)
        dfem.petsc.assemble_vector(L, l)
        dfem.apply_lifting(L, [a], [bc_essnt])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dfem.set_bc(L, bc_essnt)

        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

        solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

        solver.solve(L, uh.vector)
        timing += time.perf_counter()

        outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test_solution.xdmf", "w")
        outfile.write_mesh(domain.mesh)
        outfile.write_function(uh, 0)
        outfile.close()

        # The approximated stress
        sigma_h = 2 * ufl.sym(ufl.grad(uh)) + pi_1 * ufl.div(uh) * ufl.Identity(2)

        # The number of primal DOFs
        ndofs = 2 * uh.function_space.dofmap.index_map.size_global

    print(f"Primal problem solved in {timing:.4e} s")

    return uh, ndofs, sigma_h


# --- The flux equilibration
def equilibrate_flux(
    domain: AdaptiveCMembrane,
    p_0: float,
    sigma_h: typing.Any,
    degree: int,
    check_equilibration: typing.Optional[bool] = True,
) -> typing.Tuple[typing.Any, dfem.Function]:
    """Equilibrate the stress

    Args:
        domain:              The domain
        p_0:                 The the traction in y direction on surface 3
        sigma_h:             The stress calculated from the primal solution uh
        degree:              The order of RT elements used for equilibration
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The difference between equilibrated and projected stress (as ufl tensor)
    """

    # Check input
    if order_eqlb < 2:
        raise ValueError("Stress equilibration only possible for k>1")

    # Project stress and RHS into required DG space
    V_rhs_proj = dfem.FunctionSpace(domain.mesh, ("DG", degree - 1))
    V_flux_proj = dfem.VectorFunctionSpace(domain.mesh, ("DG", degree - 1))

    sigma_proj = local_projection(
        V_flux_proj,
        [
            ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]),
            ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]),
        ],
    )

    rhs_proj = [dfem.Function(V_rhs_proj), dfem.Function(V_rhs_proj)]

    # Initialise equilibrator
    equilibrator = FluxEqlbSE(degree, domain.mesh, rhs_proj, sigma_proj, True, True)

    # Set BCs
    fluxbcs = [[], []]

    fctfkt = domain.facet_functions
    V_flux = equilibrator.V_flux
    czero = dfem.Constant(domain.mesh, PETSc.ScalarType(0.0))
    cp0 = dfem.Constant(domain.mesh, PETSc.ScalarType(-p_0))

    ## (sigma x n)[0] = 0 on surfaces 2-4
    fluxbcs[0].append(
        fluxbc(czero, fctfkt.indices[np.isin(fctfkt.values, [2, 3, 4])], V_flux, False)
    )

    ## (sigma x n)[1] = 0 on surface 2 and 4
    fluxbcs[1].append(
        fluxbc(czero, fctfkt.indices[np.isin(fctfkt.values, [2, 4])], V_flux, False)
    )

    ## (sigma x n)[1] = p_0 on surface 3
    fluxbcs[1].append(fluxbc(cp0, fctfkt.indices[fctfkt.values == 3], V_flux, False))

    ## Set BCs
    equilibrator.set_boundary_conditions(
        [fctfkt.indices[fctfkt.values == 1], fctfkt.indices[fctfkt.values == 1]],
        fluxbcs,
    )

    # Solve equilibration
    timing = 0

    timing -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing += time.perf_counter()

    print(f"Equilibration solved in {timing:.4e} s")

    # --- Check equilibration conditions ---
    if check_equilibration:
        # Stresses as ufl tensor
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

        # Check divergence condition
        rhs_proj_vecval = dfem.Function(V_flux_proj)

        div_condition_fulfilled = check_divergence_condition(
            stress_eqlb,
            stress_proj,
            rhs_proj_vecval,
            mesh=domain.mesh,
            degree=order_eqlb,
            flux_is_dg=True,
        )

        if not div_condition_fulfilled:
            raise ValueError("Divergence conditions not fulfilled")

        # Check if stress is H(div)
        for i in range(domain.mesh.geometry.dim):
            jump_condition_fulfilled = check_jump_condition(
                equilibrator.list_flux[i], sigma_proj[i]
            )

            if not jump_condition_fulfilled:
                raise ValueError("Jump conditions not fulfilled")

        # Check weak symmetry condition
        wsym_condition = check_weak_symmetry_condition(equilibrator.list_flux)

        if not wsym_condition:
            raise ValueError("Weak symmetry conditions not fulfilled")

    return stress_eqlb, equilibrator.get_korn_constants()


# --- Estimate the error
def estimate_error(
    domain: AdaptiveCMembrane,
    pi_1: float,
    delta_sigmaR: typing.Any,
    korns_constants: dfem.Function,
    use_up: typing.Optional[bool] = False,
    eqlb_is_sym: typing.Optional[bool] = False,
) -> typing.Tuple[dfem.Function, typing.List[float]]:
    """Estimates the error for elasticity

    The estimate is calculated based on/following [1]. For the given problem the
    error due to data oscitation is zero.

    [1] Bertrand, F. et al., https://doi.org/10.1002/num.22741, 2021

    Args:
        domain:       The domain
        pi_1:         The ratio of lambda and mu
        delta_sigmaR: The difference of equilibrated and projected flux
        use_up:       True, if a u-p formulation shall be applied
        eqlb_is_sym:  True, if the equilibrated stress is assumed to be symmetric

    Returns:
        The cell-local error estimates
        The total error estimate
    """

    # Initialize storage of error
    V_e = dfem.FunctionSpace(
        domain.mesh, ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
    )
    v = ufl.TestFunction(V_e)

    if use_up:
        raise NotImplementedError("UP not implemented")
    else:
        # The error estimate
        a_delta_sigma = 0.5 * (
            delta_sigmaR
            - (pi_1 / (2 + 2 * pi_1)) * ufl.tr(delta_sigmaR) * ufl.Identity(2)
        )
        err_wsym = 0.5 * korns_constants * (delta_sigmaR[0, 1] - delta_sigmaR[1, 0])

        if eqlb_is_sym:
            eta = ufl.inner(delta_sigmaR, a_delta_sigma)
        else:
            eta = ufl.inner(delta_sigmaR, a_delta_sigma) + ufl.inner(err_wsym, err_wsym)

    # Assemble cell-local errors
    form_eta = dfem.form(eta * v * ufl.dx)
    L_eta = dfem.petsc.create_vector(form_eta)
    dfem.petsc.assemble_vector(L_eta, form_eta)

    if use_up:
        raise NotImplementedError("UP not implemented")
    else:
        # Assemble error contributions
        err_dsigmaR = domain.mesh.comm.allreduce(
            dfem.assemble_scalar(
                dfem.form(ufl.inner(a_delta_sigma, delta_sigmaR) * ufl.dx)
            ),
            op=MPI.SUM,
        )
        err_asym = domain.mesh.comm.allreduce(
            dfem.assemble_scalar(dfem.form(ufl.inner(err_wsym, err_wsym) * ufl.dx)),
            op=MPI.SUM,
        )
        return L_eta, [
            np.sqrt(np.sum(L_eta.array)),
            np.sqrt(err_dsigmaR),
            np.sqrt(err_asym),
        ]


# --- Post processing
def post_processing(
    domain: AdaptiveCMembrane,
    pi_1: float,
    u_ext: dfem.Function,
    list_uh: typing.List[dfem.Function],
    results: NDArray,
    use_up: typing.Optional[bool] = False,
):
    # Prepare error evaluation
    uext_W = []
    err_W = []

    if use_up:
        # The function-space W for the error
        degree = u_ext.function_space.sub(0).element.basix_element.degree + 2
        Vu_W = dfem.VectorFunctionSpace(domain.mesh, ("P", degree))
        Vp_W = dfem.FunctionSpace(domain.mesh, ("P", degree - 1))

        # Interpolate the exact solution into space W
        uext_W.append(dfem.Function(Vu_W))
        uext_W[0].interpolate(u_ext.sub(0).collapse())

        uext_W.append(dfem.Function(Vp_W))
        uext_W[1].interpolate(u_ext.sub(1).collapse())

        # The error in W
        err_W.append(dfem.Function(Vu_W))
        err_W.append(dfem.Function(Vp_W))
    else:
        # The function-space W for the error
        degree = u_ext.function_space.element.basix_element.degree + 2
        Vu_W = dfem.VectorFunctionSpace(domain.mesh, ("P", degree))

        # The exact solution in W
        uext_W.append(dfem.Function(Vu_W))
        uext_W[0].interpolate(u_ext)

        # The error in W
        err_W.append(dfem.Function(Vu_W))

    # Evaluate error for esch uh
    for n, uh in enumerate(list_uh):

        # Calculate the error of the primal variables
        for i, (err_i, uext_i) in enumerate(zip(err_W, uext_W)):
            # Reinitalisation
            err_i.x.array[:] = 0.0

            # Interpolate uh into higer order space W
            if use_up:
                err_i.interpolate(uh.sub(i).collapse())
            else:
                err_i.interpolate(uh)

            # Calculate error
            err_i.x.array[:] -= uext_i.x.array[:]

        # Calculate the error
        if use_up:
            raise NotImplementedError("UP not implemented")
        else:
            form_err = dfem.form(
                (
                    ufl.inner(ufl.sym(ufl.grad(err_W[0])), ufl.sym(ufl.grad(err_W[0])))
                    + pi_1 * ufl.div(err_W[0]) * ufl.div(err_W[0])
                )
                * ufl.dx
            )

        results[n, 2] = np.sqrt(
            domain.mesh.comm.allreduce(dfem.assemble_scalar(form_err), op=MPI.SUM)
        )


if __name__ == "__main__":
    # --- Parameters ---
    # The ration of the Lamé parameters
    pi_1 = 2.333

    # The traction
    p_0 = 0.03

    # The orders of the FE spaces
    order_prime = 2
    order_eqlb = 3

    # Adaptive algorithm
    nref = 15
    doerfler = 0.5

    # --- Execute adaptive calculation ---
    # The domain
    domain = AdaptiveCMembrane(10)

    # Storage of results
    list_uh = []
    results = np.zeros((nref, 9))

    for n in range(0, nref):
        # Solve
        u_h, ndofs, sigma_h = solve(domain, pi_1, p_0, order_prime)

        # Equilibrate the flux
        delta_sigmaR, korns_constants = equilibrate_flux(
            domain, p_0, -sigma_h, order_eqlb, True
        )

        # Mark
        eta_h, eta_h_tot = estimate_error(domain, pi_1, delta_sigmaR, korns_constants)

        # Store results
        list_uh.append(u_h)

        id = domain.refinement_level
        results[id, 0] = domain.mesh.topology.index_map(2).size_global
        results[id, 1] = ndofs
        for i, val in enumerate(eta_h_tot):
            results[id, 4 + i] = eta_h_tot[i]

        # Refine
        domain.refine(doerfler, eta_h, outname="CooksMembrane")

    # --- Post processing
    # Uniform mesh refinement
    domain.refine(1.0)

    # Calculate over-kill solution
    u_ext, ndofs, sigma_ext = solve(domain, pi_1, p_0, order_prime + 1)

    # Evaluate error
    post_processing(domain, pi_1, u_ext, list_uh, results)

    # Export results
    results[1:, 3] = np.log(results[1:, 2] / results[:-1, 2]) / np.log(
        results[:-1, 1] / results[1:, 1]
    )
    results[1:, 7] = np.log(results[1:, 4] / results[:-1, 4]) / np.log(
        results[:-1, 1] / results[1:, 1]
    )
    results[:, 8] = results[:, 4] / results[:, 2]

    outname = "CooksMembrane_porder-{}_eorder-{}.csv".format(order_prime, order_eqlb)
    header = "nelmt, ndofs, erruh1, rateuh1, eetot, eedsigR, eeasym, rateetot, ieff"

    np.savetxt(outname, results, delimiter=",", header=header)
