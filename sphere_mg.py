from firedrake import *  # noqa: F403
import asQ
from utils.mg import icosahedral_mesh
import numpy as np
np.random.seed(6)

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(
    description="ParaDiag solver for diffusion correlation operator on the unit sphere",
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral sphere. Total number of cells is 20*4^ref_level.')
parser.add_argument('--base_level', type=int, default=1, help='Refinement level of coarsest grid. Total number of cells is 20*4^ref_level.')
parser.add_argument('--D', type=float, default=0.2, help='Daley lengthscale.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the CG finite element space.')
parser.add_argument('--nslices', type=int, default=1, help='Number of time-slices in the all-at-once system. Must divide the number of MPI ranks exactly.')
parser.add_argument('--slice_length', type=int, default=8, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# Set up time parallelism
time_partition = [args.slice_length for _ in range(args.nslices)]
ensemble = asQ.create_ensemble(time_partition)
nt = sum(time_partition)

# Icosahedral sphere mesh:
# - has multigrid hierarchy, from coarsest mesh with base_refs
# - coarsest level has base_level refinements
# - halos (overlap) set up for Vanka or Star patches
mesh = icosahedral_mesh(
    R0=1, degree=1,
    base_level=args.base_level,
    nrefs=args.ref_level-args.base_level,
    distribution_parameters={
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)},
    comm=ensemble.comm
)

V = FunctionSpace(mesh, "CG", args.degree)

# Show number of degrees of freedom
PETSc.Sys.Print(f"{V.dim() = }")

# Random ICs to hit range of modes
ics = Function(V)
ics.dat.data[:] = np.random.random_sample(ics.dat.data.shape)

# diffusion coefficient from correlation lengthscale
D = args.D
nu = D*D/(2*nt - 4)
nu_c = Constant(nu)


# mass matrix
def form_mass(u, v):
    return inner(u, v)*dx


# stiffness matrix
def form_function(u, v, t):
    return inner(nu_c*grad(u), grad(v))*dx


# import multigrid transfers for non-nested grids
from utils.mg import ManifoldTransferManager  # noqa: F401

### Configure the solver
parameters = {
    "snes_type": "ksponly",               # No nonlinear solver
    "ksp": {                              # Print out some useful information
        "view": ":logs/ksp_view.log",     # Write linear solver details to file
        "converged_rate": None,           # Print out convergence rate and iteration count
        "monitor": None,                  # Print out residual at each iteration
    },
    "ksp_rtol": 1e-6,                     # Outer relative tolerance
    "ksp_type": "richardson",             # Fixed point iterations
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",  # ParaDiag preconditioner
    "circulant_state": "linear",          # No need to rebuild the blocks
    "circulant_alpha": 1e-5,              # Circulant parameter
    "circulant_block": {
        "ksp_max_it": 100,                # Maximum linear iterations
        "ksp_converged_maxits": None,     # Don't crash if we hit max iterations
        "ksp_rtol": 1e-6,                 # Inner relative tolerance
        "ksp_type": "preonly",            # Single V-cycle application
        "pc_type": "mg",                  # Multigrid preconditioner
        "pc_mg_type": "multiplicative",   # Apply levels multiplicatively
        "pc_mg_cycle_type": "v",          # V-cycles
        "mg_transfer_manager": f"{__name__}.ManifoldTransferManager",
        "mg_levels": {                    # Solver options for each level
            "ksp_max_it": 3,              # Number of smoothing steps
            "ksp_type": "chebyshev",      # Chebyshev iterations
            "ksp_chebyshev_esteig": "0,0.25,0,1.05",  # target top end of spectrum
            "esteig_ksp_max_it": 20,
            # Additive Schwarz method with "star" patches
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star": {
                "construct_dim": 0,
                "sub_sub_pc_type": "lu"
            },
            # Alternatively use point Jacobi again with "pc_type": "pbjacobi"
        },
        "mg_coarse": {  # Direct LU solve with MUMPS on coarsest grid
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_mat_type": "aij",
            "assembled_ksp_type": "preonly",
            "assembled_pc_type": "lu",
            "assembled_pc_factor_mat_solver_type": "mumps",
        },
    },
}
# Configure options specifically for block j with 'circulant_block_{j}'
for j in range(nt):
    parameters[f"circulant_block_{j}"] = {
        "ksp_converged_rate": f":logs/block_{j}_ksp_rate.log",  # Log convergence rates
        # "ksp_max_it": j  # Variable iterations per block?
    }

### Set up the all at once system
paradiag = asQ.Paradiag(
    ensemble=ensemble,
    time_partition=time_partition,
    form_mass=form_mass,
    form_function=form_function,
    ics=ics, dt=1, theta=1,
    solver_parameters=parameters)

paradiag.solve()

### Print out some iteration count diagnostics

# Number of outer iterations
aaos_its = paradiag.linear_iterations
# Extra preconditioner application to precondition rhs
pc_its = aaos_its + 1

# Number of inner iterations
block_its = paradiag.block_iterations.data()
max_its = max(block_its)
min_its = min(block_its)

PETSc.Sys.Print(f"Outer iterations: {aaos_its:>4d}")
PETSc.Sys.Print(f"Minimum iterations (per block solve): {min_its:>4d} ({min_its/pc_its:.4e})")
PETSc.Sys.Print(f"Maximum iterations (per block solve): {max_its:>4d} ({max_its/pc_its:.4e})")
