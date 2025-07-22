from firedrake import *  # noqa: F403
import asQ
from utils.mg import icosahedral_mesh
from utils.mg import ManifoldTransferManager  # noqa: F401
import numpy as np
np.random.seed(6)

nt = 8
coarse_level = 1
fine_level = 5

D = 0.2
nu = D*D/(2*nt - 4)
nu_c = Constant(nu)

assert nt % COMM_WORLD.size == 0
time_partition = [nt//COMM_WORLD.size for _ in range(COMM_WORLD.size)]
ensemble = asQ.create_ensemble(time_partition)

mesh = icosahedral_mesh(
    R0=1, degree=1,
    base_level=coarse_level,
    nrefs=fine_level-coarse_level,
    distribution_parameters={
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)},
    comm=ensemble.comm
)

V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print(f"{V.dim() = }")
ics = Function(V)
ics.dat.data[:] = np.random.random_sample(ics.dat.data.shape)


def form_mass(u, v):
    return inner(u, v)*dx


def form_function(u, v, t):
    return inner(nu_c*grad(u), grad(v))*dx


parameters = {
    "snes_type": "ksponly",
    "ksp": {
        "view": ":logs/ksp_view.log",
        "converged_rate": None,
        "monitor": None,
        "rtol": 1e-10,
    },
    "ksp_type": "richardson",
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "circulant_state": "linear",
    "circulant_alpha": 1e-8,
    "circulant_block": {
        "ksp_rtol": 1e-6,
        "ksp_type": "gmres",
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycle_type": "v",
        "mg_transfer_manager": f"{__name__}.ManifoldTransferManager",
        "mg_levels": {
            "ksp_max_it": 3,
            "ksp_type": "chebyshev",
            "ksp_chebyshev_esteig": "0,0.25,0,1.05",
            "esteig_ksp_max_it": 20,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka": {
                "construct_dim": 0,
                "sub_sub_pc_type": "lu"
            },
        },
        "mg_coarse": {
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_mat_type": "aij",
            "assembled_ksp_type": "preonly",
            "assembled_pc_type": "lu",
            "assembled_pc_factor_mat_solver_type": "mumps",
        },
    },
}
for j in range(nt):
    prefix = f"circulant_block_{j}"
    log_prefix = f":logs/block_{j}"
    parameters[f"{prefix}_ksp_converged_rate"] = f"{log_prefix}_ksp_rate.log"
    parameters[f"{prefix}_ksp_view"] = f"{log_prefix}_ksp_view.log"

paradiag = asQ.Paradiag(
    ensemble=ensemble,
    time_partition=time_partition,
    form_mass=form_mass,
    form_function=form_function,
    ics=ics, dt=1, theta=1,
    solver_parameters=parameters)

paradiag.solve()

aaos_its = paradiag.linear_iterations + 1  # extra application to precondition rhs
block_its = paradiag.block_iterations.data()
max_its = max(block_its)
min_its = min(block_its)

PETSc.Sys.Print(f"Block iterations: {block_its}")

PETSc.Sys.Print(f"Minimum iterations (per block solve): {min_its:>4d} ({min_its/aaos_its:.4e})")
PETSc.Sys.Print(f"Maximum iterations (per block solve): {max_its:>4d} ({max_its/aaos_its:.4e})")
