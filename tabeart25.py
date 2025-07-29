from firedrake import *  # noqa: F403
import asQ
import numpy as np
np.random.seed(6)

nt = 10
nx = 50
D = 0.2

# Set up time parallelism
# Partition of length 1 => time-serial
time_partition = [nt]
ensemble = asQ.create_ensemble(time_partition)

mesh = UnitSquareMesh(nx, nx, comm=ensemble.comm)

V = FunctionSpace(mesh, "CG", 1)

# Random ICs to hit range of modes
ics = Function(V)
ics.dat.data[:] = np.random.random_sample(ics.dat.data.shape)

# homogeneous boundary conditions
bcs = [DirichletBC(V, 0, "on_boundary")]
bcs[0].apply(ics)

# diffusion coefficient from correlation lengthscale
nu = D*D/(2*nt - 4)
nu_c = Constant(nu)


# mass matrix
def form_mass(u, v):
    return inner(u, v)*dx


# stiffness matrix
def form_function(u, v, t):
    return inner(nu_c*grad(u), grad(v))*dx


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
    "circulant_block": {                  # Configure default block solver options
        "ksp_max_it": 500,                        # Maximum linear iterations
        "ksp_converged_maxits": None,             # Don't crash if we hit max iterations
        "ksp_rtol": 1e-6,                         # Inner relative tolerance
        "ksp_type": "chebyshev",                  # Chebyshev iterations
        "ksp_chebyshev_esteig": "0.95,0,0,1.05",  # Arnoldi estimate of eigenvalues
        "esteig_ksp_max_it": 30,                  # Number of Arnoldi iterations
        "pc_type": "pbjacobi"                     # Point Jacobi preconditioner (use "none" for unpreconditioned)
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
    ics=ics, dt=1, theta=1, bcs=bcs,
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
