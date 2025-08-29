import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import FieldVariable, Material, Problem, Integral, Equation, Equations
from sfepy.base.base import IndexedStruct
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

print("🔍 Starting SfePy piezoelectric cantilever static simulation...")

# ------------------------------------------------------------------
# 1. Load mesh
# ------------------------------------------------------------------
mesh = Mesh.from_file('../CAD/exports/beam_piezo_v3.mesh')
domain = FEDomain('domain', mesh)
print(f"✅ Mesh loaded: {len(mesh.coors)} vertices")

# ------------------------------------------------------------------
# 2. Regions (all via 'cells of group')
# ------------------------------------------------------------------
Omega     = domain.create_region('Omega', 'all')
Substrate = domain.create_region('Substrate', 'cells of group 25')
Piezo     = domain.create_region('Piezo',     'cells of group 26')
TopElect  = domain.create_region('TopElectrode',    'cells of group 27')
BotElect  = domain.create_region('BottomElectrode', 'cells of group 28')
Clamp     = domain.create_region('Clamp', 'cells of group 29')

print("✅ Regions created (using cells of group)")

# ------------------------------------------------------------------
# 3. Fields & Variables
# ------------------------------------------------------------------
field_u   = Field.from_args('displacement', np.float64, 'vector', Omega, approx_order=1)
field_phi = Field.from_args('potential',    np.float64, 'scalar', Piezo, approx_order=1)

u    = FieldVariable('u', 'unknown', field_u)
phi  = FieldVariable('phi', 'unknown', field_phi)
v    = FieldVariable('v', 'test', field_u, primary_var_name='u')
psi  = FieldVariable('psi', 'test', field_phi, primary_var_name='phi')

print("✅ Fields & variables created")

# ------------------------------------------------------------------
# 4. Materials
# ------------------------------------------------------------------
# Substrate: steel-like
D_sub = stiffness_from_youngpoisson(3, 200e9, 0.3)
mat_sub = Material('Substrate', D=D_sub, rho=7800.0)

# Piezoelectric: simplified PZT-5H
D_pzt = stiffness_from_youngpoisson(3, 60e9, 0.31)

# Piezo coupling matrix [C/m^2]
e_matrix = np.array([
    [0, 0, 0, 0, 12.3, 0],
    [0, 0, 0, 12.3, 0, 0],
    [-6.5, -6.5, 23.3, 0, 0, 0]
], dtype=np.float64)

# Permittivity (absolute, diagonal)
eps_r = np.array([1500.0, 1500.0, 1700.0])
epsilon = np.diag(eps_r) * 8.8541878128e-12

mat_pzt = Material('Piezo', D=D_pzt, e=e_matrix, epsilon=epsilon, rho=7500.0)

print("✅ Materials defined")

# ------------------------------------------------------------------
# 5. Integrals & Terms
# ------------------------------------------------------------------
i = Integral('i', order=2)

# Mechanical terms
t_sub = Term.new('dw_lin_elastic(Substrate.D, v, u)', i, Substrate,
                 Substrate=mat_sub, v=v, u=u)
t_pzt = Term.new('dw_lin_elastic(Piezo.D, v, u)', i, Piezo,
                 Piezo=mat_pzt, v=v, u=u)

# Coupling terms
t_c_vphi = Term.new('dw_piezo_coupling(Piezo.e, v, phi)', i, Piezo,
                    Piezo=mat_pzt, v=v, phi=phi)
t_c_upsi = Term.new('dw_piezo_coupling(Piezo.e, u, psi)', i, Piezo,
                    Piezo=mat_pzt, u=u, psi=psi)

# Dielectric term
t_eps = Term.new('dw_diffusion(Piezo.epsilon, psi, phi)', i, Piezo,
                 Piezo=mat_pzt, psi=psi, phi=phi)

# Equations
eq_mech = Equation('mech', t_sub + t_pzt - t_c_vphi)
eq_elec = Equation('elec', t_c_upsi + t_eps)
eqs = Equations([eq_mech, eq_elec])
print("✅ Equations created")

# ------------------------------------------------------------------
# 6. Boundary Conditions
# ------------------------------------------------------------------
ebc_clamp = EssentialBC('fix', Clamp, {'u.all': 0.0})
ebc_ground = EssentialBC('ground', BotElect, {'phi.all': 0.0})
ebc_voltage = EssentialBC('voltage', TopElect, {'phi.all': 10.0})

bcs = Conditions([ebc_clamp, ebc_ground, ebc_voltage])
print("✅ Boundary conditions set")

# ------------------------------------------------------------------
# 7. Problem & Solvers
# ------------------------------------------------------------------
pb = Problem('piezo_static', equations=eqs)

ls = ScipyDirect({})
nls = Newton({'i_max': 20, 'eps_a': 1e-8, 'eps_r': 1e-6}, lin_solver=ls)
pb.set_solver(nls)
pb.set_linear(True)

pb.time_update(ebcs=bcs)
print("✅ Problem & solvers configured")

# ------------------------------------------------------------------
# 8. Solve
# ------------------------------------------------------------------
status = IndexedStruct()
state = pb.solve(status=status)

if state is not None:
    print("✅ Solution computed")
    u_data = state.get_state_parts()['u']
    phi_data = state.get_state_parts()['phi']
    print(f"   - Max displacement |u|: {np.abs(u_data).max():.3e} m")
    print(f"   - Max potential |phi|: {np.abs(phi_data).max():.3e} V")
else:
    print("⚠️ Solver returned None")

# ------------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------------
pb.save_state('piezo_static.vtk', state=state)
print("💾 Results saved to 'piezo_static.vtk' (open in ParaView right now)")
