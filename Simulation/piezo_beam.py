import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import FieldVariable, Material, Problem, Integral, Equations, Equation, Function
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

print("⚙️  Starting coupled piezoelectric STATIC solve (mechanics + electrostatics)")

# ------------------------------------------------------------------
# 1) Mesh & Regions
# ------------------------------------------------------------------
mesh = Mesh.from_file('../CAD/exports/beam_piezo_v3.mesh')  # adjust if needed
domain = FEDomain('domain', mesh)

Omega = domain.create_region('Omega', 'all')

# Volume groups from your Gmsh physical volumes (edit IDs if different)
Substrate = domain.create_region('Substrate', 'cells of group 25')
Piezo     = domain.create_region('Piezo',     'cells of group 26')

# Clamp at x = xmin
(xmin, ymin, zmin), (xmax, ymax, zmax) = domain.get_mesh_bounding_box()
L = xmax - xmin
tol = 1e-6 * L
Clamp = domain.create_region('Clamp', f'vertices in (x < {xmin + tol})', kind='facet')

# Free end (for traction) at x ~ xmax
FreeEnd = domain.create_region('FreeEnd', f'vertices in (x > {xmax - tol})', kind='facet')

# Electrode surfaces from your Gmsh physical surfaces (edit IDs if different)
TopElectrode    = domain.create_region('TopElectrode',    'facets of group 27', kind='facet')
BottomElectrode = domain.create_region('BottomElectrode', 'facets of group 28', kind='facet')

# ------------------------------------------------------------------
# 2) Fields & Variables (vector displacement u, scalar potential p)
# ------------------------------------------------------------------
order_u = 2
order_p = 2

field_u = Field.from_args('displacement', np.float64, 'vector', Omega, approx_order=order_u)
field_p = Field.from_args('potential',    np.float64, 'scalar', Omega, approx_order=order_p)

u = FieldVariable('u', 'unknown', field_u)
v = FieldVariable('v', 'test',    field_u, primary_var_name='u')

p = FieldVariable('p', 'unknown', field_p)
q = FieldVariable('q', 'test',    field_p, primary_var_name='p')

# ------------------------------------------------------------------
# 3) Materials
# ------------------------------------------------------------------
# Substrate: set your (E, nu). (Example: steel.)
E_sub = 200e9
nu_sub = 0.30
C_sub = stiffness_from_youngpoisson(3, E_sub, nu_sub)
mat_sub = Material('sub', D=C_sub)

# Piezo layer properties:
# We follow the official SfePy piezo example formulation (PZT-5H, 3D), converting to
# stress–electric-displacement form and consistent Voigt ordering. :contentReference[oaicite:1]{index=1}
epsT = np.array([[1700., 0, 0],
                 [0, 1700., 0],
                 [0, 0, 1450.0]])  # relative permittivity (at constant stress)
dv = 1e-12 * np.array([[   0,    0,    0,    0, 741.,    0],
                       [   0,    0,    0,  741,    0,    0],
                       [-274, -274,  593,    0,    0,    0]])  # m/V (strain-charge form)

# Elasticity (Voigt in strain-charge form -> stress-charge form)
CEv = np.array([[1.27e+11, 8.02e+10, 8.47e+10, 0, 0, 0],
                [8.02e+10, 1.27e+11, 8.47e+10, 0, 0, 0],
                [8.47e+10, 8.47e+10, 1.17e+11, 0, 0, 0],
                [0, 0, 0, 2.34e+10, 0, 0],
                [0, 0, 0, 0, 2.30e+10, 0],
                [0, 0, 0, 0, 0, 2.35e+10]])

# Convert to SfePy Voigt order: SfePy uses [11,22,33,12,13,23].
voigt_map = [0, 1, 2, 5, 4, 3]
ix, iy = np.meshgrid(voigt_map, voigt_map, sparse=True)

CE = CEv[ix, iy]               # 6x6 stiffness (C^E)
ev = dv @ CEv                  # 3x6 piezo matrix in stress-charge form
e  = ev[:, voigt_map]          # reorder columns to SfePy's Voigt

eps0   = 8.8541878128e-12      # vacuum permittivity (F/m)
epsS   = epsT - dv @ ev.T      # relative permittivity at constant strain
kappa  = epsS * eps0           # absolute permittivity tensor (3x3)

mat_pz = Material('pz', C=CE, e=e, kappa=kappa)

# Tip traction (uniform, along -z). You can tune tip_pressure [Pa].
tip_pressure = 2e4  # N/m^2
def tip_load(ts, coors, mode=None, **kwargs):
    if mode == 'qp':
        # traction vector [tx, ty, tz]
        val = np.tile([0.0, 0.0, -tip_pressure], (coors.shape[0], 1, 1))
        return {'val': val}

load_fun = Function('tip_load', tip_load)
load = Material('load', function=load_fun)

# ------------------------------------------------------------------
# 4) Integrals
# ------------------------------------------------------------------
# Use a reasonable integration order for quadratic elements.
integral_v = Integral('iv', order=2 * max(order_u, order_p))
integral_s = Integral('is', order=2 * max(order_u, order_p))  # for surface terms

# ------------------------------------------------------------------
# 5) Terms (assembled per-region)
#    Mechanics:   dw_lin_elastic
#    Coupling:    dw_piezo_coupling (see docs)  :contentReference[oaicite:2]{index=2}
#    Dielectric:  dw_diffusion
#    Traction:    dw_surface_ltr  (RHS)         :contentReference[oaicite:3]{index=3}
# ------------------------------------------------------------------
# Mechanical stiffness: substrate + piezo
t_mech_sub = Term.new('dw_lin_elastic(sub.D, v, u)', integral_v, Substrate, sub=mat_sub, v=v, u=u)
t_mech_pz  = Term.new('dw_lin_elastic(pz.C,  v, u)', integral_v, Piezo,     pz=mat_pz,  v=v, u=u)

# Piezo coupling: appears with opposite signs in mech vs electric equations. :contentReference[oaicite:4]{index=4}
t_cpl_mech = Term.new('dw_piezo_coupling(pz.e, v, p)', integral_v, Piezo, pz=mat_pz, v=v, p=p)
t_cpl_elec = Term.new('dw_piezo_coupling(pz.e, u, q)', integral_v, Piezo, pz=mat_pz, u=u, q=q)

# Dielectric term in piezo region
t_dielectric = Term.new('dw_diffusion(pz.kappa, q, p)', integral_v, Piezo, pz=mat_pz, q=q, p=p)

# Tip traction on the free end
t_load = Term.new('dw_surface_ltr(load.val, v)', integral_s, FreeEnd, load=load, v=v)

# ------------------------------------------------------------------
# 6) Equations (two-block system)
#    Standard static piezo system:
#      (i)  dw_lin_elastic - dw_piezo_coupling  + dw_surface_ltr = 0
#      (ii)  dw_piezo_coupling + dw_diffusion                       = 0
# ------------------------------------------------------------------
eq_mech = Equation('mech',
                   t_mech_sub + t_mech_pz - t_cpl_mech + t_load)

eq_elec = Equation('elec',
                   t_cpl_elec + t_dielectric)

eqs = Equations([eq_mech, eq_elec])

# ------------------------------------------------------------------
# 7) Problem & Boundary Conditions
# ------------------------------------------------------------------
pb = Problem('piezo_static', equations=eqs)

# Clamp u = 0 at root; ground top electrode p=0
pb.set_bcs(ebcs={
    'clamp' : ('Clamp', {'u.all' : 0.0}),
    'ground': ('TopElectrode', {'p.all' : 0.0}),
})

# NOTE: BottomElectrode is left "floating" (natural Neumann → no-flux),
# which corresponds to open-circuit measurement of induced voltage.

# ------------------------------------------------------------------
# 8) Solvers
# ------------------------------------------------------------------
ls = ScipyDirect({})
nls = Newton({'i_max': 1, 'eps_a': 1e-10})  # linear problem → 1 Newton iter

pb.set_solver(ls)
pb.set_solver(nls)

# ------------------------------------------------------------------
# 9) Solve & Save
# ------------------------------------------------------------------
state = pb.solve()
pb.save_state('03_piezo_static.vtk', state)

print("✅ Done. Results saved to 03_piezo_static.vtk (u, p fields).")
print("   • Visualise p (electric potential) and u (displacement) in ParaView.")
print("   • Voltage (open-circuit) ≈ mean(p) on BottomElectrode (Top is 0 V).")
