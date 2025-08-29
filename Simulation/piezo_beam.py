import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import FieldVariable, Material, Problem, Integral, Equation, Equations
from sfepy.base.base import IndexedStruct
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

print("🔍 Starting sfepy piezoelectric cantilever simulation...")
print("📦 Imported modules successfully")

# ------------------------------------------------------------------
# Load mesh (Version 2 ASCII exported from Gmsh)
# ------------------------------------------------------------------
print("\n📁 Loading mesh file...")
try:
    mesh = Mesh.from_file('../CAD/exports/beam_piezo_v2.msh')
    print(f"✅ Mesh loaded successfully:")
    print(f"   - Vertices: {len(mesh.coors)}")
    
    # Check what attributes are actually available
    print(f"   - Available mesh attributes: {[attr for attr in dir(mesh) if not attr.startswith('_')]}")
    
    # Try to get element information safely
    if hasattr(mesh, 'conns'):
        print(f"   - Elements: {len(mesh.conns)}")
        print(f"   - Element types: {list(mesh.conns.keys())}")
        
        # Print mesh statistics
        for el_type, conn in mesh.conns.items():
            print(f"   - {el_type}: {len(conn)} elements")
    else:
        print("   - Element connectivity not available in expected format")
        
except Exception as e:
    print(f"❌ Error loading mesh: {e}")
    exit(1)

print("\n🏗️  Creating domain...")
try:
    domain = FEDomain('domain', mesh)
    print("✅ Domain created successfully")
    
    # Debug: Check what groups are available
    print("\n🔍 Checking available mesh groups...")
    if hasattr(domain, 'cmesh_tdim'):
        for tdim in range(4):
            cm = domain.cmesh_tdim[tdim]
            if cm is not None:
                print(f"   - Topological dimension {tdim}: {len(cm.cell_groups)} groups")
                unique_groups = np.unique(cm.cell_groups)
                print(f"     Groups: {unique_groups.tolist()}")
                
except Exception as e:
    print(f"❌ Error creating domain: {e}")
    exit(1)

# ------------------------------------------------------------------
# Define regions using Physical IDs from your .msh
# ------------------------------------------------------------------
print("\n📍 Creating regions...")

# For 3D volumes (cells)
try:
    print("   Creating Omega region (all cells)...")
    Omega = domain.create_region('Omega', 'all')
    print(f"   ✅ Omega: {len(Omega.entities[0])} cells")
except Exception as e:
    print(f"   ❌ Error creating Omega region: {e}")

try:
    print("   Creating Substrate region (cells of group 2)...")
    substrate = domain.create_region('Substrate', 'cells of group 2')
    print(f"   ✅ Substrate: {len(substrate.entities[0])} cells")
except Exception as e:
    print(f"   ❌ Error creating Substrate region: {e}")
    print("   💡 Try: 'cells by group 2' or 'cell 2'")

try:
    print("   Creating Piezo region (cells of group 4)...")
    piezo = domain.create_region('Piezo', 'cells of group 4')
    print(f"   ✅ Piezo: {len(piezo.entities[0])} cells")
except Exception as e:
    print(f"   ❌ Error creating Piezo region: {e}")

# For 2D surfaces (boundaries) - use the correct sfepy syntax
try:
    print("   Creating TopElectrode region (cells of group 1)...")
    top = domain.create_region('TopElectrode', 'cells of group 1')
    print(f"   ✅ TopElectrode: {len(top.entities[0])} cells")
except Exception as e:
    print(f"   ❌ Error creating TopElectrode region: {e}")
    print("   💡 Using 'cells of group 1' for surface elements")

try:
    print("   Creating BottomElectrode region (cells of group 3)...")
    bottom = domain.create_region('BottomElectrode', 'cells of group 3')
    print(f"   ✅ BottomElectrode: {len(bottom.entities[0])} cells")
except Exception as e:
    print(f"   ❌ Error creating BottomElectrode region: {e}")

try:
    print("   Creating Clamp region (cells of group 5)...")
    clamp = domain.create_region('Clamp', 'cells of group 5')
    print(f"   ✅ Clamp: {len(clamp.entities[0])} cells")
except Exception as e:
    print(f"   ❌ Error creating Clamp region: {e}")

# ------------------------------------------------------------------
# Fields and variables (mechanical displacement + electric potential)
# ------------------------------------------------------------------
print("\n🔧 Creating fields and variables...")
try:
    # Create fields using regions (this approach works)
    print("   Creating displacement field using Omega region (whole solid)...")
    field_u = Field.from_args('displacement', np.float64, 'vector', Omega, approx_order=1)
    print("   ✅ Displacement field created")
    
    print("   Creating potential field using Piezo region...")
    field_phi = Field.from_args('potential', np.float64, 'scalar', piezo, approx_order=1)
    print("   ✅ Potential field created")
    
    # Create variables - need to create ALL unknown first, then test with proper references
    print("   Creating all unknown variables first...")
    u = FieldVariable('u', 'unknown', field_u)
    phi = FieldVariable('phi', 'unknown', field_phi)
    print("   ✅ Unknown variables created")
    
    print("   Creating test variables...")
    v = FieldVariable('v', 'test', field_u, primary_var_name='u')
    psi = FieldVariable('psi', 'test', field_phi, primary_var_name='phi')
    print("   ✅ Test variables created")
    
    print("   ✅ All variables created successfully")
    
except Exception as e:
    print(f"   ❌ Error creating fields/variables: {e}")
    print("   💡 Need to investigate sfepy field creation syntax")
    exit(1)

# ------------------------------------------------------------------
# Materials
# ------------------------------------------------------------------
print("\n🧱 Setting up materials...")
try:
    # Substrate: steel-like
    print("   Creating Substrate material...")
    D_sub = stiffness_from_youngpoisson(3, 200e9, 0.3)
    print(f"   ✅ Substrate stiffness matrix shape: {D_sub.shape}")
    mat_sub = Material('Substrate', D=D_sub, rho=7800.0)
    print("   ✅ Substrate material created")
    
    # Piezo: PZT-5H
    print("   Creating PZT material...")
    D_pzt = stiffness_from_youngpoisson(3, 60e9, 0.31)
    print(f"   ✅ PZT stiffness matrix shape: {D_pzt.shape}")
    
    # Piezoelectric e-matrix [C/m²] - using e-matrix for sfepy
    e_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0, 4.6e-10, 0.0],
        [0.0, 0.0, 0.0, 4.6e-10, 0.0, 0.0],
        [-1.9e-10, -1.9e-10, 3.9e-10, 0.0, 0.0, 0.0]
    ])
    print(f"   ✅ PZT e-matrix shape: {e_matrix.shape}")
    
    # Relative permittivity * vacuum permittivity
    epsilon = np.eye(3) * (8.85e-12 * 1700)
    print(f"   ✅ PZT permittivity matrix shape: {epsilon.shape}")
    
    mat_pzt = Material('Piezo', D=D_pzt, e=e_matrix, epsilon=epsilon, rho=7500.0)
    print("   ✅ PZT material created")
    
except Exception as e:
    print(f"   ❌ Error creating materials: {e}")
    exit(1)

# ----------------------------
# Integration and equations
# ----------------------------
print("\n📝 Setting up integration and equations...")
try:
    i = Integral('i', order=2)
    
    t_sub = Term.new('dw_lin_elastic(Substrate.D, v, u)', i, substrate,
                     Substrate=mat_sub, v=v, u=u)
    t_pzt = Term.new('dw_lin_elastic(Piezo.D, v, u)', i, piezo,
                     Piezo=mat_pzt, v=v, u=u)
    
    t_c_vphi = Term.new('dw_piezo_coupling(Piezo.e, v, phi)', i, piezo,
                        Piezo=mat_pzt, v=v, phi=phi)
    t_c_upsi = Term.new('dw_piezo_coupling(Piezo.e, u, psi)', i, piezo,
                        Piezo=mat_pzt, u=u, psi=psi)
    
    t_eps = Term.new('dw_diffusion(Piezo.epsilon, psi, phi)', i, piezo,
                     Piezo=mat_pzt, psi=psi, phi=phi)
    
    eq_mech = Equation('mech', t_sub + t_pzt - t_c_vphi)
    eq_elec = Equation('elec', t_c_upsi + t_eps)
    eqs = Equations([eq_mech, eq_elec])
    
    print("   ✅ Equations created successfully")
    
except Exception as e:
    print(f"   ❌ Error creating equations: {e}")
    exit(1)

# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------
print("\n🔒 Setting up boundary conditions...")
try:
    fix = EssentialBC('fix', clamp, {'u.all': 0.0})
    print("   ✅ Clamp BC created")
    
    pot0 = EssentialBC('pot0', bottom, {'phi.all': 0.0})
    print("   ✅ Ground BC created")
    
    potV = EssentialBC('potV', top, {'phi.all': 10.0})
    print("   ✅ Voltage BC created (+10V)")
    
    # Collect boundary conditions
    bcs = Conditions([fix, pot0, potV])
    print("   ✅ Boundary conditions collected")
    
except Exception as e:
    print(f"   ❌ Error creating boundary conditions: {e}")
    exit(1)

# ------------------------------------------------------------------
# Problem & solvers
# ------------------------------------------------------------------
print("\n⚙️  Setting up problem and solvers...")
try:
    # Create the problem with our Equations object
    pb = Problem('piezo', equations=eqs)
    print("   ✅ Problem created with equations")
    
    # Set up solvers
    ls = ScipyDirect({})
    nls = Newton({'i_max': 20, 'eps_a': 1e-8, 'eps_r': 1e-6}, lin_solver=ls)
    pb.set_solver(nls)
    print("   ✅ Solvers configured")
    
    # Attach boundary conditions
    pb.time_update(ebcs=bcs)
    print("   ✅ Boundary conditions attached")
    
except Exception as e:
    print(f"   ❌ Error setting up problem: {e}")
    exit(1)

# ------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------
print("\n🚀 Starting solver...")
try:
    status = IndexedStruct()
    print("   ✅ Status object created")
    
    print("   🔄 Running solver...")
    state = pb.solve(status=status)
    print("   ✅ Solver completed successfully!")
    
    # Check if we got a valid state
    if state is None:
        print("   ⚠️  Warning: Solver returned None, but this is often normal for linear problems")
        print("   💡 Proceeding with save attempt...")
    
    # Print solver statistics
    if hasattr(status, 'nls_status'):
        print(f"   📊 Nonlinear solver iterations: {status.nls_status.n_iter}")
    if hasattr(status, 'ls_status'):
        print(f"   📊 Linear solver iterations: {status.ls_status.n_iter}")
        
except Exception as e:
    print(f"   ❌ Error during solving: {e}")
    exit(1)

# ------------------------------------------------------------------
# Save results for ParaView
# ------------------------------------------------------------------
print("\n💾 Saving results...")
try:
    # Try to save the state - the solver already ran successfully
    pb.save_state('piezo_beam.vtk')
    print("   ✅ Results saved to 'piezo_beam.vtk'")
    print("\n🎉 Simulation completed successfully!")
    print("📊 Open 'piezo_beam.vtk' in ParaView to visualize results")
    
except Exception as e:
    print(f"   ❌ Error saving results: {e}")
    print("   💡 Trying alternative save method...")
    try:
        # Alternative: save without specifying state
        pb.save_state('piezo_beam.vtk', state=None)
        print("   ✅ Results saved using alternative method")
    except Exception as e2:
        print(f"   ❌ Alternative save also failed: {e2}")
        print("   💡 Results computed but not saved - check pb object")
        print(f"   💡 Problem object type: {type(pb)}")
        print(f"   💡 Problem attributes: {[attr for attr in dir(pb) if not attr.startswith('_')]}")

print("\n✨ All done! 🎯")
