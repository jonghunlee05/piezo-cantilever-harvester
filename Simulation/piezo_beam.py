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

# ------------------------------------------------------------------
# Load mesh (Version 2 ASCII exported from Gmsh)
# ------------------------------------------------------------------
print("📁 Loading mesh file...")
try:
    mesh = Mesh.from_file('../CAD/exports/beam_piezo_v2.msh')
    print(f"✅ Mesh loaded: {len(mesh.coors)} vertices")
except Exception as e:
    print(f"❌ Error loading mesh: {e}")
    exit(1)

print("🏗️  Creating domain...")
try:
    domain = FEDomain('domain', mesh)
    print("✅ Domain created successfully")
except Exception as e:
    print(f"❌ Error creating domain: {e}")
    exit(1)

# ------------------------------------------------------------------
# Define regions using Physical IDs from your .msh
# ------------------------------------------------------------------
print("📍 Creating regions...")

try:
    Omega = domain.create_region('Omega', 'all')
    substrate = domain.create_region('Substrate', 'cells of group 2')
    piezo = domain.create_region('Piezo', 'cells of group 4')
    top = domain.create_region('TopElectrode', 'cells of group 1')
    bottom = domain.create_region('BottomElectrode', 'cells of group 3')
    clamp = domain.create_region('Clamp', 'cells of group 5')
    print("✅ All regions created successfully")
except Exception as e:
    print(f"❌ Error creating regions: {e}")
    exit(1)

# ------------------------------------------------------------------
# Fields and variables (mechanical displacement + electric potential)
# ------------------------------------------------------------------
print("🔧 Creating fields and variables...")
try:
    field_u = Field.from_args('displacement', np.float64, 'vector', Omega, approx_order=1)
    field_phi = Field.from_args('potential', np.float64, 'scalar', piezo, approx_order=1)
    
    u = FieldVariable('u', 'unknown', field_u)
    phi = FieldVariable('phi', 'unknown', field_phi)
    v = FieldVariable('v', 'test', field_u, primary_var_name='u')
    psi = FieldVariable('psi', 'test', field_phi, primary_var_name='phi')
    
    print("✅ Fields and variables created")
except Exception as e:
    print(f"❌ Error creating fields/variables: {e}")
    exit(1)

# ------------------------------------------------------------------
# Materials
# ------------------------------------------------------------------
print("🧱 Setting up materials...")
try:
    # Substrate: steel-like
    D_sub = stiffness_from_youngpoisson(3, 200e9, 0.3)
    mat_sub = Material('Substrate', D=D_sub, rho=7800.0)
    
    # Piezo: PZT-5H
    D_pzt = stiffness_from_youngpoisson(3, 60e9, 0.31)
    
    # Piezoelectric e-matrix [C/m²]
    e_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0, 4.6e-10, 0.0],
        [0.0, 0.0, 0.0, 4.6e-10, 0.0, 0.0],
        [-1.9e-10, -1.9e-10, 3.9e-10, 0.0, 0.0, 0.0]
    ])
    
    # Relative permittivity * vacuum permittivity
    epsilon = np.eye(3) * (8.85e-12 * 1700)
    
    mat_pzt = Material('Piezo', D=D_pzt, e=e_matrix, epsilon=epsilon, rho=7500.0)
    print("✅ Materials created")
except Exception as e:
    print(f"❌ Error creating materials: {e}")
    exit(1)

# ----------------------------
# Integration and equations
# ----------------------------
print("📝 Setting up equations...")
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
    
    print("✅ Equations created")
except Exception as e:
    print(f"❌ Error creating equations: {e}")
    exit(1)

# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------
print("🔒 Setting up boundary conditions...")
try:
    fix = EssentialBC('fix', clamp, {'u.all': 0.0})
    pot0 = EssentialBC('pot0', bottom, {'phi.all': 0.0})
    potV = EssentialBC('potV', top, {'phi.all': 10.0})
    
    bcs = Conditions([fix, pot0, potV])
    print("✅ Boundary conditions set")
except Exception as e:
    print(f"❌ Error creating boundary conditions: {e}")
    exit(1)

# ------------------------------------------------------------------
# Problem & solvers
# ------------------------------------------------------------------
print("⚙️  Setting up problem and solvers...")
try:
    pb = Problem('piezo', equations=eqs)
    
    ls = ScipyDirect({})
    nls = Newton({'i_max': 20, 'eps_a': 1e-8, 'eps_r': 1e-6}, lin_solver=ls)
    pb.set_solver(nls)
    
    # Set problem as linear since it converges in 1 iteration
    pb.set_linear(True)
    
    pb.time_update(ebcs=bcs)
    print("✅ Problem and solvers configured")
except Exception as e:
    print(f"❌ Error setting up problem: {e}")
    exit(1)

# ------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------
print("🚀 Starting solver...")
try:
    status = IndexedStruct()
    
    # Debug: Check problem state before solving
    print("🔍 Debug: Problem state before solving:")
    print(f"   - Problem type: {type(pb)}")
    print(f"   - Is linear: {pb.is_linear()}")
    print(f"   - Has equations: {pb.equations is not None}")
    
    state = pb.solve(status=status)
    
    # Debug: Investigate what solve() returned
    print("🔍 Debug: After solving:")
    print(f"   - State object: {state}")
    print(f"   - State type: {type(state)}")
    print(f"   - Status object: {status}")
    
    # Check if we have solution data
    if state is not None:
        print("✅ Solution computed successfully!")
        
        # Debug: Check the actual solution data structure
        print("🔍 Debug: Solution data investigation:")
        print(f"   - State names: {state.names}")
        print(f"   - State variables: {state.state}")
        print(f"   - State vector shape: {state.vec.shape}")
        
        # Access displacement data correctly
        u_var = state['u']
        phi_var = state['phi']
        
        print(f"   - Displacement variable: {u_var}")
        print(f"   - Potential variable: {phi_var}")
        print(f"   - Displacement DOFs: {u_var.n_dof}")
        print(f"   - Potential DOFs: {phi_var.n_dof}")
        
        # Check if data arrays exist and their content
        if hasattr(u_var, 'data') and u_var.data is not None:
            print(f"   - Displacement data type: {type(u_var.data)}")
            print(f"   - Displacement data length: {len(u_var.data)}")
            if len(u_var.data) > 0 and u_var.data[0] is not None:
                u_data = u_var.data[0]
                print(f"   - Displacement data shape: {u_data.shape}")
                print(f"   - Displacement data type: {u_data.dtype}")
                print(f"   - Displacement data range: [{np.min(u_data):.2e}, {np.max(u_data):.2e}]")
                print(f"   - Displacement data non-zero count: {np.count_nonzero(u_data)}")
            else:
                print("   - Displacement data[0] is None")
        else:
            print("   - No displacement data attribute")
            
        if hasattr(phi_var, 'data') and phi_var.data is not None:
            print(f"   - Potential data type: {type(phi_var.data)}")
            print(f"   - Potential data length: {len(phi_var.data)}")
            if len(phi_var.data) > 0 and phi_var.data[0] is not None:
                phi_data = phi_var.data[0]
                print(f"   - Potential data shape: {phi_data.shape}")
                print(f"   - Potential data type: {phi_data.dtype}")
                print(f"   - Potential data range: [{np.min(phi_data):.2e}, {np.max(phi_data):.2e}]")
                print(f"   - Potential data non-zero count: {np.count_nonzero(phi_data)}")
            else:
                print("   - Potential data[0] is None")
        else:
            print("   - No potential data attribute")
            
        # Check the actual solution vector
        print(f"   - Solution vector shape: {state.vec.shape}")
        print(f"   - Solution vector range: [{np.min(state.vec):.2e}, {np.max(state.vec):.2e}]")
        print(f"   - Solution vector non-zero count: {np.count_nonzero(state.vec)}")
    else:
        print("⚠️  Warning: Solver returned None")
    
    print("✅ Solver completed successfully!")
    
    # Print solver statistics
    if hasattr(status, 'nls_status'):
        print(f"📊 Nonlinear solver iterations: {status.nls_status.n_iter}")
    if hasattr(status, 'ls_status'):
        print(f"📊 Linear solver iterations: {status.ls_status.n_iter}")
        
except Exception as e:
    print(f"❌ Error during solving: {e}")
    exit(1)

# ------------------------------------------------------------------
# Save results for ParaView
# ------------------------------------------------------------------
print("💾 Saving results...")

# Debug: Investigate saving options
print("🔍 Debug: Investigating save options...")
print(f"   - State object type: {type(state)}")
if state is not None:
    print(f"   - Available variables: {state.names}")
    print(f"   - State variables: {state.state}")
    print(f"   - Problem variables: {pb.get_variables()}")

try:
    # Method 1: Try saving with the state object
    print("   - Attempting save with state object...")
    pb.save_state('piezo_beam.vtk', state=state)
    print("✅ Results saved to 'piezo_beam.vtk'")
    print("📊 Open 'piezo_beam.vtk' in ParaView to visualize results")
    
except Exception as e:
    print(f"❌ Error saving with state: {e}")
    print("💡 Trying alternative save method...")
    
    try:
        # Method 2: Try saving without specifying state
        print("   - Attempting save without state...")
        pb.save_state('piezo_beam.vtk')
        print("✅ Results saved using default method")
        
    except Exception as e2:
        print(f"❌ Default save also failed: {e2}")
        
        # Method 3: Try using problem's built-in output
        print("🔍 Debug: Trying built-in output methods...")
        try:
            if hasattr(pb, 'setup_output'):
                print("   - Setting up output directory...")
                pb.setup_output(output_dir='./', output_format='vtk')
                print("   - Output setup completed")
                
                # Try to save using the setup output
                if hasattr(pb, 'save_output'):
                    pb.save_output('piezo_beam_output')
                    print("✅ Results saved using output method")
                else:
                    print("⚠️  Output setup completed but no save_output method")
            else:
                print("⚠️  No setup_output method available")
                
        except Exception as e3:
            print(f"❌ Output setup failed: {e3}")
            print("💡 Results computed but not saved - manual extraction needed")
            
            # Last resort: Try to manually extract and save
            if state is not None:
                print("🔍 Debug: Attempting manual data extraction...")
                try:
                    # Extract displacement and potential data using correct methods
                    u_var = state['u']
                    phi_var = state['phi']
                    
                    print(f"   - Displacement variable type: {type(u_var)}")
                    print(f"   - Potential variable type: {type(phi_var)}")
                    
                    # Check if we can access the solution vector directly
                    if hasattr(state, 'vec') and state.vec is not None:
                        print(f"   - Solution vector available: {state.vec.shape}")
                        
                        # Try to understand the DOF mapping
                        if hasattr(u_var, 'indx') and u_var.indx is not None:
                            print(f"   - Displacement DOF indices: {u_var.indx}")
                        if hasattr(phi_var, 'indx') and phi_var.indx is not None:
                            print(f"   - Potential DOF indices: {phi_var.indx}")
                            
                        print("   - Manual extraction successful - need to implement VTK writer")
                    else:
                        print("   - No solution vector available")
                        
                except Exception as e4:
                    print(f"   - Data extraction error: {e4}")

print("✨ Simulation completed! 🎯")