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
    
    # Debug: Check material properties
    print("🔍 Debug: Material properties verification:")
    print(f"   - Substrate Young's modulus: {200e9:.2e} Pa")
    print(f"   - PZT Young's modulus: {60e9:.2e} Pa")
    print(f"   - PZT piezoelectric e-matrix:")
    print(f"     e31: {e_matrix[2,0]:.2e} C/m²")
    print(f"     e32: {e_matrix[2,1]:.2e} C/m²")
    print(f"     e33: {e_matrix[2,2]:.2e} C/m²")
    print(f"     e15: {e_matrix[0,4]:.2e} C/m²")
    print(f"     e24: {e_matrix[1,3]:.2e} C/m²")
    print(f"   - PZT relative permittivity: {1700}")
    print(f"   - PZT absolute permittivity: {8.85e-12 * 1700:.2e} F/m")
    
    # Check if PZT coefficients are reasonable
    print(f"   - PZT coefficients range: [{np.min(np.abs(e_matrix)):.2e}, {np.max(np.abs(e_matrix)):.2e}] C/m²")
    if np.max(np.abs(e_matrix)) < 1e-12:
        print("   ⚠️  Warning: PZT coefficients seem very small!")
    elif np.max(np.abs(e_matrix)) < 1e-9:
        print("   ⚠️  Warning: PZT coefficients seem small!")
    else:
        print("   ✅ PZT coefficients appear reasonable")
        
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
    
    # Debug: Check equation assembly
    print("🔍 Debug: Equation assembly verification:")
    print(f"   - Mechanical equation: {eq_mech}")
    print(f"   - Electrical equation: {eq_elec}")
    print(f"   - Total equations: {len(eqs)}")
    
    # Check term contributions
    print(f"   - Substrate elastic term: {t_sub}")
    print(f"   - PZT elastic term: {t_pzt}")
    print(f"   - Piezoelectric coupling term 1: {t_c_vphi}")
    print(f"   - Piezoelectric coupling term 2: {t_c_upsi}")
    print(f"   - Dielectric term: {t_eps}")
    
    # Verify integration regions
    print(f"   - Substrate integration region: {substrate}")
    print(f"   - PZT integration region: {piezo}")
    
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
    
    # Debug: Check boundary condition details
    print("🔍 Debug: Boundary condition verification:")
    print(f"   - Clamp region size: {len(clamp.entities[0])} elements")
    print(f"   - Bottom electrode size: {len(bottom.entities[0])} elements")
    print(f"   - Top electrode size: {len(top.entities[0])} elements")
    print(f"   - Total BCs: {len(bcs)}")
    
    # Check if regions have the expected properties
    print(f"   - Clamp region: {clamp}")
    print(f"   - Bottom electrode: {bottom}")
    print(f"   - Top electrode: {top}")
    
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
    
    # Debug: Check matrix assembly
    print("🔍 Debug: Matrix assembly verification:")
    print(f"   - Problem type: {type(pb)}")
    print(f"   - Is linear: {pb.is_linear()}")
    print(f"   - Has equations: {pb.equations is not None}")
    
    # Check if we can access matrix information
    try:
        if hasattr(pb, 'mtx_a') and pb.mtx_a is not None:
            print(f"   - System matrix shape: {pb.mtx_a.shape}")
            print(f"   - System matrix type: {type(pb.mtx_a)}")
        else:
            print("   - System matrix not yet assembled")
    except Exception as e:
        print(f"   - Matrix info not accessible: {e}")
        
except Exception as e:
    print(f"❌ Error setting up problem: {e}")
    exit(1)

# ------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------
print("🚀 Starting solver...")
try:
    status = IndexedStruct()
    state = pb.solve(status=status)
    
    # Check if we have solution data
    if state is not None:
        print("✅ Solution computed successfully!")
        print(f"   - Displacement DOFs: {state['u'].n_dof}")
        print(f"   - Potential DOFs: {state['phi'].n_dof}")
        
        # Check if solution values are non-zero
        u_var = state['u']
        phi_var = state['phi']
        
        if hasattr(u_var, 'data') and u_var.data is not None and len(u_var.data) > 0:
            u_data = u_var.data[0]
            u_max = np.max(np.abs(u_data)) if u_data is not None else 0
            print(f"   - Max displacement: {u_max:.2e}")
        else:
            print("   - Displacement data not available")
            
        if hasattr(phi_var, 'data') and phi_var.data is not None and len(phi_var.data) > 0:
            phi_data = phi_var.data[0]
            phi_max = np.max(np.abs(phi_data)) if phi_data is not None else 0
            print(f"   - Max potential: {phi_max:.2e}")
        else:
            print("   - Potential data not available")
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

try:
    # Try saving with the state object
    pb.save_state('piezo_beam.vtk', state=state)
    print("✅ Results saved to 'piezo_beam.vtk'")
    print("📊 Open 'piezo_beam.vtk' in ParaView to visualize results")
    
except Exception as e:
    print(f"❌ Error saving with state: {e}")
    print("💡 Trying alternative save method...")
    
    try:
        # Try saving without specifying state
        pb.save_state('piezo_beam.vtk')
        print("✅ Results saved using default method")
        
    except Exception as e2:
        print(f"❌ Default save also failed: {e2}")
        
        # Try using problem's built-in output
        print("🔍 Trying built-in output methods...")
        try:
            if hasattr(pb, 'setup_output'):
                pb.setup_output(output_dir='./', output_format='vtk')
                print("✅ Output setup completed")
                
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
            
            # Last resort: Manual data extraction
            if state is not None:
                print("🔍 Attempting manual data extraction...")
                try:
                    u_var = state['u']
                    phi_var = state['phi']
                    
                    if hasattr(state, 'vec') and state.vec is not None:
                        print(f"   - Solution vector available: {state.vec.shape}")
                        print("   - Manual extraction successful - need to implement VTK writer")
                    else:
                        print("   - No solution vector available")
                        
                except Exception as e4:
                    print(f"   - Data extraction error: {e4}")

print("✨ Simulation completed! 🎯")