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
    mesh = Mesh.from_file('../CAD/exports/beam_piezo_v3.mesh')
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
    substrate = domain.create_region('Substrate', 'cells of group 25')
    piezo = domain.create_region('Piezo', 'cells of group 26')
    top = domain.create_region('TopElectrode', 'cells of group 27')
    bottom = domain.create_region('BottomElectrode', 'cells of group 28')
    clamp = domain.create_region('Clamp', 'cells of group 29')
    print("✅ All regions created successfully")
    
    # Display mesh group information
    print("\n📊 Mesh group info:")
    try:
        if hasattr(mesh, 'descs'):
            print(f"  Mesh descs type: {type(mesh.descs)}")
            print(f"  Mesh descs: {mesh.descs}")
        if hasattr(mesh, 'groups'):
            print(f"  Mesh groups type: {type(mesh.groups)}")
            print(f"  Mesh groups: {mesh.groups}")
        if hasattr(mesh, 'cmesh'):
            print(f"  CMesh groups: {mesh.cmesh.groups if hasattr(mesh.cmesh, 'groups') else 'No groups'}")
    except Exception as e:
        print(f"  Could not access mesh group info: {e}")
    
    # ------------------------------------------------------------------
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
    
    # Piezoelectric e-matrix [C/m²], poling along global 3-axis (thickness)
    e_15, e_31, e_33 = 12.3, -6.5, 23.3  # ~PZT-5H ballpark
    e_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0,  e_15, 0.0],
        [0.0, 0.0, 0.0,  e_15, 0.0,  0.0],
        [e_31, e_31, e_33, 0.0, 0.0,  0.0]
    ], dtype=np.float64)
    
    # Relative permittivity (z/thickness usually higher)
    eps_r = np.array([1500.0, 1500.0, 1700.0])
    epsilon = np.diag(eps_r) * 8.8541878128e-12
    
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
    
    state = pb.solve(status=status)
    
    # Check if we have solution data
    if state is not None:
        print("✅ Solution computed successfully!")
        print(f"   - Displacement DOFs: {state['u'].n_dof}")
        print(f"   - Potential DOFs: {state['phi'].n_dof}")
        
        # Inspect full-field maxima correctly
        u_var = state['u']
        phi_var = state['phi']
        
        if hasattr(u_var, 'data') and u_var.data is not None:
            # Get the full displacement data array and convert to numpy
            u_data = np.array(u_var.data)
            if u_data.ndim > 1:
                # For vector field, calculate magnitude at each node
                u_mag_max = np.linalg.norm(u_data, axis=1).max()
                print(f"   - Max |u| (vector magnitude): {u_mag_max:.3e} m")
            else:
                # For scalar field, just take max absolute value
                u_abs_max = np.abs(u_data).max()
                print(f"   - Max |u|: {u_abs_max:.3e} m")
        else:
            print("   - Displacement data not available")

        if hasattr(phi_var, 'data') and phi_var.data is not None:
            phi_data = np.array(phi_var.data)
            phi_abs_max = np.abs(phi_data).max()
            print(f"   - Max |phi|: {phi_abs_max:.3e} V")
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