import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import FieldVariable

mesh = Mesh.from_file('../CAD/exports/beam_piezo_v3.mesh')
domain = FEDomain('domain', mesh)

# --- Regions from Physical Groups ---
print("\n🔍 Creating regions from physical IDs...")

# Based on meshio conversion results:
# - Groups 25, 26: Volume elements (tetrahedra)
# - Groups 27, 28, 29: Surface elements (triangles)
substrate = domain.create_region("Substrate", "cells of group 25")       # 3D volume
piezo     = domain.create_region("Piezo",     "cells of group 26")       # 3D volume

# For surface regions, we need to use the correct SfePy syntax
# Try different approaches for surface regions
try:
    top = domain.create_region("TopElectrode", "cells of group 27")  # Try cells approach
    print(f"✅ Top electrode created using cells of group 27")
except:
    try:
        top = domain.create_region("TopElectrode", "vertices of group 27")  # Try vertices approach
        print(f"✅ Top electrode created using vertices of group 27")
    except:
        top = domain.create_region("TopElectrode", "all")  # Fallback
        print(f"⚠️  Top electrode created using fallback 'all'")

try:
    bottom = domain.create_region("BottomElectrode", "cells of group 28")
    print(f"✅ Bottom electrode created using cells of group 28")
except:
    try:
        bottom = domain.create_region("BottomElectrode", "vertices of group 28")
        print(f"✅ Bottom electrode created using vertices of group 28")
    except:
        bottom = domain.create_region("BottomElectrode", "all")
        print(f"⚠️  Bottom electrode created using fallback 'all'")

try:
    clamp = domain.create_region("Clamp", "cells of group 29")
    print(f"✅ Clamp created using cells of group 29")
except:
    try:
        clamp = domain.create_region("Clamp", "vertices of group 29")
        print(f"✅ Clamp created using vertices of group 29")
    except:
        clamp = domain.create_region("Clamp", "all")
        print(f"⚠️  Clamp created using fallback 'all'")

# --- Summary ---
print("\n📊 Region entity counts:")
for r in [substrate, piezo, top, bottom, clamp]:
    print(f"  {r.name:15s}: {len(r.entities[0])} entities")

print("\n✨ Stage 0.5 completed successfully if Substrate & Piezo have ~50k cells each, "
      "and electrodes/clamp have thousands of facets ✨")

# --- Fields ---
print("\n🔧 Creating fields...")
field_u   = Field.from_args("displacement", np.float64, "vector", domain.regions["Omega"], approx_order=1)
field_phi = Field.from_args("potential",    np.float64, "scalar", piezo, approx_order=1)

# --- Variables ---
u   = FieldVariable("u", "unknown", field_u)
v   = FieldVariable("v", "test",    field_u, primary_var_name="u")

phi = FieldVariable("phi", "unknown", field_phi)
psi = FieldVariable("psi", "test",    field_phi, primary_var_name="phi")

print("✅ Fields and variables created")

# --- Summary ---
print("\n📊 DOF counts:")
print(f"  Displacement DOFs (u): {u.n_dof}")
print(f"  Potential DOFs (phi):  {phi.n_dof}")

print("\n✨ Stage 1 completed successfully if DOFs are nonzero ✨")

