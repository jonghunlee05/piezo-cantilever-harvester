//-------------------------------------------------
// beam_piezo_v3.geo
// Piezoelectric cantilever mesh for SfePy
// Uses Gmsh v2.2 ASCII format
//-------------------------------------------------

SetFactory("OpenCASCADE");

// === CAD Import ===
Merge "beam_piezo_v0.step";

// === Physical Groups ===
// Volumes (3D domains)
// NOTE: check in Gmsh GUI that substrate = {2}, piezo = {1}.
// If IDs differ, update these numbers.
Physical Volume("Substrate", 25) = {2};
Physical Volume("Piezo",     26) = {1};

// Surfaces (2D boundaries)
// NOTE: check in Gmsh GUI that these IDs match:
//   Top electrode surface = {5}
//   Bottom electrode surface = {6}
//   Root clamp (fixed face[s]) = {2, 8}
Physical Surface("Top_Electrode",    27) = {5};
Physical Surface("Bottom_Electrode", 28) = {6};
Physical Surface("Root_Fixed",       29) = {2, 8};

// === Mesh Size Control ===
// Units = mm (since STEP was in mm)

// Global baseline size
Field[1] = Constant;
Field[1].VIn  = 0.50;   // mm
Field[1].VOut = 0.50;   // mm

// Refine near electrodes
Field[2] = Distance;
Field[2].SurfacesList = {5, 6};  // electrode surface IDs
Field[2].Sampling = 20;

Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = 0.12;  // fine mesh near electrodes
Field[3].SizeMax = 0.80;  // coarse mesh away
Field[3].DistMin = 0.0;
Field[3].DistMax = 0.4;

Field[4] = Min;
Field[4].FieldsList = {1, 3};
Background Field = 4;

// === Mesh Generation ===
Mesh.MshFileVersion = 2.2;  // force v2.2 ASCII for SfePy
Mesh 3;
