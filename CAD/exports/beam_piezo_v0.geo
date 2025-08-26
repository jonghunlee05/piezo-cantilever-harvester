SetFactory("OpenCASCADE");

// --- CAD ---
Merge "beam_piezo_v0.step";

// --- Physical groups (keep your IDs) ---
Physical Volume("Substrate", 25) = {2};
Physical Volume("Piezo",     26) = {1};

Physical Surface("Top_Electrode",    27) = {5};
Physical Surface("Bottom_Electrode", 28) = {6};
Physical Surface("Root_Fixed",       29) = {2, 8};

// ===== Mesh size control (mm units in Gmsh) =====

// Global baseline size (coarser away from piezo)
Field[1] = Constant;
Field[1].VIn  = 0.50;   // mm
Field[1].VOut = 0.50;   // mm

// Distance from the *elementary* electrode faces
Field[2] = Distance;
Field[2].SurfacesList = {5, 6};  // <-- electrode surface IDs
Field[2].Sampling = 20;

// Refine near the piezo layer
Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = 0.12;  // fine size near electrodes (mm)
Field[3].SizeMax = 0.80;  // coarse size away (mm)
Field[3].DistMin = 0.0;
Field[3].DistMax = 0.4;   // refine within ~0.4 mm

// Use the finest of (global, refined)
Field[4] = Min;
Field[4].FieldsList = {1, 3};
Background Field = 4;

// Generate mesh if running via CLI
Mesh 3;
// Mesh.MshFileVersion = 4.1; // optional: force 4.1
