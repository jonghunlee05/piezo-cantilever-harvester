SetFactory("OpenCASCADE");
Merge "beam_piezo_v0.step";
//+
Physical Volume("Substrate", 25) = {2};
//+
Physical Volume("Piezo", 26) = {1};
//+
Physical Surface("Top_Electrode", 27) = {5};
//+
Physical Surface("Bottom_Electrode", 28) = {6};
//+
Physical Surface("Root_Fixed", 29) = {2, 8};
//+
Field[1] = Distance;
//+
Field[2] = Threshold;
//+
Field[2].DistMax = 0.4;
//+
Field[2].DistMin = 0;
//+
Field[2].InField = 1;
//+
Field[2].SizeMax = 0.8;
//+
Field[2].SizeMin = 0.12;
