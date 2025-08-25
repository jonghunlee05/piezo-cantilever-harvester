# Piezoelectric Cantilever Beam Energy Harvester

**Objective:** Design and simulate a piezoelectric cantilever beam that turns vibration into voltage.
- Show mechanical deformation → voltage output
- Compare aluminium vs steel substrate
- Plot voltage vs vibration frequency

## Project Overview

This project focuses on designing and analyzing a piezoelectric cantilever beam for energy harvesting applications. The cantilever beam will convert mechanical vibrations into electrical energy through the piezoelectric effect, demonstrating the relationship between mechanical deformation and voltage output.

## Key Objectives

1. **Mechanical-Electrical Coupling**: Demonstrate how mechanical deformation translates to voltage output
2. **Material Comparison**: Analyze performance differences between aluminium and steel substrates
3. **Frequency Response**: Plot voltage output versus vibration frequency to identify optimal operating conditions

## Technical Specifications

- **Beam Dimensions**: 50 × 10 × 1 mm (length × width × thickness)
- **Piezoelectric Layer**: 0.2 mm thickness on top surface
- **Substrate Materials**: Aluminium and Steel for comparison
- **Analysis Type**: Piezoelectric coupling simulation

## Milestone 0 – Setup Checklist

**Software Requirements (tick if you have it):**
- [ ] SolidWorks installed (use OnShape for free version)
- [ ] ANSYS Workbench (with Piezo coupling) installed
- [ ] A place to save results (this folder)

## Next Steps

**Immediate Action Required:**
Create a simple rectangular cantilever (50×10×1 mm) in CAD with a 0.2 mm piezo layer on top and export a STEP file.

## Project Structure

```
Piezo_Cantilever/
├── README.md
├── CAD/                    # SolidWorks files and STEP exports
├── Simulation/             # ANSYS simulation files
├── Results/                # Analysis results and plots
└── Documentation/          # Additional project documentation
```

## Expected Deliverables

1. **CAD Model**: SolidWorks assembly with cantilever beam and piezoelectric layer
2. **Simulation Results**: ANSYS analysis showing deformation and voltage output
3. **Performance Comparison**: Aluminium vs Steel substrate analysis
4. **Frequency Response**: Voltage vs frequency plots
5. **Technical Report**: Summary of findings and recommendations

## Notes

- Ensure proper piezoelectric material properties are defined in ANSYS
- Consider boundary conditions and loading scenarios
- Document all simulation parameters for reproducibility
