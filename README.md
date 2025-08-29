# Piezoelectric Cantilever Beam Energy Harvester

Finite element simulation in Python of a piezoelectric cantilever beam for vibration energy harvesting.  
This project models the coupled electromechanical response of a cantilever beam with surface-mounted piezoelectric layers and predicts harvested voltage, current, power, and energy under different vibration conditions.

---

## Overview

Piezoelectric cantilevers are widely studied as energy harvesters for self-powered sensors and low-power electronics.  
The goal of this project was to implement a complete computational pipeline for evaluating the mechanical and electrical response of such devices. The simulation workflow includes modal analysis, harmonic response, transient response under base excitation, and electrical circuit coupling through an external resistive load. Post-processing routines provide quantitative measures of harvested energy and performance maps across operating conditions.

---

## Features

- **Finite Element Modeling (FEM)** with [SfePy](http://sfepy.org/)  
- **Analyses implemented:**
  - Modal analysis (natural frequencies and mode shapes)
  - Static response (boundary condition and coupling check)
  - Harmonic sweep (frequency response and resonance amplification)
  - Circuit coupling with resistive loads
  - Transient response with multiple excitation types (sinusoidal, step, random)
- **Post-processing:**
  - Time-series of displacement, voltage, current, and power
  - Cumulative harvested energy (time-integrated power)
  - FFT spectra of displacement and voltage
  - Parametric sweeps (frequency × resistance) with summary tables and heatmaps

---

## Example Results

- Frequency response curves
- Transient waveforms of voltage and displacement
- Cumulative energy harvested over time
- Heatmap of maximum power versus frequency and resistance

*(Example plots from `results/` can be embedded here to illustrate outputs.)*

---

## Repository Structure

01_modal.py # Modal analysis
02_static.py # Static check
03_harmonic_base.py # Harmonic response
04_circuit_coupling.py # Circuit coupling sweep
05_transient.py # Transient response (sin/step/random)
06_sweep_summary.py # Parametric sweeps + summary aggregation
CAD/exports/ # Mesh files (Gmsh exports)
results/ # Example output data and plots

yaml
코드 복사

---

## Dependencies

- Python 3.10+
- [SfePy](http://sfepy.org/) (Finite Element framework)
- NumPy
- SciPy
- Matplotlib
- Pandas

Install all dependencies with:

```bash
pip install sfepy numpy scipy matplotlib pandas
Running the Code
Clone the repository and navigate into the project folder.

Run any script, for example:


python transient.py
Results (plots and CSV files) are saved to the working directory or into results/.

For parametric studies, run:

python sweep_summary.py
This generates a sweep_summary.csv, line plots, and heatmaps.

Motivation
Vibration energy harvesting is an important research area for enabling autonomous and low-power devices. Piezoelectric cantilevers are one of the most common designs due to their simplicity and high electromechanical coupling.
This project demonstrates a complete simulation workflow for evaluating such harvesters, from mechanical vibrations to electrical output and harvested energy. It highlights the integration of finite element methods, dynamic simulation, and circuit modeling in a single Python framework.