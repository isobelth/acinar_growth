# Modelling the Impact of Extracellular Matrix Stiffness on Mammary Epithelial Cells
- This repository contains an agent-based computational model incorporating cell-cell and cell-ECM forces, alongside cellular processes of growth, division and apoptosis
- The simulation has been used to investigate acinar growth within an embedding matrix

## Example Simulation Outputs
<table>
  <tr>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run0_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run1_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run2_output_15s.gif" width="300"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run3_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run4_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run5_output_15s.gif" width="300"></td>
  </tr>
    <tr>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run6_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run7_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run8_output_15s.gif" width="300"></td>
  </tr>
      <tr>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run9_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run10_output_15s.gif" width="300"></td>
    <td><img src="https://github.com/isobelth/acinar_growth/blob/main/Output_Gifs/Run11_output_15s.gif" width="300"></td>
  </tr>
</table>

## Files Contained in the Repo
- Simulation.py
  - Main simulation file
  - Code employs an ordinary differential equation solver to update the positions of interacting bodies at each timepoint, with biological events triggered based on predefined conditions. 
- Execute_Simulation.ipynb
- Parameter_Optimisation.ipynb
  - Outlines target functions and contains code using TPE to find biologically meaningful parameter sets
- Animate_Simulation_Outputs.ipynb
    - Contains code using vPython to animate individual simulations, or create .png files of final acinar configurations
 
## Simulation.py Flow
- The cells and lumen are modelled as spheres, whose central positions are updated at each timestep in a direction that reduces the mechanical energy of the aggregate.
- The cellular aggregate is surrounded by a BM and embedded in a matrix, the combined effect of which can be modelled as a series of elastic sheets surrounding the acinus.
- Simulated bodies experience forces due to their neighbours. Cells on the acinus exterior also experience forces due to the BM and embedding matrix, and an outward pressure due to the growing aggregate.
- At birth, cells are assigned a normally distributed lifetime. Their radius is incremented linearly as they age. Once they exceed their lifetime, they undergo division into two daughter cells.
- Cells whose centre lies within a set distance of the lumen "undergo apoptosis" and are removed from the simulation. A proportion of their mass is redistributed to the lumen, whose volume increases as a result of the apoptotic event.
![code_flow](https://github.com/user-attachments/assets/8a882c51-650d-4ab6-a759-b80c20f019c4)
