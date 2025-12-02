# sim

https://github.com/calculix/CalculiX-Windows/blob/master/releases/CalculiX-2.23.0-win-x64.zip

```bash
# Run sim
cmd
conda activate py313
python stl_voxel_to_ccx_job.py MODELS/cube.stl --out-dir OUTPUT --cube-size 5 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe" --cyl-radius 199.82 --export-lattice
python gen_grid.py MODELS/fugo_para_long.stl cube_job --cube-size 5 --cyl-radius 199.82

# Convert to paraview
python -m ccx2paraview donut_job.frd vtu

# View input file
C:/Users/4y5t6/Downloads/CalculiX-2.23.0-win-x64/bin/cgx.exe -c donut_job.inp
```
