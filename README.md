# sim

https://github.com/calculix/CalculiX-Windows/blob/master/releases/CalculiX-2.23.0-win-x64.zip

```bash
cmd
conda activate py313
python stl_voxel_to_ccx_job.py donut_.stl donut_job --cube-size 3 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/CalculiX-2.23.0-win-x64/bin/ccx.exe" --base-temp 293 --heat-flux 1e3
python stl_voxel_to_ccx_job.py donut_.stl donut_job --cube-size 3 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe"  --base-temp 293 --heat-flux 1e3
python -m ccx2paraview donut_job.frd vtu

C:/Users/4y5t6/Downloads/CalculiX-2.23.0-win-x64/bin/cgx.exe -c donut_job.inp
```

Last working

```
python stl_voxel_to_ccx_job.py donut_.stl donut_job --cube-size 5 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe"  --base-temp 100 --heat-flux 1e5 --export-lattice
```

python stl_voxel_to_ccx_job.py donut_vert_.stl cube_job --cube-size 0.75 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe"