# sim

https://github.com/calculix/CalculiX-Windows/blob/master/releases/CalculiX-2.23.0-win-x64.zip

```bash
# Run sim
cmd
conda activate py313
python stl_voxel_to_ccx_job.py MODELS/CSC16_U00P_.stl --out-dir OUTPUT --cube-size 5 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe" --cyl-radius 199.82 --export-lattice
python gen_grid.py MODELS/fugo_para_long.stl cube_job --cube-size 5 --cyl-radius 199.82 --output-stride 5
python apply_ffd_vtk.py MODELS/CSC16_U00P_0.2_remesh_.stl OUTPUT/CSC16_U00P__lattice_ffd.json FFD.stl
python apply_ffd.py --input-stl MODELS\CSC16_U00P_0.2_remesh_.stl --output-stl FFD.stl --lat-base OUTPUT\CSC16_U00P__json_lattice1_base.json --lat-curved OUTPUT\CSC16_U00P__json_lattice2_curved.json --lat-deformed OUTPUT\CSC16_U00P__json_lattice3_deformed.json

python visualize_lattice_mappings.py --lat-base OUTPUT\CSC16_U00P__json_lattice1_base.json --lat-curved OUTPUT\CSC16_U00P__json_lattice2_curved.json --lat-deformed OUTPUT\CSC16_U00P__json_lattice3_deformed.json --out-prefix DEBUG


# New workflow
python stl_voxel_to_ccx_job.py MODELS/CSC16_U00P_.stl --out-dir OUTPUT --cube-size 1 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe" --cyl-radius 199.82 --export-lattice
python TEST.py --input-stl MODELS/CSC16_U00P_.stl --ffd-json-1 OUTPUT/CSC16_U00P__json_lattice2_curved.json --ffd-json-2 OUTPUT/CSC16_U00P__json_lattice3_deformed.json --output-stl FFD.stl

# Tets workflow (NEW) 15 min
python stl_voxel_to_ccx_job_NO_CUBE.py MODELS/CSC16_U00P_.stl --out-dir OUTPUT --cube-size 0.5 --run-ccx --ccx-cmd "C:/Users/4y5t6/Downloads/PrePoMax v2.4.0/Solver/ccx_dynamic.exe" --cyl-radius 199.82

# Convert to paraview
python -m ccx2paraview donut_job.frd vtu

# View input file
C:/Users/4y5t6/Downloads/CalculiX-2.23.0-win-x64/bin/cgx.exe -c donut_job.inp

# Last best script
python CLEAR-slicing-connect-tet-slices-simulate.py
```
