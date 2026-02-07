#!/usr/bin/env python3
"""
STL -> feature-aware isotropic remesh (split-only, 1 iter) -> STL

pip install pymeshlab

Example:
  python remesh_like_cgal.py in.stl out.stl --target-edge 0.25 --feature-deg 1.0
"""

from __future__ import annotations
import argparse
import pymeshlab as pml


def _pure_value(x: float):
    """Absolute value wrapper across PyMeshLab versions."""
    if hasattr(pml, "PureValue"):        # newer (2023.12+)
        return pml.PureValue(x)
    if hasattr(pml, "AbsoluteValue"):    # older (2022.2 and earlier docs)
        return pml.AbsoluteValue(x)
    return x  # last resort (some wrappers accept float directly)


def remesh_like_cgal(input_stl: str, output_stl: str, *, target_edge: float, feature_deg: float) -> None:
    ms = pml.MeshSet()
    ms.load_new_mesh(input_stl)

    # Method name differs in old PyMeshLab (0.2.x) vs newer.
    remesh_fn = getattr(ms, "meshing_isotropic_explicit_remeshing", None) or getattr(
        ms, "remeshing_isotropic_explicit_remeshing", None
    )
    if remesh_fn is None:
        raise RuntimeError("Your PyMeshLab version doesn't expose isotropic explicit remeshing.")

    # Mirrors your CGAL parameters:
    # - number_of_iterations(1)          -> iterations=1
    # - edge_is_constrained_map(eif)     -> featuredeg ~ crease angle to preserve
    # - do_collapse(false)               -> collapseflag=False
    # - do_flip(false)                   -> swapflag=False
    # - number_of_relaxation_steps(0)    -> smoothflag=False
    # - do_project(false)                -> reprojectflag=False
    # We keep splitflag=True (refine) so edges can be split towards target length.
    remesh_fn(
        iterations=1,
        adaptive=False,
        selectedonly=False,
        targetlen=_pure_value(float(target_edge)),  # absolute target edge length
        featuredeg=float(feature_deg),
        checksurfdist=False,
        splitflag=True,
        collapseflag=False,
        swapflag=False,
        smoothflag=False,
        reprojectflag=False,
    )

    ms.save_current_mesh(output_stl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_stl")
    ap.add_argument("output_stl")
    ap.add_argument("--target-edge", type=float, required=True, help="Absolute target edge length (model units).")
    ap.add_argument("--feature-deg", type=float, default=1.0, help="Feature/crease angle in degrees (default 1.0).")
    args = ap.parse_args()

    remesh_like_cgal(
        args.input_stl,
        args.output_stl,
        target_edge=args.target_edge,
        feature_deg=args.feature_deg,
    )


if __name__ == "__main__":
    main()
