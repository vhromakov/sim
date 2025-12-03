#!/usr/bin/env python3
import json
import numpy as np
import trimesh
from pygem import FFD
import argparse


def load_ffd(path):
    with open(path, "r") as f:
        data = json.load(f)

    nx, ny, nz = data["nx"], data["ny"], data["nz"]

    ffd = FFD(n_control_points=[nx, ny, nz])
    ffd.box_origin[:] = np.array(data["box_origin"], dtype=float)
    ffd.box_length[:] = np.array(data["box_length"], dtype=float)

    ffd.array_mu_x[:, :, :] = np.array(data["mu_x"], dtype=float)
    ffd.array_mu_y[:, :, :] = np.array(data["mu_y"], dtype=float)
    ffd.array_mu_z[:, :, :] = np.array(data["mu_z"], dtype=float)

    return ffd


def apply_ffd_to_stl(input_stl, ffd_json, output_stl):
    ffd = load_ffd(ffd_json)

    mesh = trimesh.load(input_stl)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    orig = mesh.vertices.copy()
    mesh.vertices = ffd(orig)

    mesh.export(output_stl)
    print(f"Deformed STL written to {output_stl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_stl")
    parser.add_argument("ffd_json")
    parser.add_argument("output_stl")

    args = parser.parse_args()
    apply_ffd_to_stl(args.input_stl, args.ffd_json, args.output_stl)


if __name__ == "__main__":
    main()
