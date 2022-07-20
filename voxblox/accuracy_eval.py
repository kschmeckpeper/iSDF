import numpy as np
import os
import subprocess
from datetime import datetime

import vxblx_utils


def gen_params_files():

    # os.makedirs(save_dir)

    params_list = []
    for voxel_size in voxel_sizes:
        nested_save_dir = os.path.join(save_dir, str(voxel_size))
        # os.mkdir(nested_save_dir)

        for i in range(len(seqs)):

            (dataset_format, seq, gt_sdf_dir) = seqs[i]

            if dataset_format == "replicaCAD":
                seq_dir = seq_root + "ReplicaCAD-seqs/" + seq + "/"
            elif dataset_format == "ScanNet":
                seq_dir = seq_root + "ScanNet-seqs/" + seq + "/"
            gt_mesh_file = gt_sdf_root + gt_sdf_dir + "/mesh.obj"
            gt_sdf_dir = gt_sdf_root + gt_sdf_dir + "/1cm/"

            params = base_params.copy()

            params["save_dir"] = os.path.join(
                nested_save_dir, seq + "/")

            params["voxel_size"] = voxel_size

            params["seq"] = seq_dir
            params["gt_mesh_file"] = gt_mesh_file
            params["gt_sdf_dir"] = gt_sdf_dir
            params["dataset_format"] = dataset_format
            params["scannet_dir"] = scannet_dir + seq + "/"

            params_list.append(params)

    return params_list


if __name__ == "__main__":

    # Experiment settings
    noisy_depth = 1
    orb_traj = False
    voxel_sizes = [0.055, 0.063, 0.078, 0.11]

    # Fixed settings
    save_interval = 0.5
    update_esdf_every_n_sec = 0.1

    root_save = "/home/joe/projects/incSDF/res/voxblox/"
    now = datetime.now()
    exp_name = now.strftime("%m-%d-%y_%H-%M-%S")
    save_dir = os.path.join(root_save, exp_name)

    base_params = {
        "save_dir": save_dir,
        "orb_traj": orb_traj,

        "dataset_format": None,
        "seq": None,
        "gt_sdf_dir": None,

        "voxel_size": None,
        "fps": 30.0,
        "save_interval": save_interval,
        "update_esdf_every_n_sec": update_esdf_every_n_sec,
        "im_indices": None,
        "noisy_depth": noisy_depth,
    }

    seqs = [
        # (dataset_format, seq_name, gt_sdf_dir)

        # ReplicaCAD sequences
        ("replicaCAD", "apt_2_nav", "apt_2"),
        ("replicaCAD", "apt_2_obj", "apt_2"),
        ("replicaCAD", "apt_2_mnp", "apt_2_v1"),
        ("replicaCAD", "apt_3_nav", "apt_3"),
        ("replicaCAD", "apt_3_obj", "apt_3"),
        ("replicaCAD", "apt_3_mnp", "apt_3_v1"),

        # ScanNet longer sequences
        ("ScanNet", "scene0010_00", "scene0010_00"),
        ("ScanNet", "scene0030_00", "scene0030_00"),
        ("ScanNet", "scene0031_00", "scene0031_00"),

        # ScanNet shorter sequences
        ("ScanNet", "scene0004_00", "scene0004_00"),
        ("ScanNet", "scene0005_00", "scene0005_00"),
        ("ScanNet", "scene0009_00", "scene0009_00"),
    ]

    seq_root = "/home/joe/projects/incSDF/data/"
    gt_sdf_root = "/home/joe/projects/incSDF/data/gt_sdfs/"
    scannet_dir = "/mnt/sda/ScanNet/scans/"

    params_list = gen_params_files()

    start = 0
    end = None

    print("Number of experiments ---------------------------------------->",
          len(params_list[start:end]))

    # check traj files
    for params in params_list:
        T = np.loadtxt(params['seq'] + "traj.txt").reshape(-1, 4, 4)[:, 0, 0]
        assert np.isinf(T).sum() == 0
        assert np.isnan(T).sum() == 0

    # Run experiments -----------------------------------------------

    for i, params in enumerate(params_list[start:end]):
        print(params)

        # Create launchfile and run ros
        vxblx_utils.run_exp(params)

        print("\n\nDone experiment!")

    # Do evaluations -------------------------------------------

    for i, params in enumerate(params_list[start:end]):
        print(params)

        params_file = os.path.join(params['save_dir'], "params.json")

        subprocess.run([
            "python", "voxblox/run_eval.py",
            "--params_file", params_file,
        ])
