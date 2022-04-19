import numpy as np
import os
import subprocess
import json
import trimesh
import torch
from datetime import datetime

from isdf.datasets import sdf_util
from isdf.eval import eval_pts


incSDF_dir = "/home/joe/projects/incSDF/"
sdf_fusion_dir = "/home/joe/projects/incSDF/sdf_fusion"


def eval(params):

    res_dir = params["save_dir"]
    eval_pts_dir = params["eval_pts_dir"]

    files = os.listdir(res_dir)
    times = [
        float(f[:-4]) for f in files
        if f[-4:] == ".txt" and len(f.split('.')) == 3
    ]
    times.sort()

    vox_res = {}

    """
    Visualise output
    """

    sdf, transform = sdf_util.read_sdf_gpufusion(
        os.path.join(res_dir, "final_sdf.txt"),
        os.path.join(res_dir, "transform.txt"))
    occ = np.loadtxt(res_dir + "/final_occ.txt")
    occ = occ.reshape(sdf.shape)
    prob = np.loadtxt(res_dir + "/final_prob.txt")
    prob = prob.reshape(sdf.shape)

    from isdf.train.trainer import draw_cams
    mesh_file = params['gt_sdf_root'] + params['gt_sdf_dir'] + "/mesh.obj"
    mesh = trimesh.load(mesh_file)
    poses = np.loadtxt(params['data_dir'] + f"/{seq}/traj.txt")
    poses = poses.reshape(-1, 4, 4)

    # view SDF
    # import ipdb; ipdb.set_trace()
    # from isdf.visualisation.sdf_viewer import SDFViewer
    # scene = trimesh.Scene(mesh)
    # draw_cams(poses.shape[0], poses, scene)
    # SDFViewer(sdf_grid=sdf, grid2world=transform, scene=scene, sdf_range=[-2, 2])

    # # view occupancy from save probabilities
    # prob_tm = trimesh.voxel.VoxelGrid(prob > 0, transform=transform)
    # scene = trimesh.Scene([mesh, prob_tm])
    # draw_cams(poses.shape[0], poses, scene)
    # scene.show()

    # view occupancy from saved occupancy grid
    # Values should all be either 0 or 10000? But there are many in between
    # import ipdb; ipdb.set_trace()
    # occ_tm = trimesh.voxel.VoxelGrid(occ < 1000, transform=transform)
    # scene = trimesh.Scene([mesh, occ_tm])
    # draw_cams(poses.shape[0], poses, scene)
    # scene.show()

    # visualise level set
    from isdf.eval.figs import mesh_vis
    from isdf.eval import plot_utils
    gpuf_sdf_interp = plot_utils.get_gpuf_sdf_interp(
        res_dir, eval_t=None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_pc, scene_scale, bounds_transform = mesh_vis.make_grid_pc(
        mesh, 256, device)
    gpuf_mesh = mesh_vis.get_mesh(
        grid_pc, 256, gpuf_sdf_interp, scene_scale, bounds_transform)
    gpuf_mesh.show()

    gt_sdf_dir = params["gt_sdf_dir"]
    gt_sdf_root = params["gt_sdf_root"]
    gt_sdf_file = gt_sdf_root + gt_sdf_dir + "/1cm/sdf.npy"
    sdf_transf_file = gt_sdf_root + gt_sdf_dir + "/1cm/transform.txt"
    sdf_grid = np.load(gt_sdf_file)
    if params["dataset_format"] == "ScanNet":
        sdf_grid = np.abs(sdf_grid)
    sdf_transform = np.loadtxt(sdf_transf_file)
    gt_sdf_interp = sdf_util.sdf_interpolator(
        sdf_grid, sdf_transform)

    seq_len = np.loadtxt(
        params["data_dir"] + params['seq'] + "/traj.txt").shape[0]

    cached_dataset = eval_pts.get_cache_dataset(
        params["data_dir"] + params['seq'] + "/",
        params["dataset_format"], params["scannet_dir"])

    dirs_C = eval_pts.get_dirs_C(
        params["dataset_format"], params["scannet_dir"])

    for t in times:
        t_str = f"{t:.3f}"
        print(t_str)

        sdf, transform = sdf_util.read_sdf_gpufusion(
            os.path.join(res_dir, t_str + ".txt"),
            os.path.join(res_dir, "transform.txt"))

        sdf_fn = sdf_util.sdf_interpolator(sdf, transform)

        seq_dir = params['data_dir'] + params['seq']
        res = eval_pts.fixed_pts_eval(
            sdf_fn, t, eval_pts_dir, seq_dir, params["dataset_format"],
            cached_dataset, dirs_C, gt_sdf_interp,
            params['eval_pts_root_vol'], seq_len,
        )

        vox_res[t_str] = res

    print("Saving res")
    with open(os.path.join(res_dir, 'vox_res.json'), 'w') as f:
        json.dump(vox_res, f, indent=4)


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

eval_pts_root = incSDF_dir + "data/eval_pts/gt_traj/0.055/"
eval_pts_root_vol = incSDF_dir + "data/eval_pts/"
save_root = incSDF_dir + "res/gpu_fusion/"
data_dir = incSDF_dir + "data/"
gt_sdf_root = incSDF_dir + "data/gt_sdfs/"
scannet_root = "/mnt/sda/ScanNet/scans/"

now = datetime.now()
exp_name = now.strftime("%m-%d-%y_%H-%M-%S")
save_dir = os.path.join(save_root, exp_name)
os.mkdir(save_dir)

noisy_depth = 1
frac_time_perception = 1.0
fps = 30 / frac_time_perception
vsm = 0.055
orb_traj = False

os.chdir(incSDF_dir + "sdf_fusion")

for seq_info in seqs:

    (dataset_format, seq, gt_sdf_dir) = seq_info

    seq_save_dir = os.path.join(save_dir, seq) + "/"
    os.mkdir(seq_save_dir)

    # Load intrinsics
    eval_pts_dir = eval_pts_root + seq + "/eval_pts/"

    if dataset_format == "replicaCAD":
        seq_data_dir = data_dir + "ReplicaCAD-seqs/"
        depth_factor = 3276.75
        fx = 600.0
        fy = 600.0
        cx = 599.5
        cy = 339.5
        w = 1200
        h = 680

    elif dataset_format == "ScanNet":
        seq_data_dir = data_dir + "ScanNet-seqs/"
        depth_factor = 1000.0

        intrinsic_file = scannet_root + seq + \
            "/frames/intrinsic/intrinsic_depth.txt"
        K = np.loadtxt(intrinsic_file)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        w = 640
        h = 480

    orb_traj_str = "false"
    if orb_traj:
        orb_traj_str = "true"

    # Compute scene range
    print("loading mesh...")
    mesh_file = gt_sdf_root + gt_sdf_dir + "/mesh.obj"
    mesh = trimesh.load(mesh_file)
    range_vec = mesh.extents / 0.9
    origin = mesh.bounds[0] - (range_vec - mesh.extents) / 2

    print("Running seq...")

    incomplete = True
    while incomplete:
        subprocess.run([
            "./build/sdf_fusion/Jing06_testSDFFusion",
            "--dataset_format", dataset_format,
            "--seq", seq,
            "--save_dir", seq_save_dir,
            "--data_dir", seq_data_dir,
            "--eval_pts_dir", eval_pts_dir,
            "--noisy_depth", str(noisy_depth),
            "--frac_time_perception", str(frac_time_perception),
            "--fps", str(fps),
            "--vsm", str(vsm),
            "--ox", str(origin[0]),
            "--oy", str(origin[1]),
            "--oz", str(origin[2]),
            "--rx", str(range_vec[0]),
            "--ry", str(range_vec[1]),
            "--rz", str(range_vec[2]),
            "--fx", str(fx),
            "--fy", str(fy),
            "--cx", str(cx),
            "--cy", str(cy),
            "--w", str(w),
            "--h", str(h),
            "--depth_factor", str(depth_factor),
            "--orb_traj", orb_traj_str,
            "--verbose", "false",
        ])

        n_saved = len(os.listdir(seq_save_dir))
        n_save_times = len(os.listdir(eval_pts_dir))
        print("N saved files:", n_saved, " -- should be:", n_save_times + 4)

        if n_saved != n_save_times + 4:
            print("Redo experiment with larger grid")
            origin -= np.array([0.5, 0.5, 0.5])
            range_vec += np.array([1., 1., 1.])
            print("Retrying with new origin and range:")
            print(origin, range_vec, "\n")
        else:
            print("Success! Correct number of saved outputs.")
            incomplete = False

    # Save params file for evaluation
    params = {
        "dataset_format": dataset_format,
        "seq": seq,
        "gt_sdf_root": gt_sdf_root,
        "gt_sdf_dir": gt_sdf_dir,
        "save_dir": seq_save_dir,
        "data_dir": seq_data_dir,
        "eval_pts_dir": eval_pts_dir,
        "eval_pts_root_vol": eval_pts_root_vol,
        "noisy_depth": noisy_depth,
        "fps": fps,
        "frac_time_perception": frac_time_perception,
        "vsm": vsm,
        "ox": origin[0],
        "oy": origin[1],
        "oz": origin[2],
        "rx": range_vec[0],
        "ry": range_vec[1],
        "rz": range_vec[2],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "orb_traj": orb_traj,
        "depth_factor": depth_factor,
        "scannet_dir": scannet_root + "/" + seq + "/",
    }
    with open(seq_save_dir + "params.json", 'w') as f:
        json.dump(params, f, indent=4)

    print("Doing eval...")
    eval_pts_dir = eval_pts_root + seq + "/eval_pts/"

    eval(params)
