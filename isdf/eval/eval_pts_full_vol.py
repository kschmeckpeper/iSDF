# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
import os
import torch
import cv2

from isdf.datasets import sdf_util
from isdf.geometry import frustum, transform

"""
Generate the points for the full volume evaluation.
"""


def replicaCAD_pts(samples):
    """
        Discard points with stage SDF less than zero. i.e. outside of room.
        Discard points in cupboard that is never navigable.
    """
    n = int(samples * 1.5)

    root = "/home/joe/projects/incSDF/data/gt_sdfs/apt_2/1cm/"
    gt_sdf_file = root + "sdf.npy"
    sdf_transf_file = root + "transform.txt"
    stage_sdf_file = root + "stage_sdf.npy"

    seq_dir = "/home/joe/projects/incSDF/data/ReplicaCAD-seqs/apt_2_nav/"

    sdf_grid = np.load(gt_sdf_file)
    sdf_transform = np.loadtxt(sdf_transf_file)
    sdf_dims = np.array(sdf_grid.shape)

    eval_pts = np.random.rand(n, 3)
    eval_pts = eval_pts * (sdf_dims - 1)
    eval_pts = eval_pts * sdf_transform[0, 0]
    eval_pts = eval_pts + sdf_transform[:3, 3]

    stage_sdf = np.load(stage_sdf_file)
    transf = np.loadtxt(sdf_transf_file)
    stage_sdf_interp = sdf_util.sdf_interpolator(stage_sdf, transf)

    eval_stage_sdf = stage_sdf_interp(eval_pts)

    # discard_pts = eval_pts[eval_stage_sdf <= 0]
    eval_pts = eval_pts[eval_stage_sdf > 0]

    min_xy = np.loadtxt(seq_dir + 'bounds.txt')
    islands = np.loadtxt(seq_dir + 'unnavigable.txt')
    px = np.floor((eval_pts[:, 0] - min_xy[0]) / min_xy[2])
    py = np.floor((eval_pts[:, 2] - min_xy[1]) / min_xy[2])
    px = np.clip(px, 0, islands.shape[1] - 1).astype(int)
    py = np.clip(py, 0, islands.shape[0] - 1).astype(int)

    # discard2_pts = eval_pts[islands[py, px] == 1]
    eval_pts = eval_pts[islands[py, px] == 0]

    eval_pts = eval_pts[:samples]

    # Vis evaluation points
    # scene_file = "/home/joe/projects/incSDF/data/gt_sdfs/apt_2/mesh.obj"
    # mesh_gt = trimesh.load(scene_file)
    # scene = trimesh.Scene(mesh_gt)
    # pc = trimesh.PointCloud(eval_pts, [0, 255, 0, 255])
    # pc1 = trimesh.PointCloud(discard_pts, [255, 0, 0, 255])
    # pc2 = trimesh.PointCloud(discard2_pts, [255, 0, 0, 255])
    # scene.add_geometry([pc])
    # scene.show()

    return eval_pts


def scanNet_pts(samples, seq):

    # Generate many random points in mesh bounds

    mesh_file = "/home/joe/projects/incSDF/data/gt_sdfs/" + seq + "/mesh.obj"
    mesh = trimesh.load(mesh_file)

    transf, extents = trimesh.bounds.oriented_bounds(mesh)
    transf = np.linalg.inv(transf)

    corner = transf[:3, 3] - transf[:3, :3] @ extents / 2

    n = int(samples * 4)

    offsets = np.random.rand(n, 3)
    pts = np.full([n, 3], corner)
    pts += offsets[:, 0][:, None] * transf[:3, 0] * extents[0]
    pts += offsets[:, 1][:, None] * transf[:3, 1] * extents[1]
    pts += offsets[:, 2][:, None] * transf[:3, 2] * extents[2]

    # Prune points

    device = "cuda" if torch.cuda.is_available() else "cpu"

    traj_file = "/home/joe/projects/incSDF/data/ScanNet-seqs/" + \
        seq + "/traj.txt"
    traj = np.loadtxt(traj_file).reshape(-1, 4, 4)

    H = 480
    W = 640
    scannet_dir = "/home/joe/projects/incSDF/data/ScanNet/scans/" + seq
    intrinsic_file = scannet_dir + "/frames/intrinsic/intrinsic_depth.txt"
    K = np.loadtxt(intrinsic_file)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    depth_dir = os.path.join(scannet_dir, "frames", "depth/")
    depth_batch = []
    for i in range(len(traj)):
        depth_file = depth_dir + str(i) + ".png"
        depth = cv2.imread(depth_file, -1)
        depth = depth.astype(float) / 1000.0
        depth_batch.append(depth)

    traj = torch.FloatTensor(traj).to(device)
    pts_torch = torch.FloatTensor(pts).to(device)
    depth_batch = torch.FloatTensor(depth_batch).to(device)

    bsize = 10000
    batches = int(np.ceil(pts.shape[0] / bsize))

    with torch.no_grad():
        visible = []
        for b in range(batches):
            pts_batch = pts_torch[b * bsize: (b + 1) * bsize]

            visible_b = frustum.is_visible_torch(
                pts_batch, traj, depth_batch,
                H, W, fx, fy, cx, cy,
                trunc=0., use_projection=True)
            visible_b = visible_b.sum(dim=0) > 0
            visible.append(visible_b)
    visible = torch.cat(visible)

    keep_pts = pts[visible.cpu().numpy()]
    discard_pts = pts[~visible.cpu().numpy()]

    # scene = trimesh.Scene()
    # for j, depth in enumerate(depth_batch):
    #     pcd = transform.pointcloud_from_depth(
    #         depth.cpu().numpy(), fx, fy, cx, cy)
    #     pcd = pcd[::40, ::40]
    #     geom = trimesh.PointCloud(vertices=pcd.reshape(-1, 3))
    #     scene.add_geometry(geom, transform=traj[j].cpu().numpy())

    # box = trimesh.primitives.Box(transform=transf, extents=extents)
    # scene = trimesh.Scene(
    #     [mesh,
    #      trimesh.PointCloud(keep_pts, [0, 255, 0, 255]),
    #      trimesh.PointCloud(discard_pts, [255, 0, 0, 255]),
    #      box])

    eval_pts = keep_pts[:samples]

    return eval_pts


def get_gt_sdf(gt_sdf, pts, dataset_format):
    gt_sdf_dir = "/home/joe/projects/incSDF/data/gt_sdfs/" + gt_sdf
    sdf_grid = np.load(gt_sdf_dir + "/1cm/sdf.npy")

    sdf_transform = np.loadtxt(gt_sdf_dir + "/1cm/transform.txt")
    gt_sdf_interp = sdf_util.sdf_interpolator(sdf_grid, sdf_transform)

    if dataset_format == "ScanNet":
        sdf_grid = np.abs(sdf_grid)
        gt_sdf = sdf_util.eval_sdf_interp(
            gt_sdf_interp, pts,
            handle_oob='fill', oob_val=np.nan)

        valid = ~np.isnan(gt_sdf)
        gt_sdf = gt_sdf[valid]
        pts = pts[valid]

        return gt_sdf, pts

    else:
        gt_sdf = gt_sdf_interp(pts)
        return gt_sdf


if __name__ == "__main__":

    samples = 200000

    # # replicaCAD

    # replicaCAD_seqs = ['apt_2_mnp', 'apt_2_nav', 'apt_2_obj',
    #                    'apt_3_mnp', 'apt_3_nav', 'apt_3_obj']
    # gt_sdfs = ['apt_2_v1', 'apt_2', 'apt_2', 'apt_3_v1', 'apt_3', 'apt_3']

    # pts = replicaCAD_pts(samples)
    # print(pts.shape)

    # save_file = "/home/joe/projects/incSDF/data/eval_pts/full_vol/replicaCAD.npy"
    # np.save(save_file, pts)

    # for i, seq in enumerate(replicaCAD_seqs):

    #     print(seq)

    #     gt_sdf = get_gt_sdf(gt_sdfs[i], pts, "replicaCAD")

    #     save_f = f"/home/joe/projects/incSDF/data/eval_pts/full_vol/gt_{seq}.npy"
    #     np.save(save_f, gt_sdf)


    # ScanNet

    scanNet_seqs = ['scene0010_00', 'scene0030_00', 'scene0031_00',
                    'scene0004_00', 'scene0005_00', 'scene0009_00']
    for seq in scanNet_seqs:

        print(seq)

        pts = scanNet_pts(samples, seq)

        gt_sdf, pts = get_gt_sdf(seq, pts, "ScanNet")

        save_file = f"/home/joe/projects/incSDF/data/eval_pts/full_vol/{seq}.npy"
        np.save(save_file, pts)
        print(pts.shape)

        save_f = f"/home/joe/projects/incSDF/data/eval_pts/full_vol/gt_{seq}.npy"
        np.save(save_f, gt_sdf)
