import numpy as np
import trimesh
import torch
import os
import json
import cv2

from isdf.modules import trainer, geometry, fc_map
from isdf.datasets import sdf_util, replicaCAD_gt_sdf
from isdf.eval import plot_utils, metrics
from isdf.visualisation import draw3D


def make_grid_pc(gt_mesh, grid_dim, device):

    inv_bounds_transform, extents = trimesh.bounds.oriented_bounds(gt_mesh)
    bounds_transform = np.linalg.inv(inv_bounds_transform)
    scene_scale = extents / (2 * 0.9)
    grid_pc = geometry.transform.make_3D_grid(
        grid_range=[-1.0, 1.0],
        dim=grid_dim,
        device=device,
        transform=torch.FloatTensor(bounds_transform).to(device),
        scale=torch.FloatTensor(scene_scale).to(device),
    )
    grid_pc = grid_pc.view(-1, 3).to(device)

    return grid_pc, scene_scale, bounds_transform


def iSDF_mesh(grid_pc, grid_dim, sdf_map, scene_scale, bounds_transform):
    with torch.set_grad_enabled(False):
        chunk_size = 200000
        sdf = fc_map.chunks(grid_pc, chunk_size, sdf_map)
        sdf = sdf.view(grid_dim, grid_dim, grid_dim)

    sdf_mesh = draw3D.draw_mesh(sdf, scene_scale, bounds_transform)

    return sdf_mesh


def get_mesh(grid_pc, grid_dim, sdf_interp, scene_scale, bounds_transform):
    sdf = sdf_util.eval_sdf_interp(
        sdf_interp, grid_pc.cpu().numpy(), handle_oob='fill', oob_val=np.nan)
    sdf = sdf.reshape(grid_dim, grid_dim, grid_dim)

    print("nans", np.isnan(sdf).sum(), "out of", np.prod(sdf.shape))

    sdf_mesh = draw3D.draw_mesh(sdf, scene_scale, bounds_transform)
    sdf_mesh = clean_mesh(sdf_mesh)

    return sdf_mesh


def clean_mesh(mesh):
    values, counts = np.unique(mesh.faces, return_counts=True)
    vertex = values[np.argmax(counts)]
    keep_faces = (mesh.faces == vertex).sum(axis=1) == 0
    mesh.faces = mesh.faces[keep_faces]
    mesh.visual.face_colors = [160, 160, 160, 255]

    return mesh


def load_undistorted_mesh(dataset_path, gt_sdf_dir):
    scene_config = \
        dataset_path + f"configs/scenes/{gt_sdf_dir}.scene_instance.json"
    joint_cfg = {}
    if gt_sdf_dir == "apt_2_v1":
        joint_cfg = {"fridge": {"top_door_hinge": np.pi / 2.}}
    elif gt_sdf_dir == "apt_3_v1":
        joint_cfg = {"kitchen_counter": {"middle_slide_top": 0.38}}

    scene = replicaCAD_gt_sdf.load_replicaCAD(
        scene_config, dataset_path, joint_cfg=joint_cfg, verbose=False)

    meshes = scene.dump()
    out_scene = trimesh.Scene(meshes)

    return out_scene


def process_slices(sdf, pc_shape):

    img = cmap.to_rgba(sdf.flatten(), alpha=1., bytes=False)
    img = img.reshape(*pc_shape, 4)
    img = (img * 255).astype(np.uint8)[..., :3][..., ::-1]

    gaps = img.sum(axis=-1) == 0
    img[gaps] = [255, 255, 255]

    slices = [
        np.flip(np.flip(img[:, 0], axis=1), axis=0),
        np.flip(np.flip(img[:, 1], axis=1), axis=0)
    ]
    pad_shape = np.array(slices[0].shape)
    pad_shape[1] = 8
    pad = np.full(pad_shape, 255).astype(np.uint8)

    slices = np.hstack((slices[0], pad, slices[1]))

    return slices


def draw_poses(scene, seq, isdf_dir, seq_root, dataset_format):
    with open(isdf_dir + seq + "_0/res.json", 'r') as f:
        kf_indices = json.load(f)["kf_indices"]

    seq_dir = "ReplicaCAD-seqs" if dataset_format == "replicaCAD" else "ScanNet-seqs"
    traj_file = seq_root + seq_dir + "/" + seq + "/traj.txt"
    traj = np.loadtxt(traj_file).reshape(-1, 4, 4)
    kf_poses = traj[kf_indices]

    draw3D.draw_trajectory(
        traj[:, :3, 3], scene, color=(1.0, 0.0, 0.0))
    trainer.draw_cams(
        len(kf_poses), kf_poses, scene, color=(0.0, 1.0, 0.0, 0.8))


def write_ply(mesh, filename):
    data = trimesh.exchange.ply.export_ply(mesh)
    out = open(filename, "wb+")
    out.write(data)
    out.close()


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    isdf_dir = "/home/joe/projects/incSDF/incSDF/res/iSDF/models/"
    gpuf_dir = "/home/joe/projects/incSDF/incSDF/res/gpu_fusion/7cm_unocc/"
    voxblox_dir = "/home/joe/projects/incSDF/incSDF/res/voxblox/gt_traj/0.055/"

    gt_sdf_path = "/home/joe/projects/incSDF/incSDF/data/gt_sdfs/"
    replicaCAD_path = "/mnt/sda/ReplicaCAD/replica_cad/"
    seq_root = "/home/joe/projects/incSDF/incSDF/data/"

    save_dir = "/home/joe/projects/incSDF/incSDF/res/figs/meshes/"

    replicaCAD_topdown_pose = np.array(
        [[2.52673972e-03, -9.93030930e-01, 1.17826943e-01, 2.45915858e+00],
         [-2.79531494e-03, 1.17819845e-01, 9.93031052e-01, 1.41784008e+01],
         [-9.99992901e-01, -2.83849442e-03, -2.47813411e-03, 1.63345018e+00],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    )
    replicaCAD_slant_pose = np.array(
    [[ 1.11639533e-02, -8.09075685e-01,  5.87598418e-01,  9.30019048e+00],
     [ 1.39525274e-02,  5.87703873e-01,  8.08955799e-01,  1.29741147e+01],
     [-9.99840334e-01, -8.32661731e-04,  1.78497493e-02,  1.91871071e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    )
    scene0010_view_pose = np.array(
    [[-0.68511851,  0.72717809,  0.042716  ,  3.32456965],
     [-0.72782697, -0.68098297, -0.08080899,  2.50794434],
     [-0.02967366, -0.0864536 ,  0.99581386, 10.50600524],
     [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    scene0030_view_pose = np.array(
    [[ 9.88937077e-01, -1.47752544e-01,  1.31394212e-02,  3.89325491e+00],
     [-1.48333960e-01, -9.84618523e-01,  9.23222658e-02,  4.11096763e+00],
     [-7.03532152e-04, -9.32499340e-02, -9.95642483e-01, -8.93398376e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    )
    scene0031_view_pose = np.array(
    [[ 0.99034522, -0.07299859 , 0.11784548,  5.67616302],
     [ 0.08331299,  0.99288245, -0.08510808,  1.87019931],
     [-0.11079393,  0.09410444,  0.98937812, 13.2954802 ],
     [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    scene0004_view_pose = np.array(
    [[-0.3227421 ,  0.91281283,  0.25022044,  7.8787366 ],
     [-0.94443087, -0.2931674 , -0.14867147,  2.86895261],
     [-0.06235276, -0.28429845,  0.95670608, 15.35848871],
     [ 0.        ,  0.        ,  0.         , 1.        ]]
    )
    scene0005_view_pose = np.array(
    [[ 9.41919344e-01,  3.34364698e-01, -3.14356153e-02,  2.58756538e+00],
     [-3.34547330e-01,  9.42378767e-01, -5.85642321e-04,  2.50897200e+00],
     [ 2.94284383e-02,  1.10683290e-02,  9.99505607e-01,  1.01325643e+01],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    )
    scene0009_view_pose = np.array(
    [[ 1.91406085e-01,  9.81450210e-01, -1.09176604e-02,  2.74733832e+00],
     [-9.81113557e-01,  1.91633244e-01,  2.63227706e-02,  5.03179749e+00],
     [ 2.79266754e-02,  5.67312613e-03,  9.99593876e-01,  1.40309918e+01],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    )

    scene0031_slanted_pose = np.array(
    [[ 0.99892633, -0.01247383,  0.04461594,  4.8650144 ],
     [ 0.04168092,  0.66235458, -0.74803016, -4.96430571],
     [-0.02022077,  0.74908666,  0.66216335,  8.88041906],
     [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    scene0010_slanted_pose = np.array(
    [[-0.71765794,  0.54363472, -0.4352337 , -0.8679222 ],
     [-0.69497237, -0.5191489 ,  0.49749153,  7.64451141],
     [ 0.04450257,  0.65950414,  0.75038244,  7.88999738],
     [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    scene0031_slanted_pose = np.array(
    [[ 0.99892633, -0.01247383,  0.04461594,  4.8650144 ],
        [ 0.04168092,  0.66235458, -0.74803016, -4.96430571],
        [-0.02022077,  0.74908666,  0.66216335,  8.88041906],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    scene0010_slanted_pose = np.array(
    [[-0.71765794,  0.54363472, -0.4352337 , -0.8679222 ],
        [-0.69497237, -0.5191489 ,  0.49749153,  7.64451141],
        [ 0.04450257,  0.65950414,  0.75038244,  7.88999738],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )

    seqs = [
        # (dataset_format, seq_name, gt_sdf_dir)

        # ReplicaCAD sequences
        ("replicaCAD", "apt_2_nav", "apt_2", replicaCAD_slant_pose),
        ("replicaCAD", "apt_2_obj", "apt_2", replicaCAD_slant_pose),
        ("replicaCAD", "apt_2_mnp", "apt_2_v1", replicaCAD_slant_pose),
        ("replicaCAD", "apt_3_nav", "apt_3", replicaCAD_slant_pose),
        ("replicaCAD", "apt_3_obj", "apt_3", replicaCAD_slant_pose),
        ("replicaCAD", "apt_3_mnp", "apt_3_v1", replicaCAD_slant_pose),

        # ScanNet longer sequences
        ("ScanNet", "scene0010_00", "scene0010_00", scene0010_slanted_pose),
        ("ScanNet", "scene0030_00", "scene0030_00", scene0030_view_pose),
        ("ScanNet", "scene0031_00", "scene0031_00", scene0031_slanted_pose),

        # ScanNet shorter sequences
        ("ScanNet", "scene0004_00", "scene0004_00", scene0004_view_pose),
        ("ScanNet", "scene0005_00", "scene0005_00", scene0005_view_pose),
        ("ScanNet", "scene0009_00", "scene0009_00", scene0009_view_pose),
    ]


    # """
    # Save meshes for video
    # """
    # dataset_format, seq, gt_dir, view_pose = seqs[8]

    # gt_mesh = trimesh.load(f"{gt_sdf_path}/{gt_dir}/mesh.obj")

    # eval_t = 


    # gpuf_sdf_interp = plot_utils.get_gpuf_sdf_interp(
    #     gpuf_dir + seq, eval_t=None)
    # vox_sdf_interp = plot_utils.get_voxblox_sdf_interp(
    #     voxblox_dir + seq, gt_mesh, eval_t=None)


    """
    Vis level sets
    """

    grid_dim = 512

    for i in range(1, 2):
        dataset_format, seq, gt_dir, view_pose = seqs[i]
        print(seq)

        scene = trimesh.Scene()

        gt_mesh = trimesh.load(f"{gt_sdf_path}/{gt_dir}/mesh.obj")

        # iSDF
        load_file = f"{isdf_dir}/{seq}_0/model.pth"
        sdf_map = plot_utils.load_model(load_file, gt_mesh, device)

        # gpu fusion
        # gpuf_sdf_interp = plot_utils.get_gpuf_sdf_interp(
        #     gpuf_dir + seq, eval_t=None)

        # voxblox
        vox_sdf_interp = plot_utils.get_voxblox_sdf_interp(
            voxblox_dir + seq, gt_mesh, eval_t=None)

        grid_pc, scene_scale, bounds_transform = make_grid_pc(
            gt_mesh, grid_dim, device)



        # Full scene level sets -------------------------------------

        # if dataset_format == "replicaCAD":
        #     gt_mesh = load_undistorted_mesh(replicaCAD_path, gt_dir)

        isdf_mesh = iSDF_mesh(
            grid_pc, grid_dim, sdf_map, scene_scale, bounds_transform)
        voxblox_mesh = get_mesh(
            grid_pc, grid_dim, vox_sdf_interp, scene_scale, bounds_transform)
        # gpuf_mesh = get_mesh(
        #     grid_pc, grid_dim, gpuf_sdf_interp, scene_scale, bounds_transform)

        # meshes = [
        #     ("isdf", iSDF_mesh),
        #     ("vox", voxblox_mesh),
        #     ("gpuf", gpuf_mesh),
        #     ("gt", gt_mesh),
        # ]

        write_ply(isdf_mesh, f"/home/joe/projects/incSDF/incSDF/res/video_mats/meshes/isdf_512_{seq}.ply")

        # for data in meshes:
        #     name, mesh = data

        #     if name != "gt":
        #         write_ply(mesh, save_dir + f"ply_files/{seq}_{name}.ply")

        #     cap_scene = trimesh.Scene(mesh)
        #     if name == "gt":
        #         draw_poses(cap_scene, seq, isdf_dir, seq_root, dataset_format)

        #     image = draw3D.capture_scene_im(cap_scene, view_pose, tm_pose=True)
        #     cv2.imwrite(
        #         save_dir + f"level_sets/{seq}_{name}.png",
        #         image[..., :3][..., ::-1])

        # Objects ---------------------------------------------------

        # shaker_view = np.array(
        # [[-0.34211277, -0.48862875,  0.8026212 , -1.06511032],
        #  [ 0.01703154,  0.85079748,  0.52521764,  1.59867992],
        #  [-0.93950454,  0.19335354, -0.28274658,  2.89085729],
        #  [ 0.        ,  0.        ,  0.        ,  1.        ]]
        # )

        # obj_bounds_file = f"{seq_root}/ReplicaCAD-seqs/{seq}/obj_bounds.txt"

        # if os.path.exists(obj_bounds_file):

        #     obj_bounds = metrics.get_obj_eval_bounds(
        #         obj_bounds_file, 1, expand_m=1.0, expand_down=True)
        #     tight_obj_bounds = metrics.get_obj_eval_bounds(
        #         obj_bounds_file, 1, expand_m=0.2, expand_down=True)

        #     for i, bounds in enumerate(obj_bounds):

        #         scene_obj = trimesh.Scene()

        #         T = np.eye(4)
        #         extents = bounds[1] - bounds[0]
        #         T[:3, 3] = bounds[0] + 0.5 * extents
        #         scene_scale = 0.5 * (bounds[1] - bounds[0])

        #         dim = 256
        #         x = torch.linspace(bounds[0, 0], bounds[1, 0], dim)
        #         y = torch.linspace(bounds[0, 1], bounds[1, 1], dim)
        #         z = torch.linspace(bounds[0, 2], bounds[1, 2], dim)
        #         xx, yy, zz = torch.meshgrid(x, y, z)
        #         pc = torch.cat(
        #             (xx[..., None], yy[..., None], zz[..., None]), dim=3)
        #         pc = pc.view(-1, 3).to(device)

        #         offset = np.array([1.5 * (bounds[1, 0] - bounds[0, 0]), 0, 0])

        #         box = trimesh.primitives.Box(extents=extents, transform=T)
        #         gt_obj = gt_mesh.slice_plane(box.facets_origin, -box.facets_normal)

        #         gt_mesh_show = load_undistorted_mesh(replicaCAD_path, gt_dir)

        #         isdf_mesh = iSDF_mesh(pc, dim, sdf_map, scene_scale, T)
        #         # voxblox_mesh = get_mesh(pc, dim, vox_sdf_interp, scene_scale, T)
        #         # gpuf_mesh = get_mesh(pc, dim, gpuf_sdf_interp, scene_scale, T)

        #         # meshes = [
        #         #     ("isdf", isdf_mesh),
        #         #     ("vox", voxblox_mesh),
        #         #     ("gpuf", gpuf_mesh),
        #         #     ("gt", gt_mesh_show),
        #         # ]

        #         # for data in meshes:
        #         #     name, mesh = data
        #         #     cap_scene = trimesh.Scene(mesh)
        #         #     image = draw3D.capture_scene_im(cap_scene, shaker_view, tm_pose=True)
        #         #     cv2.imwrite(
        #         #         save_dir + f"obj_closeups/{seq}_{name}.png",
        #         #         image[..., :3][..., ::-1])

        #         # scene_obj.add_geometry(gt_obj)
        #         # scene_obj.add_geometry(box)

        #         # isdf_mesh.apply_translation(offset)
        #         # scene_obj.add_geometry(isdf_mesh)
        #         # box_isdf = trimesh.primitives.Box(extents=extents, transform=T)
        #         # box_isdf.apply_translation(offset)
        #         # scene_obj.add_geometry(box_isdf)

        #         # voxblox_mesh.apply_translation(offset * 2)
        #         # scene_obj.add_geometry(voxblox_mesh)
        #         # box_vox = trimesh.primitives.Box(extents=extents, transform=T)
        #         # box_vox.apply_translation(offset * 2)
        #         # scene_obj.add_geometry(box_vox)

        #         # gpu fusion mesh
        #         # gpuf_mesh.apply_translation(offset * 3)
        #         # scene_obj.add_geometry(gpuf_mesh)
        #         # box_gpuf = trimesh.primitives.Box(extents=extents, transform=T)
        #         # box_gpuf.apply_translation(offset * 3)
        #         # scene_obj.add_geometry(box_gpuf)

        #         # scene_obj.show()

        #         isdf_chkp_file = "/home/joe/projects/incSDF/incSDF/isdf/" + \
        #             "train/examples/experiments/apt_2_mnp_obj_vis/" + \
        #             "checkpoints/t_2000.pth"
        #         # files = os.listdir(isdf_dir)
        #         # steps = [int(f.split('_')[1].split('.')[0]) for f in files]
        #         # ixs = np.argsort(steps)
        #         # files = np.array(files)[ixs]
        #         # files = [isdf_dir + f for f in files]

        #         # for f in files[-1:]:
        #         #     print(f)
        #         sdf_map_obj = plot_utils.load_model(isdf_chkp_file, gt_mesh)
        #         # isdf_mesh = iSDF_mesh(pc, dim, sdf_map_obj, scene_scale, T)
        #         # cap_scene = trimesh.Scene(isdf_mesh)
        #         # image = draw3D.capture_scene_im(cap_scene, shaker_view, tm_pose=True)
        #         # cv2.imwrite(
        #         #     save_dir + f"obj_closeups/{seq}_isdf.png",
        #         #     image[..., :3][..., ::-1])

        #         # save slices --------------------------

        #         obj_center = np.mean(bounds, axis=0)
        #         bs = tight_obj_bounds[i]
        #         x = torch.linspace(bs[0, 0], bs[1, 0], dim)
        #         y = torch.tensor([1.03, 1.1])
        #         z = torch.linspace(bs[0, 2], bs[1, 2], dim)

        #         xx, yy, zz = torch.meshgrid(x, y, z)
        #         pc = torch.cat(
        #             (xx[..., None], yy[..., None], zz[..., None]), dim=3)
        #         pc_shape = pc.shape[:-1]
        #         pc = pc.view(-1, 3)

        #         # trimesh.Scene(
        #         #    [isdf_mesh, trimesh.PointCloud(pc.cpu())]).show()

        #         import ipdb; ipdb.set_trace()

        #         isdf = sdf_map_obj(pc.to(device)).detach().cpu().numpy()

        #         gt_sdf_dir = gt_sdf_path + gt_dir + "/1cm/"
        #         gt_sdf_interp, sdf_dims, sdf_transform = plot_utils.load_gt_sdf(gt_sdf_dir)
        #         gt_sdf = sdf_util.eval_sdf_interp(
        #             gt_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=0.0)

        #         vox_sdf = sdf_util.eval_sdf_interp(
        #             vox_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=np.nan)

        #         gpuf_sdf = sdf_util.eval_sdf_interp(
        #             gpuf_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=np.nan)

        #         cmap = sdf_util.get_colormap(sdf_range=[-0.2, 0.3])

        #         gt_sdf = process_slices(gt_sdf, pc_shape)
        #         isdf = process_slices(isdf, pc_shape)
        #         vox_sdf = process_slices(vox_sdf, pc_shape)
        #         gpuf_sdf = process_slices(gpuf_sdf, pc_shape)



        #         pad_shape = np.array(isdf.shape)
        #         pad_shape[0] = 8
        #         pad = np.full(pad_shape, 255).astype(np.uint8)
        #         vis = np.vstack((gt_sdf, pad, isdf, pad, vox_sdf, pad, gpuf_sdf))

        #         cv2.imwrite(
        #             save_dir + f"obj_closeups/apt_2_mnp_slices.png", vis)
        #         import ipdb; ipdb.set_trace()


    """
    Object close up figure
    """

    shaker_view = np.array(
    [[-0.34211277, -0.48862875,  0.8026212 , -1.06511032],
     [ 0.01703154,  0.85079748,  0.52521764,  1.59867992],
     [-0.93950454,  0.19335354, -0.28274658,  2.89085729],
     [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )

    gpuf_dir = "/home/joe/projects/incSDF/res/figs/meshes/obj_closeups/gpuf_closeup/"

    isdf_chkp_file = "/home/joe/projects/incSDF/incSDF/isdf/" + \
        "train/examples/experiments/apt_2_mnp_obj_vis/" + \
        "checkpoints/t_2000.pth"
    grid_dim = 256

    dataset_format, seq, gt_dir, view_pose = seqs[2]
    print(seq)

    gt_mesh = trimesh.load(f"{gt_sdf_path}/{gt_dir}/mesh.obj")

    gpuf_sdf_interp = plot_utils.get_gpuf_sdf_interp(
        gpuf_dir + seq, eval_t=None)

    vox_sdf_interp = plot_utils.get_voxblox_sdf_interp(
        voxblox_dir + seq, gt_mesh, eval_t=None)


    obj_bounds_file = f"{seq_root}/ReplicaCAD-seqs/{seq}/obj_bounds.txt"
    bounds = metrics.get_obj_eval_bounds(
        obj_bounds_file, 1, expand_m=1.0, expand_down=True)[0]
    bs = metrics.get_obj_eval_bounds(
        obj_bounds_file, 1, expand_m=0.2, expand_down=True)[0]


    T = np.eye(4)
    extents = bounds[1] - bounds[0]
    T[:3, 3] = bounds[0] + 0.5 * extents
    scene_scale = 0.5 * (bounds[1] - bounds[0])

    dim = 256
    x = torch.linspace(bounds[0, 0], bounds[1, 0], dim)
    y = torch.linspace(bounds[0, 1], bounds[1, 1], dim)
    z = torch.linspace(bounds[0, 2], bounds[1, 2], dim)
    xx, yy, zz = torch.meshgrid(x, y, z)
    pc = torch.cat(
        (xx[..., None], yy[..., None], zz[..., None]), dim=3)
    pc = pc.view(-1, 3).to(device)

    sdf_map = plot_utils.load_model(isdf_chkp_file, gt_mesh)

    gt_mesh_show = load_undistorted_mesh(replicaCAD_path, gt_dir)
    isdf_mesh = iSDF_mesh(pc, dim, sdf_map, scene_scale, T)
    voxblox_mesh = get_mesh(pc, dim, vox_sdf_interp, scene_scale, T)
    gpuf_mesh = get_mesh(pc, dim, gpuf_sdf_interp, scene_scale, T)

    meshes = [
        ("isdf", isdf_mesh),
        ("vox", voxblox_mesh),
        ("gpuf", gpuf_mesh),
        ("gt", gt_mesh_show),
    ]

    for data in meshes:
        name, mesh = data
        cap_scene = trimesh.Scene(mesh)
        image = draw3D.capture_scene_im(cap_scene, shaker_view, tm_pose=True)
        cv2.imwrite(
            save_dir + f"obj_closeups/{seq}_{name}.png",
            image[..., :3][..., ::-1])

    # save slices --------------------------

    obj_center = np.mean(bounds, axis=0)
    x = torch.linspace(bs[0, 0], bs[1, 0], dim)
    y = torch.tensor([1.03, 1.1])
    z = torch.linspace(bs[0, 2], bs[1, 2], dim)

    xx, yy, zz = torch.meshgrid(x, y, z)
    pc = torch.cat(
        (xx[..., None], yy[..., None], zz[..., None]), dim=3)
    pc_shape = pc.shape[:-1]
    pc = pc.view(-1, 3)

    # trimesh.Scene(
    #    [isdf_mesh, trimesh.PointCloud(pc.cpu())]).show()

    isdf = sdf_map(pc.to(device)).detach().cpu().numpy()

    gt_sdf_dir = gt_sdf_path + gt_dir + "/1cm/"
    gt_sdf_interp, sdf_dims, sdf_transform = plot_utils.load_gt_sdf(gt_sdf_dir)
    gt_sdf = sdf_util.eval_sdf_interp(
        gt_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=0.0)

    vox_sdf = sdf_util.eval_sdf_interp(
        vox_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=np.nan)

    gpuf_sdf = sdf_util.eval_sdf_interp(
        gpuf_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=np.nan)
    prob_fn = plot_utils.get_gpuf_prob_interp(gpuf_dir + seq)
    prob = sdf_util.eval_sdf_interp(
        prob_fn, pc, handle_oob='fill', oob_val=0.)
    unmapped = prob == 0
    gpuf_sdf[unmapped] = np.nan

    cmap = sdf_util.get_colormap(sdf_range=[-0.2, 0.3])

    gt_sdf = process_slices(gt_sdf, pc_shape)
    isdf = process_slices(isdf, pc_shape)
    vox_sdf = process_slices(vox_sdf, pc_shape)
    gpuf_sdf = process_slices(gpuf_sdf, pc_shape)

    gt_sdf = np.flip(gt_sdf, axis=0)
    isdf = np.flip(isdf, axis=0)
    vox_sdf = np.flip(vox_sdf, axis=0)
    gpuf_sdf = np.flip(gpuf_sdf, axis=0)

    pad_shape = np.array(isdf.shape)
    pad_shape[0] = 8
    pad = np.full(pad_shape, 255).astype(np.uint8)
    vis = np.vstack((gt_sdf, pad, isdf, pad, vox_sdf, pad, gpuf_sdf))

    cv2.imwrite(
        save_dir + f"obj_closeups/apt_2_mnp_slices.png", vis)
    import ipdb; ipdb.set_trace()





    """
    Beanbag figure in paper
    """

    # dataset_format, seq, gt_dir, _ = seqs[0]
    # print(seq)

    # gt_mesh = trimesh.load(f"{gt_sdf_path}/{gt_dir}/mesh.obj")
    # gt_sdf_dir = gt_sdf_path + gt_dir + "/1cm/"
    # gt_sdf_interp, sdf_dims, sdf_transform = plot_utils.load_gt_sdf(gt_sdf_dir)

    # # iSDF
    # load_file = f"{isdf_dir}/{seq}_0/model.pth"
    # sdf_map = plot_utils.load_model(load_file, gt_mesh)

    # # gpu fusion
    # gpuf_sdf_interp = plot_utils.get_gpuf_sdf_interp(
    #     gpuf_dir + seq, eval_t=None)

    # # voxblox
    # vox_sdf_interp = plot_utils.get_voxblox_sdf_interp(
    #     voxblox_dir + seq, gt_mesh, eval_t=None)

    # # Full scene level sets -------------------------------------
    # grid_dim = 256
    # grid_pc, scene_scale, bounds_transform = make_grid_pc(
    #     gt_mesh, grid_dim, device)

    # if dataset_format == "replicaCAD":
    #     offset = np.array([0, 0, -16])
    #     step = np.array([8, 0, 0])
    #     gt_mesh = load_undistorted_mesh(replicaCAD_path, gt_dir)
    # else:
    #     offset = np.array([gt_mesh.extents[0], 0, 0])
    #     step = np.array([0, 0, 0])

    # beanbag = np.array([3.8216347, 0.3372789, 7.3370823])

    # # viewing position
    # # cam_pos = np.array([3.2, 1.7, 5])
    # # R, t = transform.look_at(cam_pos, beanbag, [0, -1, 0])
    # # T = np.eye(4)
    # # T[:3, :3] = R
    # # T[:3, 3] = t

    # T = np.array(
    #     [[-9.66392836e-01, 1.27629919e-01, -2.23149031e-01,  3.33394186e+00],
    #      [ 1.14029224e-05, 8.68069663e-01,  4.96442403e-01,  1.37841438e+00],
    #      [ 2.57069808e-01, 4.79755838e-01, -8.38897759e-01,  5.50321724e+00],
    #      [ 0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # scene = trimesh.Scene([gt_mesh])
    # gt_im = draw3D.capture_scene_im(scene, T, tm_pose=True)

    # # isdf mesh
    # isdf_mesh = iSDF_mesh(
    #     grid_pc, grid_dim, sdf_map, scene_scale, bounds_transform)
    # scene = trimesh.Scene([isdf_mesh])
    # isdf_im = draw3D.capture_scene_im(scene, T, tm_pose=True)

    # # voxblox mesh
    # voxblox_mesh = get_mesh(
    #     grid_pc, grid_dim, vox_sdf_interp, scene_scale, bounds_transform)
    # scene = trimesh.Scene([voxblox_mesh])
    # vox_im = draw3D.capture_scene_im(scene, T, tm_pose=True)

    # # gpu fusion mesh
    # gpuf_mesh = get_mesh(
    #     grid_pc, grid_dim, gpuf_sdf_interp, scene_scale, bounds_transform)
    # scene = trimesh.Scene([gpuf_mesh])
    # gpuf_im = draw3D.capture_scene_im(scene, T, tm_pose=True)

    # cv2.imwrite(save_dir + "beanbag_apt_2_nav/gt.png", gt_im)
    # cv2.imwrite(save_dir + "beanbag_apt_2_nav/isdf.png", isdf_im)
    # cv2.imwrite(save_dir + "beanbag_apt_2_nav/vox.png", vox_im)
    # cv2.imwrite(save_dir + "beanbag_apt_2_nav/gpuf.png", gpuf_im)

    # vis = np.hstack((gt_im, isdf_im, vox_im, gpuf_im))
    # vis = cv2.resize(vis, (vis.shape[1] // 4, vis.shape[0] // 4))
    # cv2.imshow('i', vis)
    # cv2.waitKey(0)

    # save slices --------------------------

    # x = torch.linspace(beanbag[0] - 1, beanbag[0] + 1, 100)
    # z = torch.linspace(beanbag[2] - 1, beanbag[2] + 1, 100)
    # y = torch.tensor([0.15, 0.68])

    # xx, yy, zz = torch.meshgrid(x, y, z)
    # pc = torch.cat((xx[..., None], yy[..., None], zz[..., None]), dim=3)
    # pc_shape = pc.shape[:-1]
    # pc = pc.view(-1, 3)

    # # scene.add_geometry(trimesh.PointCloud(pc.cpu()))
    # # scene.show()

    # isdf = sdf_map(pc.to(device)).detach().cpu().numpy()

    # gt_sdf = sdf_util.eval_sdf_interp(
    #     gt_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=0.0)

    # vox_sdf = sdf_util.eval_sdf_interp(
    #     vox_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=np.nan)

    # gpuf_sdf = sdf_util.eval_sdf_interp(
    #     gpuf_sdf_interp, pc.numpy(), handle_oob='fill', oob_val=np.nan)
    # prob_fn = plot_utils.get_gpuf_prob_interp(gpuf_dir + seq)
    # prob = sdf_util.eval_sdf_interp(
    #     prob_fn, pc, handle_oob='fill', oob_val=0.)
    # unmapped = prob == 0
    # gpuf_sdf[unmapped] = np.nan

    # cmap = sdf_util.get_colormap()

    # gt_sdf = process_slices(gt_sdf, pc_shape)
    # isdf = process_slices(isdf, pc_shape)
    # vox_sdf = process_slices(vox_sdf, pc_shape)
    # gpuf_sdf = process_slices(gpuf_sdf, pc_shape)

    # pad_shape = np.array(isdf.shape)
    # pad_shape[0] = 8
    # pad = np.full(pad_shape, 255).astype(np.uint8)
    # vis = np.vstack((gt_sdf, pad, isdf, pad, vox_sdf, pad, gpuf_sdf))

    # cv2.imwrite(save_dir + "beanbag_apt_2_nav/slices.png", vis)

    # import ipdb; ipdb.set_trace()
