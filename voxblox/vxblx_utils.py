import numpy as np
import matplotlib.pylab as plt
import os
from datetime import datetime
import json
import time
import shutil
import scipy
import trimesh
import rospy
import roslaunch
from torchvision import transforms
import torch
import cv2
import imgviz

from isdf.eval import metrics, eval_pts
from isdf.visualisation import sdf_viewer
from isdf.datasets import dataset, image_transforms, sdf_util
from isdf.train import trainer
from isdf import geometry
from isdf.random_gen import random

np.random.seed(0)
torch.manual_seed(0)


def create_launchfile(params):
    voxel_size = params['voxel_size']
    save_dir = params['save_dir']
    save_sdf_dir = os.path.join(save_dir, "out/")

    traj_file = "/traj.txt"
    if params['orb_traj'] and params['dataset_format'] == "replicaCAD":
        traj_file = "/orb_traj.txt"

    save_interval = params['save_interval']
    seq = params['seq']
    fps = params['fps']
    im_indices = params['im_indices']
    update_esdf_every_n_sec = params['update_esdf_every_n_sec']
    noisy_depth = params['noisy_depth']
    dataset_format = params['dataset_format']
    scannet_dir = params['scannet_dir']

    path = f'{save_dir}/launchfile.launch'

    with open(path, 'w') as f:
        f.write(
f'<launch>\n\
  <arg name="voxel_size" default="{voxel_size}"/>\n\
  <arg name="save_sdf_dir" default="{save_sdf_dir}" />\n\
  <arg name="save_dir" default="{save_dir}" />\n\
  <arg name="seq" default="{seq}" />\n\
  <arg name="save_interval" default="{save_interval}" />\n\
\n\
  <!-- Dataset feeder node to feed pointclouds and transforms -->\n\
  <node name="dataset_feeder" pkg="voxblox_ros" type="dataset_feeder.py" output="screen">\n\
    <param name="dir" value="$(arg seq)/" />\n\
    <param name="fps" value="{fps}" />\n\
    <param name="im_indices" value="{im_indices}" />\n\
    <param name="noisy_depth" value="{noisy_depth}" />\n\
    <param name="traj_file" value="{traj_file}" />\n\
    <param name="dataset_format" value="{dataset_format}" />\n\
    <param name="scannet_dir" value="{scannet_dir}" />\n\
  </node>\n\
\n\
  <node name="voxblox_node" pkg="voxblox_ros" type="esdf_server" output="screen" args="-alsologtostderr" clear_params="true">\n\
\n\
    <remap from="pointcloud" to="/replica_node/pointcloud_from_depth"/>\n\
    <remap from="transform" to="/replica_node/transform" /> \n\
\n\
    <remap from="voxblox_node/esdf_map_out" to="esdf_map" />\n\
\n\
    <param name="update_esdf_every_n_sec" value="{update_esdf_every_n_sec}" />\n\
    <param name="esdf_max_distance_m" value="4.0" />\n\
    <param name="esdf_default_distance_m" value="4.0" />\n\
    <param name="publish_pointclouds" value="false" />\n\
    <param name="publish_slices" value="false" />\n\
    <param name="publish_esdf_map" value="true" />\n\
    <param name="publish_tsdf_map" value="false" />\n\
    <param name="output_mesh_as_pcl_mesh" value="false" />\n\
    <param name="tsdf_voxel_size" value="$(arg voxel_size)" />\n\
    <param name="tsdf_voxels_per_side" value="16" />\n\
    <param name="voxel_carving_enabled" value="true" />\n\
    <param name="max_ray_length_m" value="12.0" />\n\
    <param name="color_mode" value="color" />\n\
    <param name="use_tf_transforms" value="false" />\n\
    <param name="update_mesh_every_n_sec" value="10000.0" />\n\
    <param name="min_time_between_msgs_sec" value="0.0" />\n\
    <param name="method" value="fast" />\n\
    <param name="use_const_weight" value="false" />\n\
    <param name="allow_clear" value="true" />\n\
    <param name="verbose" value="true" />\n\
    <param name="mesh_filename" value="$(arg save_sdf_dir)/mesh.ply" />\n\
\n\
    <rosparam file="$(find voxblox_ros)/cfg/replicaCAD_dataset.yaml"/>\n\
  </node>\n\
\n\
  <!-- Listener node that saves point cloud -->\n\
  <node name="listener" pkg="voxblox_ros" type="listener.py" output="screen" >\n\
    <param name="save_sdf_dir" value="$(arg save_sdf_dir)" />\n\
    <param name="save_dir" value="$(arg save_dir)" />\n\
  </node>\n\
\n\
  <!-- Python script to save publishes the pointcloud and saves the mesh. -->\n\
  <node name="save_esdf" pkg="voxblox_ros" type="save_mesh_esdf.py" output="screen" >\n\
    <param name="save_sdf_dir" value="$(arg save_sdf_dir)/" />\n\
    <param name="save_interval" value="$(arg save_interval)" />\n\
  </node>\n\
\n\
</launch>')

    return path


def run_exp(params):

    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])
        os.makedirs(os.path.join(params['save_dir'], 'times'))
        os.makedirs(params['save_dir'] + "/out")

    with open(os.path.join(params['save_dir'], "params.json"), 'w') as fp:
        json.dump(params, fp, indent=4)

    path = create_launchfile(params)

    if params['im_indices'] is not None:
        exp_time = len(params['im_indices']) / params['fps'] + 20
    else:
        if params["dataset_format"] == "replicaCAD":
            n_frames = len(os.listdir(f"{params['seq']}/results/")) / 3
        elif params["dataset_format"] == "ScanNet":
            n_frames = len(os.listdir(
                f"{params['scannet_dir']}/frames/depth/"))
        exp_time = n_frames / params['fps'] + 6
        print("n frames", n_frames)
        print("experiment time", exp_time)

    # Run experiment
    rospy.init_node('placeholder', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(
        uuid, [path])
    launch.start()

    rospy.sleep(exp_time)

    launch.shutdown()


def eval_mesh(params, mesh_gt, n_mesh_samples=200000):
    save_dir = params['save_dir']
    save_sdf_dir = save_dir + "/out/"
    saved_files = os.listdir(save_sdf_dir)
    mesh_files = [f for f in saved_files if f[-4:] == ".ply"]
    mesh_files.sort()

    res = {}

    for file in mesh_files:
        vox_mesh = trimesh.load(os.path.join(save_sdf_dir, file))

        acc, comp = metrics.accuracy_comp(
            mesh_gt, vox_mesh, samples=n_mesh_samples)

        timestamp = file[:-4]
        t = datetime.strptime(timestamp, "%m-%d-%y_%H-%M-%S-%f").timestamp()
        res[timestamp] = {
            "t": t,
            "mesh": {
                "acc": acc,
                "comp": comp
            }
        }

        # # view meshes
        # vox_mesh.visual.face_colors = [160, 160, 160, 255]
        # trimesh.Scene([vox_mesh, mesh_gt]).show()

    return res


def eval_sdf(
    params,
    mesh_gt,
    vis_loss=False,
    vis_mapped_region=False,
    save_eval_pts=True,
):

    res = {}

    t1 = time.time()

    gt_sdf_interp, sdf_dims, sdf_transform = load_gt_sdf(
        params['gt_sdf_dir'], params['dataset_format'])
    print("loaded gt sdf")

    kf_stamps = get_kf_times(params)
    kf_ixs = np.fromiter(kf_stamps.keys(), dtype=int)
    kf_times = np.fromiter(kf_stamps.values(), dtype=float)
    t_kf0 = kf_stamps['0']
    print("kf timestamps")

    save_dir = params['save_dir']
    save_sdf_dir = save_dir + "/out/"
    saved_files = os.listdir(save_sdf_dir)
    esdf_files = [f for f in saved_files if f[-4:] == ".npy"]
    esdf_files.sort()

    eval_pts_root = None
    if save_eval_pts:
        eval_pts_root = save_dir + "/eval_pts"
        os.makedirs(eval_pts_root, exist_ok=True)

    # create axis aligned voxel grid
    vsm = params['voxel_size']
    # mesh_file = params['gt_mesh']
    # mesh_gt = trimesh.load(mesh_file)
    # mesh_gt = params['gt_mesh']
    # print("loaded mesh")
    bounds = mesh_gt.bounds.copy()
    start = bounds[0] - bounds[0] % vsm + vsm / 2 - 25 * vsm
    end = bounds[1] + 25 * vsm
    x = np.arange(start[0], end[0], step=vsm)
    y = np.arange(start[1], end[1], step=vsm)
    z = np.arange(start[2], end[2], step=vsm)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid = np.concatenate(
        (xx[..., None], yy[..., None], zz[..., None]), axis=-1)

    grid_gt_sdf, gt_sdf_mask = sdf_util.eval_sdf_interp(
        gt_sdf_interp, grid.reshape(-1, 3), handle_oob='mask')
    gt_sdf_mask = np.logical_and(gt_sdf_mask, grid_gt_sdf != 0.)
    grid_gt_sdf[~gt_sdf_mask] = np.nan
    grid_gt_sdf = grid_gt_sdf.reshape(grid.shape[:-1])
    print("grid")



    # stage_sdf_file = params['gt_sdf_dir'] + "/stage_sdf.npy"
    # stage_sdf = np.load(stage_sdf_file)
    # stage_sdf_interp = sdf_util.sdf_interpolator(
    #     stage_sdf, sdf_transform)

    if params["dataset_format"] == "replicaCAD":
        up_ix = 1
        H, W = 680, 1200
        fx, fy = W / 2., W / 2.
        cx, cy = W / 2. - 0.5, H / 2. - 0.5
    elif params["dataset_format"] == "ScanNet":
        scannet_dir = params["scannet_dir"]
        intrinsic_file = scannet_dir + "frames/intrinsic/intrinsic_depth.txt"
        K = np.loadtxt(intrinsic_file)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        H, W = 480, 640
    dirs_C = geometry.transform.ray_dirs_C(
        1, H, W, fx, fy, cx, cy, "cpu", depth_type="z")

    obj_bounds = None
    obj_bounds_file = params['seq'] + '/obj_bounds.txt'
    if os.path.exists(obj_bounds_file):
        obj_bounds = eval_pts.load_obj_bounds(obj_bounds_file)

    cached_dataset = eval_pts.get_cache_dataset(
        params['seq'], params['dataset_format'], params['scannet_dir'])
    print("size of cached_dataset", len(cached_dataset))
    got_frame_ixs = []

    n_poses = np.loadtxt(params['seq'] + 'traj.txt').shape[0]

    print(len(esdf_files))

    visible = torch.full(grid.shape[:-1], False)

    print("time to setup eval", time.time() - t1)

    for file in esdf_files:

        t1 = time.time()
        t0 = t1

        vox_sdf = np.loadtxt(os.path.join(save_sdf_dir, file))

        timestamp = file[:-4]
        t = datetime.strptime(
            timestamp, "%m-%d-%y_%H-%M-%S-%f").timestamp()
        t_elapsed = t - t_kf0
        t_str = f'{t_elapsed:.3f}'
        t_elapsed = float(t_str)

        if len(vox_sdf) != 0 and t_elapsed > 0.3:

            # update frame data
            max_ix = min(np.floor(t_elapsed * 30), n_poses)
            past_kfs_ixs = np.arange(0, max_ix).astype(int)
            add_ixs = list(set(past_kfs_ixs) - set(got_frame_ixs))

            if len(add_ixs) > 0:
                sample = cached_dataset[add_ixs]
                new_depth = torch.FloatTensor(sample["depth"])
                new_T_WC = torch.FloatTensor(sample["T"])
                got_frame_ixs = past_kfs_ixs

            sample = cached_dataset[got_frame_ixs]
            depth_batch = torch.FloatTensor(sample["depth"])
            T_WC_batch = torch.FloatTensor(sample["T"])

            print("created grids", time.time() - t1)
            t1 = time.time()

            # create grid

            vox_sdf[:, :3] = np.round(vox_sdf[:, :3] * 2 / vsm) * vsm / 2

            grid_vox_sdf = np.concatenate(
                (grid, np.full([*grid.shape[:-1], 1], np.nan)), axis=-1)
            vox_grid_ixs = np.rint((vox_sdf[:, :3] - start) / vsm).astype(int)
            check = np.logical_and(
                vox_grid_ixs < grid.shape[:-1], vox_grid_ixs > 0)
            check = check.sum(-1) == 3
            vox_grid_ixs = vox_grid_ixs[check]
            vox_sdf = vox_sdf[check]
            grid_vox_sdf[
                vox_grid_ixs[:, 0],
                vox_grid_ixs[:, 1],
                vox_grid_ixs[:, 2], 3] = vox_sdf[:, 3]

            vox_sdf_interp = scipy.interpolate.RegularGridInterpolator(
                (x, y, z), grid_vox_sdf[..., 3])

            # Visible region

            if len(add_ixs) > 0:
                not_yet_visible = visible == False
                pc = torch.FloatTensor(
                    grid[not_yet_visible].copy().reshape(-1, 3))
                batch_size = 500000
                batches = int(np.ceil(pc.shape[0] / batch_size))
                new_vis = None
                for b in range(batches):
                    pc_b = pc[b * batch_size:(b + 1) * batch_size]
                    new_visible = geometry.frustum.is_visible_torch(
                        pc_b, new_T_WC, new_depth,
                        H, W, fx, fy, cx, cy, trunc=0.05)
                    new_visible = new_visible.sum(dim=0)
                    new_visible = new_visible > 0
                    if new_vis is None:
                        new_vis = new_visible
                    else:
                        new_vis = torch.cat((new_vis, new_visible))
                visible[not_yet_visible] = new_vis
                visible_np = visible.cpu().numpy()

                print("tot visible", visible_np.sum())

                print("visible region", time.time() - t1)
                t1 = time.time()


            # check if all visible voxels are occupied in voxblox

            vis = visible_np.sum()
            vox_bool_grid = ~np.isnan(grid_vox_sdf[..., 3])
            vox = vox_bool_grid.sum()
            vis_not_vox = np.logical_and(visible_np, ~vox_bool_grid)
            vox_not_vis = np.logical_and(~visible_np, vox_bool_grid)
            # print("vis, vox, vis_not_vox, vox_not_vis")
            # print(vis, vox, vis_not_vox.sum(), vox_not_vis.sum())

            print("created grids", time.time() - t1)
            t1 = time.time()

            # check all voxblox voxels are in grid
            assert (start < vox_sdf.min(axis=0)[:3]).all()
            assert (end > vox_sdf.max(axis=0)[:3]).all()

            if vis_mapped_region:
                for z_ix in range(grid.shape[1]):
                    vox_slice = vox_bool_grid[:, z_ix]
                    vis_slice = visible_np[:, z_ix]
                    vis_not_vox_slice = vis_not_vox[:, z_ix]

                    vis_slice = vis_slice[..., None].repeat(3, 2).astype(float)
                    vox_slice = vox_slice[..., None].repeat(3, 2).astype(float)
                    vis_not_vox_slice = vis_not_vox_slice[..., None].repeat(
                        3, 2).astype(float)

                    border = np.array([[[1., 0., 0.]]]).repeat(
                        vis_slice.shape[1], 1)

                    im = np.vstack(
                        (vox_slice, border, vis_slice,
                         border, vis_not_vox_slice))

                    # cv2.imwrite(f"voxblox_visible/{z_ix:04d}.png",
                    #             (im * 255).astype(np.uint8))

                    cv2.imshow("im", im)
                    cv2.waitKey(0)

            if vis_loss:
                cmap = sdf_util.get_colormap()

                for z_ix in range(grid.shape[1]):
                    vox_slice = grid_vox_sdf[:, z_ix, :, 3]
                    gt_slice = grid_gt_sdf[:, z_ix]

                    diff = np.abs(vox_slice - gt_slice)
                    diff_viz = imgviz.depth2rgb(diff, max_value=0.5)
                    diff_viz = diff_viz[..., ::-1]

                    vox_slice = cmap.to_rgba(vox_slice, alpha=1., bytes=False)
                    gt_slice = cmap.to_rgba(gt_slice, alpha=1., bytes=False)
                    vox_slice = vox_slice[..., :3][..., ::-1]
                    gt_slice = gt_slice[..., :3][..., ::-1]

                    im = np.vstack((vox_slice, gt_slice, diff_viz))

                    cv2.imshow("im", im)
                    cv2.waitKey(0)

            # # Evaluation along future trajectory
            # pred_chomp_costs, gt_chomp_costs = sdf_traj_eval(
            #     gt_sdf_interp, vox_sdf_interp, params['traj_file'],
            #     params["seq"], t_elapsed)

            eval_pts_dir = None
            if eval_pts_root is not None:
                eval_pts_dir = eval_pts_root + '/' + t_str + '/'
                os.makedirs(eval_pts_dir, exist_ok=True)

            # Evaluation around objects

            obj_res = []
            if obj_bounds is not None:
                obj_res = sdf_objects_eval(
                    gt_sdf_interp, vox_sdf_interp, vox_sdf,
                    obj_bounds, start, vsm,
                    visible_np, T_WC_batch,
                    depth_batch,
                    eval_pts_dir=eval_pts_dir)

            # Evaluation at voxblox voxels centers

            centers_res = sdf_eval_vxblx_voxels(gt_sdf_interp, vox_sdf)

            visible_res = sdf_eval_visible(
                t_str, depth_batch, T_WC_batch, vox_sdf,
                gt_sdf_interp, vox_sdf_interp, dirs_C,
                params["dataset_format"],
                eval_pts_dir=eval_pts_dir,
            )
            visible_surf_res = sdf_eval_surface(
                t_str, depth_batch, T_WC_batch, vox_sdf,
                gt_sdf_interp, vox_sdf_interp, dirs_C,
                eval_pts_dir=eval_pts_dir,
            )

            # Evaluation in full volume
            eval_pts_vol_root = "/home/joe/projects/incSDF/data/eval_pts/"
            seq = [x for x in params['seq'].split('/') if x != ""][-1]
            vol_res = full_vol(
                vox_sdf_interp, vox_sdf, seq, eval_pts_vol_root,
                params['dataset_format']
            )

            print("visible region time", time.time() - t1)

            print(
                "t {:.3f}".format(t_elapsed),
                "  kfs ", int(len(past_kfs_ixs)),
                "  center {:.3f}".format(centers_res["av_l1"]),
                "  vol nn {:.3f}".format(vol_res["nn"]["av_l1"]),
                "  vox and vis sdf {:.3f}".format(
                    visible_res['vox']['av_l1']),
                "  nvox_over_nvis {:.3f}".format(
                    visible_res['vox']['prop_vox']),
                "  vis nn sdf {:.3f}".format(
                    visible_res['nn']['av_l1']),
                "  obj ", [res['vox']['av_l1'] for res in obj_res],
                "  eval_time {:.3f}".format(time.time() - t0),
                # "  traj {:.3f} {:.3f}".format(pred_cost, gt_cost)
            )

            # if final save was interrupted, then do not keep this eval
            if file == esdf_files[-1]:
                k = list(res.keys())[-1]
                last_prop = res[k]['rays']['vox']['prop_vox']
                print("Check if discard last results")
                print(last_prop * 0.85, visible_res['vox']['prop_vox'])
                if last_prop * 0.85 > visible_res['vox']['prop_vox']:
                    print("Not using last results")
                    if eval_pts_dir is not None:
                        shutil.rmtree(eval_pts_dir)
                    continue

            res[timestamp] = {
                "time": t_elapsed,
                "voxblox": centers_res,
                "rays": visible_res,
                "visible_surf": visible_surf_res,
                "objects": obj_res,
                "vol": vol_res,
            }

            # my_viewer = sdf_viewer.SDFViewer(mesh=mesh_gt, sdf_pc=vox_sdf)

    return res


def sdf_eval_vxblx_voxels(
    gt_sdf_interp, vox_sdf, max_voxels=200000,
):
    if len(vox_sdf) > max_voxels:
        # print("cutting num voxels from ", len(vox_sdf))
        ixs = np.random.choice(
            np.arange(len(vox_sdf)), size=max_voxels, replace=False)
        vox_sdf = vox_sdf[ixs]

    gt_sdf, gt_valid_mask = sdf_util.eval_sdf_interp(
        gt_sdf_interp, vox_sdf[:, :3], handle_oob='mask')
    gt_valid_mask = np.logical_and(gt_valid_mask, gt_sdf != 0.)

    # print("invalid gt at: ", (~gt_valid_mask).sum(), "/", len(vox_sdf))

    gt_sdf = gt_sdf[gt_valid_mask]
    vox_sdf_valid = vox_sdf[gt_valid_mask, 3]

    sdf_diff = vox_sdf_valid - gt_sdf
    sdf_diff = np.abs(sdf_diff)
    l1_sdf = sdf_diff.mean()

    res = {
        'av_l1': l1_sdf,
    }

    return res


def eval_voxblox_sdf(
    pts, gt_sdf_interp, vox_sdf_interp, vox_sdf,
    eval_pts_prefix=None, gt_sdf=None, do_grad=False,
):
    # Compute GT sdf and gradient

    if gt_sdf is None:
        gt_sdf, gt_valid_sdf = sdf_util.eval_sdf_interp(
            gt_sdf_interp, pts, handle_oob='mask')
        # gt sdf gives value 0 inside the walls. Don't include this in loss
        gt_valid_sdf = np.logical_and(gt_sdf != 0., gt_valid_sdf)
        sdf_pts = pts[gt_valid_sdf]
        gt_sdf = gt_sdf[gt_valid_sdf]
    else:
        sdf_pts = pts

    if do_grad:
        def gt_sdf_fn(points):
            sdf, valid_mask = sdf_util.eval_sdf_interp(
                gt_sdf_interp, points, handle_oob='mask')
            valid_mask = np.logical_and(sdf != 0., valid_mask)
            sdf[~valid_mask] = np.nan
            return sdf
        gt_grad_all, gt_valid_grad = eval_pts.eval_grad(
            gt_sdf_fn, pts, 0.01, is_gt_sdf=True)
        gt_grad = gt_grad_all[gt_valid_grad]
        grad_pts = pts[gt_valid_grad]

        vox_sdf_fn = lambda points : scipy.interpolate.griddata(
            vox_sdf[:, :3], vox_sdf[:, 3], points, method="nearest")

    # Save mask for which points are valid

    if eval_pts_prefix is not None:
        np.save(eval_pts_prefix + "valid_gt_sdf.npy", gt_valid_sdf)
        if do_grad:
            np.save(eval_pts_prefix + "valid_gt_grad.npy", gt_valid_grad)

    # Voxblox

    pred_sdf = sdf_util.eval_sdf_interp(
        vox_sdf_interp, sdf_pts, handle_oob='fill', oob_val=np.nan)
    vox_valid_sdf = ~np.isnan(pred_sdf)

    if do_grad:
        vox_region_gt_valid_grad = gt_valid_grad[gt_valid_sdf][vox_valid_sdf]
        vox_grad_pts = sdf_pts[vox_valid_sdf][vox_region_gt_valid_grad]
        gt_grad_vox = gt_grad_all[gt_valid_sdf][vox_valid_sdf][
            vox_region_gt_valid_grad]

        vox_res = np.linalg.norm(vox_sdf[1, :3] - vox_sdf[0, :3])
        vox_grad_1, _ = eval_pts.eval_grad(
            vox_sdf_fn, vox_grad_pts, vox_res, is_gt_sdf=False)
        vox_grad_2, _ = eval_pts.eval_grad(
            vox_sdf_fn, vox_grad_pts, vox_res * 2, is_gt_sdf=False)

    prop_vox = vox_valid_sdf.sum() / vox_valid_sdf.shape[0]

    gt_sdf_vis_and_vox = gt_sdf[vox_valid_sdf]
    vis_and_vox_pts = sdf_pts[vox_valid_sdf]
    pred_sdf = pred_sdf[vox_valid_sdf]

    if eval_pts_prefix is not None:
        np.save(eval_pts_prefix + "valid_vox_sdf.npy", vox_valid_sdf)
        # if do_grad:
        #     np.save(eval_pts_prefix + "valid_vox_grad.npy", vox_valid_grad)

    # nearest neighbour and fill

    vis_not_vox_pts = sdf_pts[~vox_valid_sdf]
    gt_sdf_vis_not_vox = gt_sdf[~vox_valid_sdf]
    pred_sdf_nn = scipy.interpolate.griddata(
        vox_sdf[:, :3], vox_sdf[:, 3], vis_not_vox_pts, method="nearest")

    if do_grad:
        vox_sdf_fn = lambda points : scipy.interpolate.griddata(
            vox_sdf[:, :3], vox_sdf[:, 3], points, method="nearest")
        vox_res = np.linalg.norm(vox_sdf[1, :3] - vox_sdf[0, :3])
        vis_grad_1, _ = eval_pts.eval_grad(
            vox_sdf_fn, grad_pts, vox_res, is_gt_sdf=False)
        vis_grad_2, _ = eval_pts.eval_grad(
            vox_sdf_fn, grad_pts, vox_res * 2, is_gt_sdf=False)

    combined_gt = np.concatenate((gt_sdf_vis_and_vox, gt_sdf_vis_not_vox))
    combined_pred_nn = np.concatenate((pred_sdf, pred_sdf_nn))
    fill_value = 0.
    combined_pred_fill = np.concatenate((
        pred_sdf, np.full(pred_sdf_nn.shape, fill_value)))

    # Compute losses

    l1, bins_loss, l1_chomp_costs = losses(
        pred_sdf, gt_sdf_vis_and_vox)

    l1_nn, bins_loss_nn, l1_chomp_costs_nn = losses(
        combined_pred_nn, combined_gt)

    l1_fill, bins_loss_fill, l1_chomp_costs_fill = losses(
        combined_pred_fill, combined_gt)

    res = {
        # evaluated at points where voxblox can do trilinear interp
        'vox': {
            'av_l1': l1,
            'prop_vox': prop_vox,
        },
        # evaluated at all points using nn where we can't interp
        'nn': {
            'av_l1': l1_nn,
        },
        # evaluated at all points using constant val where we can't interp
        'fill': {
            'fill_value': fill_value,
            'av_l1': l1_fill,
        }
    }

    if do_grad:
        cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        cosdist_vis_1 = 1 - cossim(
            torch.tensor(vis_grad_1), torch.tensor(gt_grad))
        cosdist_vis_1 = cosdist_vis_1.mean().item()

        cosdist_vox_1 = 1 - cossim(
            torch.tensor(vox_grad_1), torch.tensor(gt_grad_vox))
        cosdist_vox_1 = cosdist_vox_1.mean().item()

        cosdist_vis_2 = 1 - cossim(
            torch.tensor(vis_grad_2), torch.tensor(gt_grad))
        cosdist_vis_2 = cosdist_vis_2.mean().item()

        cosdist_vox_2 = 1 - cossim(
            torch.tensor(vox_grad_2), torch.tensor(gt_grad_vox))
        cosdist_vox_2 = cosdist_vox_2.mean().item()

        print('visible / vox region cosdist', cosdist_vis_1, cosdist_vox_1)

        res['nn']['av_cossim'] = [cosdist_vis_1, cosdist_vis_2]
        res['vox']['av_cossim'] = [cosdist_vox_1, cosdist_vox_2]

    if bins_loss is not None:
        res['vox']['binned_l1'] = bins_loss
        res['vox']['l1_chomp_costs'] = l1_chomp_costs
        res['nn']['binned_l1'] = bins_loss_nn
        res['nn']['l1_chomp_costs'] = l1_chomp_costs_nn
        res['fill']['binned_l1'] = bins_loss_fill
        res['fill']['l1_chomp_costs'] = l1_chomp_costs_fill

    return res


def sdf_eval_visible(
    t_str, depth_batch, T_WC_batch, vox_sdf,
    gt_sdf_interp, vox_sdf_interp, dirs_C,
    dataset_format,
    eval_pts_dir=None,
):
    pts = eval_pts.sample_visible_region(
        t_str, depth_batch, T_WC_batch, dataset_format, dirs_C)

    eval_pts_prefix = None
    if eval_pts_dir is not None:
        eval_pts_prefix = eval_pts_dir + "/vis_"

    res = eval_voxblox_sdf(
        pts, gt_sdf_interp, vox_sdf_interp, vox_sdf,
        eval_pts_prefix=eval_pts_prefix, do_grad=True,
    )

    return res


def sdf_eval_surface(
    t_str, depth_batch, T_WC_batch, vox_sdf,
    gt_sdf_interp, vox_sdf_interp, dirs_C,
    eval_pts_dir=None,
):
    pts = eval_pts.sample_surface(
        t_str, depth_batch, T_WC_batch, dirs_C)

    eval_pts_prefix = None
    if eval_pts_dir is not None:
        eval_pts_prefix = eval_pts_dir + "/surf_"

    res = eval_voxblox_sdf(
        pts, gt_sdf_interp, vox_sdf_interp, vox_sdf,
        eval_pts_prefix=eval_pts_prefix, do_grad=False,
    )

    return res


def sdf_objects_eval(
    gt_sdf_interp, vox_sdf_interp, vox_sdf,
    obj_bounds, start,
    vsm, visible_np, T_WC_batch, depth_batch,
    eval_pts_dir=None,
):
    obj_centers = obj_bounds.mean(axis=1)

    grid_ixs_bounds = np.round((obj_bounds - start) / vsm)
    visible_props = []
    for ixs_bounds in grid_ixs_bounds:
        x = np.linspace(ixs_bounds[0, 0], ixs_bounds[1, 0], 5)
        y = np.linspace(ixs_bounds[0, 1], ixs_bounds[1, 1], 5)
        z = np.linspace(ixs_bounds[0, 2], ixs_bounds[1, 2], 5)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid = np.concatenate(
            (xx[..., None], yy[..., None], zz[..., None]), axis=-1)
        ixs = grid.reshape(-1, 3).astype(int)

        obj_visible = visible_np[ixs[:, 0], ixs[:, 1], ixs[:, 2]]
        visible_prop = obj_visible.sum() / obj_visible.shape[0]
        visible_props.append(visible_prop)

    obj_visible = np.array(visible_props) > 0.5

    print("visible_props", visible_props, obj_visible)

    l1s = []
    prop_can_interp = []

    results = []
    for i in range(len(obj_bounds)):
        if obj_visible[i]:

            pts = eval_pts.object_eval_pts(obj_bounds[i])

            eval_pts_prefix = None
            if eval_pts_dir is not None:
                eval_pts_prefix = eval_pts_dir + f"/obj{i}_"

            res = eval_voxblox_sdf(
                pts, gt_sdf_interp, vox_sdf_interp, vox_sdf,
                eval_pts_prefix=eval_pts_prefix,
                do_grad=False,
            )
            results.append(res)

    return results


def sdf_traj_eval(
    gt_sdf_interp, vox_sdf_interp, traj_file,
    seq_dir, t_elapsed, t_ahead=5.
):
    gt_traj = np.loadtxt(seq_dir + traj_file)
    traj_start_ix = t_elapsed * 30
    traj_end_ix = (t_elapsed + t_ahead) * 30
    traj_end_ix = min(len(gt_traj) - 1, traj_end_ix)

    traj_section = gt_traj[int(traj_start_ix): int(traj_end_ix)]
    eval_pts = traj_section[:, [3, 7, 11]]

    gt_sdf, valid = sdf_util.eval_sdf_interp(
        gt_sdf_interp, eval_pts, handle_oob='mask')
    valid = np.logical_and(gt_sdf != 0., valid)

    pred_sdf = sdf_util.eval_sdf_interp(
        vox_sdf_interp, eval_pts, handle_oob='except')
    vox_nan_mask = np.isnan(pred_sdf)

    if (valid.sum() < (0.9 * valid.shape[0]) or len(traj_section) < 30 or vox_nan_mask.sum() > 0):
        return np.nan, np.nan

    gt_sdf = gt_sdf[valid]
    sdf = pred_sdf[valid]

    epsilons = [1., 1.5, 2.]
    pred_chomp_costs = [
        metrics.chomp_cost(sdf, epsilon=epsilon).sum().item()
        for epsilon in epsilons
    ]
    gt_chomp_costs = [
        metrics.chomp_cost(gt_sdf, epsilon=epsilon).sum()
        for epsilon in epsilons
    ]

    return pred_chomp_costs, gt_chomp_costs


def full_vol(vox_sdf_interp, vox_sdf, seq, eval_pts_root, dataset_format):

    if dataset_format == "replicaCAD":
        vol_pts_file = eval_pts_root + "full_vol/replicaCAD.npy"
        gt_sdf_file = eval_pts_root + f"full_vol/gt_{seq}.npy"
    if dataset_format == "ScanNet":
        vol_pts_file = eval_pts_root + f"full_vol/{seq}.npy"
        gt_sdf_file = eval_pts_root + f"full_vol/gt_{seq}.npy"

    pts = np.load(vol_pts_file)
    gt_sdf = np.load(gt_sdf_file)

    res = eval_voxblox_sdf(
        pts, None, vox_sdf_interp, vox_sdf, gt_sdf=gt_sdf
    )

    return res


def losses(sdf, gt_sdf):
    sdf = torch.from_numpy(sdf)
    gt_sdf = torch.from_numpy(gt_sdf)

    sdf_diff = sdf - gt_sdf
    sdf_diff = torch.abs(sdf_diff)
    l1_sdf = sdf_diff.mean().item()

    bins_loss = metrics.binned_losses(sdf_diff, gt_sdf)

    # chomp cost difference
    epsilons = [1., 1.5, 2.]
    l1_chomp_costs = [
        torch.abs(
            metrics.chomp_cost(sdf, epsilon=epsilon) -
            metrics.chomp_cost(gt_sdf, epsilon=epsilon)
        ).mean().item() for epsilon in epsilons
    ]

    return l1_sdf, bins_loss, l1_chomp_costs


def load_gt_sdf(gt_sdf_dir, dataset_format):
    gt_sdf_file = gt_sdf_dir + "/sdf.npy"
    # stage_sdf_file = gt_sdf_dir + "/stage_sdf.npy"
    sdf_transf_file = gt_sdf_dir + "/transform.txt"

    sdf_grid = np.load(gt_sdf_file)
    if dataset_format == "ScanNet":
        sdf_grid = np.abs(sdf_grid)
    sdf_transform = np.loadtxt(sdf_transf_file)
    gt_sdf_interp = sdf_util.sdf_interpolator(
        sdf_grid, sdf_transform)
    sdf_dims = torch.tensor(sdf_grid.shape)
    return gt_sdf_interp, sdf_dims, sdf_transform


def get_timings(params, res):
    """
    Get TSDF integration times.
    Get mean, var and max ESDF update times.
    """

    save_dir = params['save_dir']

    with open(os.path.join(
            save_dir, 'times/esdf_max_time.txt'), 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        esdf_max_time = float(last_line.split()[1])

    with open(os.path.join(
            save_dir, 'times/esdf_mean_time.txt'), 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        esdf_mean_time = float(last_line.split()[1])

    with open(os.path.join(
            save_dir, 'times/esdf_var_time.txt'), 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        esdf_var_time = float(last_line.split()[1])

    tsdf_times = []
    with open(os.path.join(
            save_dir, 'times/tsdf_integration_time.txt'), 'r') as f:
        for line in f.readlines():
            tsdf_times.append(float(line.split()[1]))

    res['times'] = {
        'esdf_max_time': esdf_max_time,
        'esdf_mean_time': esdf_mean_time,
        'esdf_var_time': esdf_var_time,
        'tsdf_times': tsdf_times
    }

    return res


def get_kf_times(params):
    kf_stamps = {}
    direc = params['save_dir']

    with open(os.path.join(direc, "kf_stamps.txt"), "r") as f:
        for line in f.readlines():
            split = line.split()
            t = datetime.strptime(split[1], "%m-%d-%y_%H-%M-%S-%f").timestamp()
            kf_stamps[split[0]] = t

    return kf_stamps


def plot_acc_comp(res, params):

    ts = [v['t'] for v in res.values()]
    acc = [v['mesh']['acc'] for v in res.values()]
    comp = [v['mesh']['comp'] for v in res.values()]

    kf_stamps = get_kf_times(params)
    t0 = list(kf_stamps.values())[0]

    kf_times = np.array(list(kf_stamps.values())) - t0
    ts = np.array(ts) - t0

    plt.plot(ts, acc)
    plt.title("accuracy")
    plt.show()

    plt.plot(ts, comp)
    plt.title("completion")
    plt.show()


if __name__ == "__main__":

    seq = "apt_2_mnp"
    gt_dir = "apt_2_v1" 
    direc = f"/home/joe/projects/incSDF/res/voxblox/gt_traj/0.055/{seq}/"

    params_file = direc + "/params.json"
    with open(params_file, 'r') as f:
        params = json.load(f)

    params["save_dir"] = "/".join(params_file.split('/')[:-1])

    mesh_gt = trimesh.load(
        f"/home/joe/projects/incSDF/data/gt_sdfs/{gt_dir}/mesh.obj")

    sdf_res = eval_sdf(
        params, mesh_gt, save_eval_pts=False
    )

    res = {
        "sdf_eval": sdf_res,
    }

    # Save evaluation results
    with open(os.path.join(params['save_dir'], "res.json"), "w") as f:
        json.dump(res, f, indent=4)
