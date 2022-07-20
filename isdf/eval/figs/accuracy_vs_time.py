import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, FixedLocator
import json
import os
import cv2

from isdf.eval import plot_utils


bin_limits = np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])
bins_lb = bin_limits[:-1]
bins_ub = bin_limits[1:]


def get_sdf_res(sdf_res):
    if 'bins_lb' in sdf_res:
        bins_lb = sdf_res.pop('bins_lb')
        bins_ub = sdf_res.pop('bins_ub')

    times = []
    ray_l1, vol_l1, vxb_l1 = [], [], []
    ray_binned_l1, vol_binned_l1, vxb_binned_l1 = [], [], []
    objects_l1 = []
    l1_chomp_costs = []
    for t in sdf_res.keys():
        times.append(sdf_res[t]['time'])
        ray_l1.append(sdf_res[t]['rays']['av_l1'])
        ray_binned_l1.append(sdf_res[t]['rays']['binned_l1'])
        if 'l1_chomp_costs' in sdf_res[t]['rays'].keys():
            l1_chomp_costs.append(sdf_res[t]['rays']['l1_chomp_costs'])
        if 'vol' in sdf_res[t].keys():
            vol_l1.append(sdf_res[t]['vol']['av_l1'])
            vol_binned_l1.append(sdf_res[t]['vol']['binned_l1'])
        if 'voxblox' in sdf_res[t].keys():
            vxb_l1.append(sdf_res[t]['voxblox']['av_l1'])
            vxb_binned_l1.append(sdf_res[t]['voxblox']['binned_l1'])
        if 'objects_l1' in sdf_res[t].keys():
            r = sdf_res[t]['objects_l1']
            r = [x if x is not None else np.nan for x in r]
            objects_l1.append(r)

    times = np.array(times)
    ray_l1 = np.array(ray_l1)
    ray_binned_l1 = np.array(ray_binned_l1)
    vol_l1 = np.array(vol_l1)
    vol_binned_l1 = np.array(vol_binned_l1)
    vxb_l1 = np.array(vxb_l1)
    vxb_binned_l1 = np.array(vxb_binned_l1)
    l1_chomp_costs = np.array(l1_chomp_costs)

    # convert to cm

    ray_l1 = ray_l1 * 100
    vol_l1 = vol_l1 * 100
    vxb_l1 = vxb_l1 * 100
    ray_binned_l1 = ray_binned_l1 * 100
    vol_binned_l1 = vol_binned_l1 * 100
    vxb_binned_l1 = vxb_binned_l1 * 100
    objects_l1 = np.array([np.array(x) * 100 for x in objects_l1])
    l1_chomp_costs = l1_chomp_costs * 100

    l1 = [ray_l1, vol_l1, vxb_l1]
    binned_l1 = [ray_binned_l1, vol_binned_l1, vxb_binned_l1]

    return (
        times, l1,
        binned_l1,
        objects_l1,
        l1_chomp_costs
    )


def ema_smooth(scalars, weight):
    """ Exponential moving average smoothing """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def aggregate_exps(exps, eval_freq_s=0.5, return_binned=False, cost="sdf"):
    assert cost in ["sdf", "chomp_1", "chomp_1.5", "chomp_2"]

    times_lists = []
    l1_lists = []
    l1_chomp_lists = []
    l1_linear_lists = []
    binned_l1_lists = []
    objects_l1_lists = []

    for exp in exps:
        res = plot_utils.load_res(exp)
        times, l1, binned_l1, objects_l1, l1_chomp_costs =\
            get_sdf_res(res['sdf_eval'])
        # Check the experiment finished
        if plot_utils.get_seq_time(exp) < times[-1] and "kf_indices" in res.keys():
            times_lists.append(times)
            l1_lists.append(l1[0])
            l1_chomp_lists.append(l1_chomp_costs)
            if return_binned:
                binned_l1_lists.append(binned_l1[0])
                objects_l1_lists.append(objects_l1)

    print("Num experiments", len(l1_lists))

    max_len = np.max([len(x) for x in l1_lists])

    if cost == "chomp_1":
        l1_lists = [x[:, 0] for x in l1_chomp_lists]
    elif cost == "chomp_1.5":
        l1_lists = [x[:, 1] for x in l1_chomp_lists]
    elif cost == "chomp_2":
        l1_lists = [x[:, 2] for x in l1_chomp_lists]

    l1_lists = np.array([
        np.concatenate((x, np.full([max_len - x.shape[0]], np.nan)))
        for x in l1_lists
    ])
    mean = np.nanmean(l1_lists, axis=0)
    std = np.nanstd(l1_lists, axis=0)

    keep_ts = (~np.isnan(l1_lists)).sum(axis=0) > (len(l1_lists) / 2)
    mean = mean[keep_ts]
    std = std[keep_ts]

    times = eval_freq_s + eval_freq_s * np.arange(len(mean))

    if return_binned is False:
        return mean, std, times

    else:
        objects_mean, objects_std = None, None
        if len(objects_l1) > 0:
            objects_l1_lists = np.array([
                np.concatenate(
                    (x, np.full([max_len - x.shape[0], x.shape[1]], np.nan)))
                for x in objects_l1_lists
            ])
            objects_mean = np.nanmean(objects_l1_lists, axis=0)
            objects_std = np.nanstd(objects_l1_lists, axis=0)
            bad = np.isnan(objects_l1_lists).sum(axis=0) > (len(l1_lists) / 2)
            objects_mean[bad] = np.nan
            objects_std[bad] = np.nan
            objects_mean = objects_mean[keep_ts]
            objects_std = objects_std[keep_ts]

        binned_l1_lists = np.array([
            np.concatenate(
                (x, np.full([max_len - x.shape[0], x.shape[1]], np.nan)))
            for x in binned_l1_lists
        ])
        binned_mean = np.nanmean(binned_l1_lists, axis=0)
        binned_std = np.nanstd(binned_l1_lists, axis=0)
        bad = np.isnan(binned_l1_lists).sum(axis=0) > (len(l1_lists) / 2)
        binned_mean[bad] = np.nan
        binned_std[bad] = np.nan
        binned_mean = binned_mean[keep_ts]
        binned_std = binned_std[keep_ts]

        return (
            mean, std, times,
            binned_mean, binned_std,
            objects_mean, objects_std,
        )


def draw_keyframes(
    exp_name,
):
    res = plot_utils.load_res(exp_name)

    with open(exp_name + "/config.json") as json_file:
        config = json.load(json_file)

    if config['dataset']['format'] == 'replicaCAD':
        seq_name = config['dataset']['seq_dir'].split('/')[-2]
        data_dir = "/home/joe/projects/incSDF/data/ReplicaCAD-seqs/"
        kf_dir = os.path.join(data_dir, seq_name, "results/")
    elif config['dataset']['format'] == 'ScanNet':
        seq_name = config['dataset']['seq_dir'].split('/')[-2]
        data_dir = "/mnt/sda/ScanNet/scans/"
        kf_dir = os.path.join(data_dir, seq_name, "frames/color/")

    if "kf_indices" in res:
        kf_ixs = res["kf_indices"]
        if len(kf_ixs) > 8:
            choice = np.random.choice(
                np.arange(len(kf_ixs), dtype=int), 8, replace=False)
            kf_ixs = np.array(kf_ixs)[choice]
            kf_ixs.sort()
    else:
        n_frames = len(os.listdir(kf_dir))
        if config['dataset']['format'] == 'replicaCAD':
            n_frames = n_frames // 3
        kf_ixs = np.random.choice(np.arange(n_frames), 8, replace=False)
        kf_ixs.sort()

    kf_times = [x / 30. for x in kf_ixs]
    kf_time_labels = [f"{x:.0f}" for x in kf_times]

    kfs = []
    for ix in kf_ixs:
        if config['dataset']['format'] == 'replicaCAD':
            s = f"{ix:06}"
            rgb_file = kf_dir + "frame" + s + ".png"
        elif config['dataset']['format'] == 'ScanNet':
            rgb_file = kf_dir + str(ix) + ".jpg"

        im = cv2.imread(rgb_file)
        kfs.append(im)

    kfs = np.array(kfs)
    h, w = kfs.shape[1:3]
    kfs = np.hstack((kfs))

    gs = ax[2, 0].get_gridspec()
    for a in ax[2, :]:
        a.remove()
    axbig = fig.add_subplot(gs[2, :])
    axbig.imshow(kfs[..., ::-1])
    axbig.set_xlabel("Keyframe times (s)")
    x_ticks = np.arange(len(kf_ixs)) * w + w / 2
    axbig.set_xticks(x_ticks)
    axbig.set_xticklabels(kf_time_labels)
    axbig.set_yticklabels([])
    axbig.set_yticks([])


def plot_l1_av(times, l1_mean, l1_std, r, c,
               title=None, label=None, smoothing=0., cost="sdf"):
    assert cost in ["sdf", "chomp_1", "chomp_1.5", "chomp_2"]
    l1_mean = ema_smooth(l1_mean, smoothing)
    if l1_std is not None:
        ax[r, c].fill_between(
            times, l1_mean + l1_std, l1_mean - l1_std, alpha=0.5)
    ax[r, c].plot(times, l1_mean, label=label)
    ax[r, c].title.set_text(title)

    if r == 1:
        ax[r, c].set_xlabel("Keyframe times (s)")
    if c == 0:
        if "chomp" in cost:
            ax[r, c].set_ylabel("CHOMP cost |pred - GT| (epsilon " + \
                                cost.split("_")[0] + "m)")
        else:
            ax[r, c].set_ylabel("|SDF predicted - SDF GT| (cm)")
    if r == 0 and c == 0:
        ax[r, c].legend()


def plot_l1_sdf(times, l1, binned_l1,
                obj_l1s=None, label=None, smoothing=0.,
                l1_std=None, binned_l1_std=None, obj_l1_std=None):
    l1 = ema_smooth(l1, smoothing)

    ax[0, 0].title.set_text("Average")
    ax[0, 0].plot(times, l1, label=label)
    if l1_std is not None:
        l1_std = ema_smooth(l1_std, smoothing)
        ax[0, 0].fill_between(
            times, l1 + l1_std, l1 - l1_std, alpha=0.5)
    ax[0, 0].set_ylabel("|SDF predicted - SDF GT| (cm)")
    ax[0, 0].legend()
    [x.set_linewidth(3.) for x in ax[0, 0].spines.values()]

    for j in range(len(bins_lb)):
        binned_l1[:, j] = ema_smooth(binned_l1[:, j], smoothing)

        r = (j + 1) // 4
        c = (j + 1) % 4
        ax[r, c].plot(times, binned_l1[:, j], label=label)
        if binned_l1_std is not None:
            binned_l1 = ema_smooth(binned_l1, smoothing)
            ax[r, c].fill_between(
                times,
                binned_l1[:, j] + binned_l1_std[:, j],
                binned_l1[:, j] - binned_l1_std[:, j],
                alpha=0.5)
        if r == 0 and c == 1:
            ax[r, c].title.set_text(
                "{}cm < s".format(int(bins_ub[j])))
        elif r == 1 and c == 3:
            ax[r, c].title.set_text(
                "s <= {}cm".format(int(bins_lb[j])))
        else:
            ax[r, c].title.set_text(
                "{}cm <= s < {}cm".format(
                    int(bins_lb[j]), int(bins_ub[j])))
        if r == 1:
            ax[r, c].set_xlabel("Keyframe times (s)")
        if c == 0:
            ax[r, c].set_ylabel("|SDF predicted - SDF GT| (cm)")
        if j == 0:
            ax[r, c].legend()

    if obj_l1s is not None:
        for j in range(obj_l1s.shape[1]):
            if j == 0:
                ax[j, -1].title.set_text("Target objects")
            ax[j, -1].plot(times, obj_l1s[:, j], label=None)
            if obj_l1_std is not None:
                ax[j, -1].fill_between(
                    times,
                    obj_l1s[:, j] + obj_l1_std[:, j],
                    obj_l1s[:, j] - obj_l1_std[:, j],
                    alpha=0.5)
            [x.set_linewidth(3.) for x in ax[j, -1].spines.values()]


def set_axis_bounds(
    exp=None, ymax=None, ymin=1, ycap=200, row=None, col=None,
):
    """ Set xticks at keyframe timestamps.
        Log scale for y ticks shared across all axes.
    """
    xticks = None
    if exp is not None:
        # get keyframe timestamps
        res = plot_utils.load_res(exp)
        if "kf_indices" in res:
            with open(exp + "/config.json") as json_file:
                config = json.load(json_file)

            if config['dataset']['format'] == 'replicaCAD':
                seq_name = config['dataset']['seq_dir'].split('/')[-2]
                data_dir = "/home/joe/projects/incSDF/data/ReplicaCAD-seqs/"
                kf_dir = os.path.join(data_dir, seq_name, "results/")
                n_frames = len(os.listdir(kf_dir)) // 3
            elif config['dataset']['format'] == 'ScanNet':
                seq_name = config['dataset']['seq_dir'].split('/')[-2]
                data_dir = "/mnt/sda/ScanNet/scans/"
                kf_dir = os.path.join(data_dir, seq_name, "frames/color/")
                n_frames = len(os.listdir(kf_dir))

            xticks = np.array(res['kf_indices'] + [n_frames]) / 30.
            xtick_labels = [f'{x:.0f}' for x in xticks]

            # sparsify x ticks
            carried = 0
            dts = xticks[1:] - xticks[:-1]
            for i, dt in enumerate(dts):
                if (dt + carried) / xticks[-1] < 0.08:
                    xtick_labels[i + 1] = ""
                    carried += dt
                else:
                    carried = 0

    if ymax is not None:
        ymax = min(ycap, ymax)
        yticks = [1, 2, 5, 10, 20, 50, 100]
        yticks = [y for y in yticks if y <= ymax and y >= ymin]
        ytick_labels = [f'{y:.0f}' for y in yticks]

    for r in range(2):
        for c in range(ax.shape[1]):
            if (row is None and col is None) or (r == row and c == col):
                if xticks is not None:
                    ax[r, c].set_xlim([-2, xticks[-1] + 6])
                    ax[r, c].set_xticks(xticks)
                    ax[r, c].set_xticklabels(xtick_labels)

                if ymax is not None:
                    ax[r, c].set_yscale('log')
                    ax[r, c].set_ylim([2., ymax])
                    ax[r, c].set_yticks(yticks)
                    ax[r, c].set_yticklabels(ytick_labels)

    # For row which displays keyframes
    if ax.shape[0] == 3:
        for c in range(ax.shape[1]):
            ax[-1, c].set_yscale('linear')
            ax[-1, c].set_xscale('linear')
            ax[-1, c].set_yticks([])


def plot_aggregate_exp(exp_names, label, smoothing=0., plot_objects=True):
    (
        mean, std, times,
        binned_mean, binned_std,
        objects_mean, objects_std
    ) = aggregate_exps(exp_names, return_binned=True)

    ymax = np.array([
        x[~np.isnan(x)].max() for x in binned_mean if len(x) != 0]).max()

    plot_l1_sdf(
        times, mean, binned_mean,
        obj_l1s=objects_mean, label=label, smoothing=smoothing,
        l1_std=std, binned_l1_std=binned_std, obj_l1_std=objects_std
    )

    return ymax


def plot_single_exp(exp_name, label=None,
                    plot_ray_loss=True,
                    plot_vol_loss=True,
                    plot_vxb_loss=False,
                    plot_objects=False,
                    smoothing=0.2,
                    set_ax_bounds=True,
                    draw_kfs=False):
    res = plot_utils.load_res(exp_name)
    times, l1, binned_l1, objects_l1, _ = get_sdf_res(
        res['sdf_eval'])

    ymax = np.array([
        x[~np.isnan(x)].max() for x in binned_l1 if len(x) != 0]).max()
    if set_ax_bounds:
        set_axis_bounds(exp=exp_name, ymax=ymax)

    obj_l1s = None
    if plot_objects and len(objects_l1) != 0:
        obj_l1s = objects_l1

    if plot_ray_loss:
        if plot_vol_loss or plot_vxb_loss:
            label = "Visible region loss"
        plot_l1_sdf(times, l1[0], binned_l1[0],
                    obj_l1s=obj_l1s, label=label, smoothing=smoothing)
    if plot_vol_loss:
        label = "Full volume loss"
        plot_l1_sdf(times, l1[1], binned_l1[1],
                    obj_l1s=obj_l1s, label=label, smoothing=smoothing)
    if plot_vxb_loss:
        label = "Voxblox centers loss"
        plot_l1_sdf(times, l1[2], binned_l1[2],
                    obj_l1s=obj_l1s, label=label, smoothing=smoothing)

    if draw_kfs:
        draw_keyframes(exp_name)

    return ymax


def plot_multiple_exp(exp_names, smoothing=0., plot_objects=False,
                      label_split=-1, labels=None):
    """
        In aggregate experiment we have multiple runs for each sequence.
    """
    ymax = 0
    for i, exp_name in enumerate(exp_names):
        if labels is None:
            label = exp_name.split('/')[label_split]
        else:
            label = labels[i]
        if isinstance(exp_name, list):
            n_ymax = plot_aggregate_exp(
                exp_name,
                label=label,
                smoothing=smoothing,
                plot_objects=plot_objects
            )
        else:
            n_ymax = plot_single_exp(
                exp_name,
                label=label,
                plot_vol_loss=False,
                plot_vxb_loss=False,
                smoothing=smoothing,
                plot_objects=plot_objects,
                set_ax_bounds=False,
                draw_kfs=False,
            )
        ymax = max(ymax, n_ymax)

    exp0 = exp_names[0]
    if isinstance(exp0, list):
        exp0 = exp_names[0][0]
    draw_keyframes(exp0)
    set_axis_bounds(exp=exp0, ymax=ymax)


def plot_multiple_sweeps(dirs, seqs, smoothing=0., labels=None, cost="sdf"):
    print(dirs)
    for d, load_dir in enumerate(dirs):
        print(load_dir)

        exp_names = os.listdir(load_dir)
        exp_names.sort()
        exp_names = [os.path.join(load_dir, d) for d in exp_names]

        check_res = np.array(['res.json' in os.listdir(x) for x in exp_names])
        assert (
            check_res.sum() == check_res.shape[0],
            np.array(exp_names)[~check_res]
        )

        all_losses = []
        for j, seq in enumerate(seqs):
            print("---------------", seq, "---------------")
            use = [seq in x for x in exp_names]
            exps = np.array(exp_names)[use]
            if len(exps) > 1:
                mean, std, times = aggregate_exps(exps, cost=cost)
            else:
                res = plot_utils.load_res(exps[0])
                times, mean, _, _, l1_chomp_costs = get_sdf_res(
                    res['sdf_eval'])
                mean = mean[0]
                if cost == "chomp_1":
                    mean = l1_chomp_costs[:, 0]
                if cost == "chomp_1.5":
                    mean = l1_chomp_costs[:, 1]
                if cost == "chomp_2":
                    mean = l1_chomp_costs[:, 2]
                std = None

            r = j // 3
            c = j % 3
            label = load_dir.split("/")[-1]
            if labels is not None:
                label = labels[d]
            plot_l1_av(times, mean, std, r, c, title=seq,
                       label=label, smoothing=smoothing, cost=cost)

            all_losses.append(mean)
            if d == 0:
                set_axis_bounds(exp=exps[0], row=r, col=c, ymin=2)

        all_losses = np.concatenate((all_losses))
        print("mean loss -------------------------->", all_losses.mean())

    set_axis_bounds(ymax=50)


if __name__ == "__main__":

    smoothing = 0.35

    root_dir = "/home/joe/projects/incSDF/incSDF/isdf/train/examples/experiments/"
    voxblox_root = "/home/joe/projects/incSDF/res/voxblox/gt_traj/0.055/"

    exps = [
        "apt_2_obj_tests/rays/",
    ]

    exps = [os.path.join(root_dir, d) for d in exps]

    voxblox_exps = [
        "apt_2_obj",
    ]
    voxblox_exps = [os.path.join(voxblox_root, d) for d in voxblox_exps]

    exps = exps + voxblox_exps

    plot_objects = True
    draw_kfs = True
    ncols = 4
    if plot_objects:
        ncols = 5
    nrows = 2
    if draw_kfs:
        nrows = 3

    # # Single experiment
    # fig, ax = plt.subplots(
    #     nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    # plot_single_exp(
    #     exps[-2],
    #     plot_vol_loss=False,
    #     plot_vxb_loss=False,
    #     plot_objects=plot_objects,
    #     draw_kfs=draw_kfs,
    # )
    # plt.show()

    # Multiple experiments
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    plot_multiple_exp(exps, smoothing=0.1, plot_objects=plot_objects)
    plt.show()

    # # exp_dir = root_dir + "/kf_replay/"
    # # exp_dir = root_dir + "/sdf_supervision/"
    # # exp_dir = root_dir + "/perception_time/"
    # exp_dir = root_dir + "/trunc_distance/"
    # exp_dir = root_dir + "/sample_kps_gtdepth/"
    # dirs = os.listdir(exp_dir)
    # dirs.sort()
    # dirs = [os.path.join(exp_dir, d) for d in dirs if d[-3:] != "png"]

    # exp_dir = root_dir + "/sample_kps/"
    # dirs1 = os.listdir(exp_dir)
    # dirs1.sort()
    # dirs1 = [os.path.join(exp_dir, d) for d in dirs1 if d[-3:] != "png"]

    # dirs = dirs + dirs1

    # dirs = [dirs[-1]]
    # print(dirs[-2:])

    vx_exp_dir = voxblox_root + "voxel_size/"
    voxblox_dirs = os.listdir(vx_exp_dir)
    voxblox_dirs.sort()
    voxblox_dirs = [os.path.join(vx_exp_dir, d) for d in voxblox_dirs]

    # dirs = dirs + voxblox_dirs

    # # Multiple sweeps
    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    # plot_multiple_sweeps(dirs, smoothing=0.4)
    # plt.show()

    root_dir = "/home/joe/projects/incSDF/incSDF/isdf/train/examples/batch_experiments/"
    seqs = ['apt_2_mnp', 'apt_2_nav', 'apt_2_obj',
            'apt_3_mnp', 'apt_3_nav', 'apt_3_obj']
##
    # Save plots --------------------------------------------------------------
    # smoothing = 0.0

    # exp_dir = root_dir + "/frac_time_perception"
    # percp_time_labels = ["25%", "50%", "75%", "100%"]
    # labels = [*percp_time_labels, *percp_time_labels]
    # vx_percp_dirs = os.listdir(voxblox_root + "/percp_time")
    # vx_percp_dirs.sort()
    # vx_percp_dirs = [voxblox_root + "/percp_time/" + d for d in vx_percp_dirs]
    # vx_percp_dirs = vx_percp_dirs[::-1]
    # # voxblox_bl_orb = voxblox_root + "5cm_orb/"
    # # voxblox_bl_gt = voxblox_root + "5cm_gt/"

    # dirs = os.listdir(exp_dir)
    # dirs.sort()
    # dirs = [os.path.join(exp_dir, d) for d in dirs if d[-3:] != "png"]

    # dirs = dirs + vx_percp_dirs #[voxblox_bl_gt] #+ [voxblox_bl2]

    # # Experiment level plot
    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    # plot_multiple_sweeps(dirs, seqs, smoothing=smoothing, labels=labels)
    # plt.savefig(os.path.join(exp_dir, "sdf.png"))

    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    # plot_multiple_sweeps(
    #     dirs, seqs, smoothing=smoothing, labels=labels, cost="chomp_1")
    # plt.savefig(os.path.join(exp_dir, "chomp_1m.png"))

    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    # plot_multiple_sweeps(
    #     dirs, seqs, smoothing=smoothing, labels=labels, cost="chomp_1.5")
    # plt.savefig(os.path.join(exp_dir, "chomp_1.5m.png"))

    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    # plot_multiple_sweeps(
    #     dirs, seqs, smoothing=smoothing, labels=labels, cost="chomp_2")
    # plt.savefig(os.path.join(exp_dir, "chomp_2m.png"))

    # # Binned plots
    # for seq in seqs:
    #     print("-------------", seq, "-------------")
    #     exps = []
    #     for d in dirs:
    #         dir_exps = os.listdir(d)
    #         dir_exps = [x for x in dir_exps if x[:9] == seq]
    #         dir_exps.sort()
    #         dir_exps = [os.path.join(d, x) for x in dir_exps]
    #         if len(dir_exps) == 1:
    #             dir_exps = dir_exps[0]
    #         exps.append(dir_exps)

    #     fig, ax = plt.subplots(
    #         nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    #     plot_multiple_exp(exps, smoothing=smoothing, plot_objects=plot_objects,
    #                       labels=labels)
    #     plt.savefig(os.path.join(exp_dir, seq + ".png"))
