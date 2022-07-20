import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pylab as plt

import vxblx_utils


# Set experiment params ----------------------------------------------

root_save = "/home/joe/projects/incSDF/res/voxblox/"

seq = "room1"
gt_mesh_file = f"/mnt/sda/Replica-seqs/{seq}/mesh.ply"
im_indices = [0, 10, 20, 50, 100, 200, 300, 400,
              500, 1000, 892, 1914, 1279, 168, 1133,
              845, 506, 374, 824, 504]

save_interval = 2.
update_esdf_every_n_sec = 0.5

fps = 1.0

voxel_sizes = [0.02, 0.05, 0.1, 0.2]

exp_params = []
for voxel_size in voxel_sizes:

    now = datetime.now()
    exp_name = now.strftime("%m-%d-%y_%H-%M-%S-%f")
    save_info_dir = os.path.join(root_save, exp_name)
    save_dir = save_info_dir + "/out/"

    params = {
        "exp_name": exp_name,
        "voxel_size": voxel_size,
        "save_info_dir": save_info_dir,
        "save_dir": save_dir,
        "seq": seq,
        "gt_mesh_file": gt_mesh_file,
        "fps": fps,
        "save_interval": save_interval,
        "update_esdf_every_n_sec": update_esdf_every_n_sec,
        "im_indices": im_indices
    }

    exp_params.append(params)


# Create launchfile and run ros ---------------------------

for params in exp_params:

    vxblx_utils.run_exp(params)

print("\n\nDone experiment!")


# Do evaluations -------------------------------------------
print("\n\nStarting eval")

for i, params in enumerate(exp_params):

    res = {}
    res = vxblx_utils.get_timings(params, res)
    # Save evaluation results
    with open(os.path.join(params['save_info_dir'], "res.json"), "w") as f:
        json.dump(res, f, indent=4)

    exp_params[i]['res'] = res

# Do plots -------------------------------------------

tsdf_av_times = []
esdf_mean_times = []
esdf_max_times = []

for params in exp_params:

    tsdf_av_times.append(np.mean(params['res']['times']['tsdf_times']))
    esdf_mean_times.append(params['res']['times']['esdf_mean_time'])
    esdf_max_times.append(params['res']['times']['esdf_max_time'])


plt.plot(voxel_sizes, tsdf_av_times,
         label='mean tsdf integration time', marker="x")
plt.plot(voxel_sizes, esdf_mean_times,
         label='mean esdf update time', marker="x")
plt.plot(voxel_sizes, esdf_max_times,
         label='max esdf update time', marker="x")
plt.plot(voxel_sizes, np.array(esdf_mean_times) + np.array(tsdf_av_times),
         label='mean esdf + tsdf update time', marker="x")

plt.title("Voxblox timings")
plt.legend()
plt.yscale('log')
plt.show()

import ipdb; ipdb.st_trace()
