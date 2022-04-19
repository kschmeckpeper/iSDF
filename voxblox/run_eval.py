import os
import argparse
import trimesh
import json

import vxblx_utils


def main(params_file):

    with open(params_file, 'r') as f:
        params = json.load(f)

    # To handle new directory name
    params["save_dir"] = "/".join(params_file.split('/')[:-1])

    print("loading mesh...")
    gt_mesh = trimesh.load(params["gt_mesh_file"])
    print("loaded mesh")

    try:

        sdf_eval_res = vxblx_utils.eval_sdf(
            params,
            gt_mesh,
            vis_loss=False,
            vis_mapped_region=False,
            save_eval_pts=True,
        )

        res = {
            "sdf_eval": sdf_eval_res,
        }

        # Save evaluation results
        print("Saving res!")
        with open(os.path.join(params['save_dir'], "res.json"), "w") as f:
            json.dump(res, f, indent=4)

    except:
        raise ValueError(
            "\n\n\n ---------------Failed eval!!!--------------\n\n\n"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, help="params file")
    args = parser.parse_args()

    params_file = args.params_file

    main(params_file)
