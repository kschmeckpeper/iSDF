## Voxblox setup

Follow instructions to install Voxblox [here](https://voxblox.readthedocs.io/en/latest/pages/Installation.html).

We need to add a node to feed Voxblox our dataset and another node to listen to and save the output of voxblox. We save both the ESDF and mesh.

Add python scripts `dataset_feeder.py`, `listener.py` and `eval_wrapper_old.py` to `voxblox/voxblox_ros/scripts` and recompile.
Make sure to change the permissions for the python files you just added with `chmod +x *.py`.

Update the `voxblox_ros/CMakeList.txt` to include the lines:

```
catkin_install_python(PROGRAMS scripts/listener.py scripts/dataset_feeder.py scripts/save_mesh_esdf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

Also modified to output timings...
Check git status in voxblox code


## Running Voxblox Evaluation

Update the directories... Then run:

```
python3 voxblox/eval_wrapper.py
```
