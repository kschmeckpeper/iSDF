## Setup

In the parent incSDF directory, clone the repository: 
```
git clone https://github.com/joeaortiz/sdf_fusion.git
```

### Requirements

- ROS melodic
- GTSAM

### Compile

```
cd sdf_fusion
mkdir build
cd build
sudo make install
```

## Running GPU SDF fusion evaluation

Update the directories to your own at the top of [run.py](https://github.com/joeaortiz/incSDF/blob/main/gpu_fusion/run.py).

Then in the parent incSDF directory run:

```
mkdir res
mkdir res/gpu_fusion
cd incSDF
conda activate incSDF
python gpu_fusion/run.py
```
