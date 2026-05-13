# ROS2 Vision Navigator

An autonomous navigation system built on ROS2 that uses an on-device quantized
ML model to detect obstacles from camera frames and emit navigation decisions
(`STOP` / `LEFT` / `RIGHT` / `FORWARD`). The entire pipeline runs locally — no
cloud inference, no simulator, no special hardware.

## Why this project

Most autonomous-stack demos depend on Gazebo or a real robot. This one focuses
on the parts that matter for embedded autonomy:

- **ROS2 message-passing** between three independent nodes
- **On-device ML inference** with an INT8-quantized model (TensorFlow Lite)
- **Deterministic decision logic** that turns detections into motion commands

The same architecture would drop onto a Jetson, a Raspberry Pi, or a Qualcomm
RB5 with only the camera source swapped.

## Status

Living checklist — the source of truth for what works, what's a stub, and
what's missing. Updated as the project moves forward.

**Pipeline nodes**
- [x] `camera_node` — loads images from disk, publishes `sensor_msgs/Image` at 10 Hz
- [ ] `detector_node` — TFLite inference on `/camera/image` → `/detections` *(stub)*
- [ ] `navigator_node` — decision logic on `/detections` → `/cmd_nav` *(stub)*

**Assets**
- [ ] `ros2_ws/src/vision_navigator/test_images/` — sample input frames
- [ ] `ros2_ws/src/vision_navigator/models/model_quant.tflite` — INT8 model
- [ ] `scripts/quantize_model.py` — Float32 → INT8 conversion

**Local environment**
- [x] Conda env `ros2_vision` (Python 3.11) with numpy + opencv
- [ ] TensorFlow / tflite-runtime (deferred until detector is real)
- [x] ROS2 Humble via RoboStack (`ros-humble-desktop` + `colcon-common-extensions`)
- [x] `colcon build --symlink-install` verified (`setup.cfg` routes entry points to `lib/vision_navigator/` for `ros2 run`)

**Verification**
- [x] All node files compile (`python -m py_compile`)
- [x] `ros2 run vision_navigator camera_node` launches under ROS2 (idles with no images — expected)
- [ ] End-to-end run: camera → detector → navigator publishes `/cmd_nav`

## Architecture

```
+----------------+    /camera/image    +-----------------+    /detections    +-----------------+    /cmd_nav
|  camera_node   | ------------------> |  detector_node  | ----------------> |  navigator_node | -----------> (robot)
+----------------+                     +-----------------+                   +-----------------+
   reads frames                          runs quantized                        applies decision
   from disk                             TFLite model                          logic
```

Three nodes, three topics. Each node is independently testable and can be
replaced (e.g. swap `camera_node` for a real USB camera driver) without
touching the others.

### Nodes

| Node             | Subscribes      | Publishes        | Responsibility                                    |
|------------------|-----------------|------------------|---------------------------------------------------|
| `camera_node`    | —               | `/camera/image`  | Streams test images at 10 Hz as `sensor_msgs/Image` |
| `detector_node`  | `/camera/image` | `/detections`    | Runs INT8 TFLite inference, emits bounding boxes  |
| `navigator_node` | `/detections`   | `/cmd_nav`       | Maps detections to `STOP`/`LEFT`/`RIGHT`/`FORWARD` |

## Project layout

```
ros2-vision-navigator/
├── ros2_ws/
│   └── src/
│       └── vision_navigator/
│           ├── vision_navigator/
│           │   ├── camera_node.py       # Publishes frames to /camera/image
│           │   ├── detector_node.py     # Quantized inference -> /detections
│           │   └── navigator_node.py    # Decision logic -> /cmd_nav
│           ├── models/
│           │   └── model_quant.tflite   # INT8-quantized model
│           ├── test_images/             # Sample input frames
│           ├── package.xml
│           └── setup.py
├── scripts/
│   └── quantize_model.py                # Float32 -> INT8 conversion
├── requirements.txt
└── README.md
```

## Tech stack

- **ROS2 Humble** (runs on macOS via RoboStack/conda; Linux native otherwise. Jazzy requires native arm64 conda — Humble is what RoboStack ships for `osx-64`.)
- **Python 3.11**
- **TensorFlow Lite** for on-device inference
- **OpenCV** for image loading and preprocessing
- **NumPy** for tensor manipulation

## Setup

```bash
# 1. Create a conda environment with ROS2 Humble
conda create -n ros2_vision python=3.11
conda activate ros2_vision
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --set channel_priority strict
conda install ros-humble-desktop colcon-common-extensions

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build the workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Running

Open three terminals (each one sourced with `install/setup.bash`):

```bash
# Terminal 1 — publish frames
ros2 run vision_navigator camera_node

# Terminal 2 — run detection
ros2 run vision_navigator detector_node

# Terminal 3 — emit navigation commands
ros2 run vision_navigator navigator_node
```

Watch the decisions stream by:

```bash
ros2 topic echo /cmd_nav
```

## Model quantization

`scripts/quantize_model.py` converts a Float32 Keras/TF model into an INT8
TFLite model using post-training quantization with a representative dataset.
INT8 cuts model size by ~4x and gives a meaningful inference-time speedup on
ARM CPUs and NPUs.

```bash
python scripts/quantize_model.py \
    --input models/model_fp32.h5 \
    --output ros2_ws/src/vision_navigator/models/model_quant.tflite \
    --representative-dir test_images/
```
