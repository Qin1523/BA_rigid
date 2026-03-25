# Rigid Template Multi-View Bundle Adjustment

This repository contains a compact C++ workflow for rigid-template reconstruction and pose-only bundle adjustment with three fixed cameras.

The project is now split into two executables:

- `generate_rigid_template`: builds a rigid template from multi-frame triangulated samples.
- `BAmotiontrack`: loads a saved template and optimizes one 6-DoF rigid pose per processed frame.

The current code keeps the camera matrices fixed and keeps the template fixed during BA. The optimization variables are only the per-frame rigid poses.

## Minimal project layout

```text
BA_rigid/
├── BAmotiontrack.cpp
├── CMakeLists.txt
├── README.md
├── generate_rigid_template.cpp
└── 3cam_rigid/
    ├── cams.txt
    ├── obs_de_front.csv
    ├── obs_de_side.csv
    └── obs_de_top.csv
```

## Dependencies

- CMake 3.10+
- A C++14 compiler
- Ceres Solver
- OpenCV with `core` and `calib3d`

The `CMakeLists.txt` links the same dependency set for both executables.

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

## Recommended workflow

Step 1: generate the rigid template

```bash
./build/generate_rigid_template
```

Step 2: run bundle adjustment with that template

```bash
./build/BAmotiontrack
```

By default, `BAmotiontrack` runs in evaluation-first mode:

- warm start is OFF unless `--warm-start` is provided,
- reprojection diagnostics are OFF unless `--save-reproj` is provided,
- warm-start analysis files are OFF unless both `--warm-start` and `--save-init` are provided,
- all BA outputs are written under `results/`.

Both executables resolve default input files by checking:

- the current working directory,
- the executable directory and a few parent directories,
- the source-tree root recorded at build time.

That means the default dataset can be found when you run from the repository root, from `build/`, or from common IDE build folders such as `cmake-build-debug/`.

## Command-line arguments

### `generate_rigid_template`

- `--cams <path>`: camera matrix file
- `--front <path>`: front-camera observation CSV
- `--side <path>`: side-camera observation CSV
- `--top <path>`: top-camera observation CSV
- `--output <path>`: output template CSV, default `mouse_model_obj.csv`
- `--fps <value>`: frame rate, default `30.0`
- `--help` / `-h`: show help

### `BAmotiontrack`

- `--cams <path>`: camera matrix file
- `--front <path>`: front-camera observation CSV
- `--side <path>`: side-camera observation CSV
- `--top <path>`: top-camera observation CSV
- `--template <path>`: rigid template CSV, default `mouse_model_obj.csv`
- `--fps <value>`: frame rate, default `30.0`
- `--warm-start`: enable per-frame pose warm start, default OFF
- `--motion-track`: enable motion-track during global BA
- `--no-motion-track`: disable motion-track
- `--save-init`: save warm-start pose/xyz analysis under `results/debug/`
- `--save-reproj`: save final reprojection diagnostics under `results/debug/`
- `--lambda <value>`: motion-track weight, default `1.0`
- `--grid-step <value>`: virtual grid spacing in mm, default `10.0`
- `--stride <value>`: process every `N`th grouped frame, default `1`
- `--help` / `-h`: show help

## Input format

### `cams.txt`

The code reads the first three valid lines. Each valid line must contain 12 numbers describing one `3x4` projection matrix. Empty lines and lines starting with `#` are ignored.

### Observation CSV

Each camera CSV must contain at least `t`, `part`, `x`, and `y`.

Supported formats:

```csv
t,part,x,y
0.033333,nose,245.67,189.34
0.033333,left_ear,198.23,156.78
```

```csv
frame,t,part,x,y
0,0.000000,nose,245.67,189.34
0,0.000000,left_ear,198.23,156.78
```

Fields:

- `frame`: optional explicit frame index
- `t`: timestamp in seconds
- `part`: body-part name
- `x`, `y`: 2D image coordinates

### Template CSV

`BAmotiontrack` expects a centered rigid template with this header:

```csv
part,X,Y,Z
nose,23.2,16.5,-2.0
left ear,4.1,9.0,15.0
```

The template coordinates are in the object frame, not world coordinates.

## How `generate_rigid_template` works

### 1. Load and group observations

The program loads the three camera matrices, loads all observation CSVs, merges them, and groups them by frame.

If the CSV contains an explicit `frame` column, that value is used directly. Otherwise the code falls back to:

```text
frame = round(t * fps)
```

### 2. Collect triangulated samples across frames

For each frame:

- group observations by `part`,
- collect all available camera views for that part,
- require at least two views,
- triangulate one world-space 3D point with linear DLT.

The program stores these world-space 3D candidates in:

```text
part_samples[part] -> [X_1, X_2, ..., X_K]
```

### 3. Aggregate each part robustly

For each part, the current implementation computes a coordinate-wise median:

```text
X_final = median({X_t})
Y_final = median({Y_t})
Z_final = median({Z_t})
```

This produces one aggregated world-space point per part.

### 4. Center the template

After aggregation, the program computes the centroid of all valid part points:

```text
C = mean of all aggregated part points
```

and converts the template into object coordinates:

```text
X_obj = X_world - C
```

### 5. Save the template

The centered template is written to `mouse_model_obj.csv` by default.

## How `BAmotiontrack` works

### 1. Load cameras, observations, and the saved template

The BA program loads the camera matrices, loads the three observation CSVs, groups them by frame, and then reads `mouse_model_obj.csv` as a fixed rigid template.

### 2. Estimate an initial translation seed

Because the template is already centered in object coordinates, the BA no longer receives a world-space centroid from template creation. Instead, it triangulates the available parts in the earliest grouped frame and uses their centroid as the initial translation seed for pose initialization.

### 3. Per-frame pose initialization

Each processed frame has one 6-parameter pose block:

- `rx, ry, rz`: angle-axis rotation
- `tx, ty, tz`: translation

Initialization strategy:

- every processed frame starts from `[0, 0, 0, Cx, Cy, Cz]`, where `C` is the first-frame translation seed,
- if `--warm-start` is enabled, the program then copies the previous optimized pose forward and runs a short single-frame Ceres solve for each frame,
- if `--warm-start` is not enabled, global BA starts directly from the seed poses.

This stage:

- uses all valid observations from all cameras,
- keeps the camera matrices fixed,
- keeps the template fixed,
- optimizes only the current frame pose,
- uses `HuberLoss(1.0)`,
- uses `DENSE_QR` with up to 5 iterations.

If fewer than three distinct template parts are available in a frame, the code skips the single-frame solve and keeps the seed.

### 4. Global bundle adjustment

Global BA is solved in one `ceres::Problem`.

The objective contains:

- all reprojection residuals for all processed frames,
- optional motion-track residual blocks when `--motion-track` is enabled and at least 5 processed frames exist.

#### Reprojection term

For each observation:

```text
X_world = R_t X_obj + t_t
u_pred = project(P_cam, X_world)
residual = u_pred - u_obs
```

The reprojection residuals use `HuberLoss(1.0)`.

#### Motion-track term

For each center frame `t`, the code predicts the center pose from frames `t-2`, `t-1`, `t+1`, and `t+2` using four-point cubic interpolation with fixed weights:

```text
-1/6, 2/3, 2/3, -1/6
```

It then evaluates the actual and predicted poses on a virtual `3 x 3 x 3` grid in object space:

```text
g = (ix * grid_step, iy * grid_step, iz * grid_step)
ix, iy, iz in {-1, 0, 1}
```

and penalizes the 3D mismatch:

```text
sqrt(lambda) * (X_actual - X_pred)
```

This is solved jointly with the reprojection residuals, not in a separate pass.

### 5. Final 3D output

The final 3D points are reconstructed from the fixed template plus the optimized pose:

```text
X_world = R_t X_obj + t_t
```

The code does not optimize independent 3D point variables per frame.

## Output files

### `generate_rigid_template`

- `mouse_model_obj.csv`: centered rigid template

### `BAmotiontrack`

Final evaluation outputs always go into `results/`.

If motion-track is OFF:

- `results/pose_rigid_no_motion.csv`
- `results/xyz_rigid_no_motion.csv`

If motion-track is ON:

- `results/pose_rigid_with_motion.csv`
- `results/xyz_rigid_with_motion.csv`
- `results/xyz_restored.csv`

Optional warm-start analysis, only when `--warm-start --save-init` is used:

- `results/debug/warm_start_pose.csv`
- `results/debug/warm_start_xyz.csv`

Optional final reprojection diagnostics, only when `--save-reproj` is used:

- `results/debug/reproj_frame.csv`
- `results/debug/reproj_frame_cam.csv`
- `results/debug/reproj_frame_part.csv`

Optional motion-track debug output, only when `--motion-track --save-reproj` is used:

- `results/debug/track_frame.csv`

## `xyz_restored.csv`

Header:

```csv
frame,t,part,X,Y,Z,restored,num_obs
```

For every processed frame and every part present in the template, the program:

1. reconstructs the final 3D point from the fixed template and final pose,
2. counts how many original 2D observations that `frame-part` pair had,
3. flags whether the point was fully observed or depends on rigid-template restoration.

Current rule with three cameras:

- `num_obs = 3` -> `restored = 0`
- `num_obs = 2`, `1`, or `0` -> `restored = 1`

## Current implementation assumptions

- exactly three fixed cameras
- one rigid object pose per processed frame
- the BA keeps camera matrices fixed
- the BA keeps the rigid template fixed
- template generation aggregates raw triangulated world-space samples directly, without per-frame pose alignment
- warm start is optional and off by default unless `--warm-start` is provided
- motion-track is optional and off by default unless `--motion-track` is provided
- reprojection diagnostics are auxiliary and are only written when explicitly requested

Because template generation aggregates world-space triangulations directly, it is best suited to cases where pose variation is limited or where that approximation is acceptable for the intended experiment.

## Suggested experiments

You can now compare different template strategies while keeping the BA stage unchanged:

- single-frame template
- multi-frame mean template
- multi-frame median template

Typical metrics to compare:

- reprojection RMSE
- 3D RMSE
- missing-point restoration quality

## License

No license file is included in the cleaned project snapshot.
