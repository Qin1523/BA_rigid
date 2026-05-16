# Rigid Template Multi-View Bundle Adjustment

This repository contains a compact C++ workflow for rigid-template reconstruction and pose-only bundle adjustment with three fixed cameras.

The project is split into two executables:

- `generate_rigid_template`: builds a rigid template from multi-frame triangulated samples.
- `BAmotiontrack`: loads a saved template and optimizes one 6-DoF rigid pose per processed frame, with optional gap-frame infill via cubic spline interpolation.

Camera matrices and the template are kept fixed throughout BA. The only optimization variables are the per-frame rigid poses.

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
- A C++17 compiler (uses `<numeric>`, `<thread>`, `<unordered_set>`)
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
- `--motion-track`: enable motion-track constraint
- `--no-motion-track`: disable motion-track constraint
- `--save-init`: save init and warm-start pose/xyz analysis under `<outdir>/debug/`
- `--save-reproj`: save final reprojection diagnostics under `<outdir>/debug/`
- `--lambda <value>`: motion-track weight, default `1.0`
- `--grid-step <value>`: virtual grid spacing in mm, default `10.0`
- `--mt-center <part>`: body-part name to use as motion-track grid center
- `--stride <value>`: process every `N`th observed frame, default `1`
- `--outdir <path>`: output directory, default `results`
- `--help` / `-h`: show help

## Input format

### `cams.txt`

The code reads the first three valid lines. Each valid line must contain 12 numbers describing one `3×4` projection matrix. Empty lines and lines starting with `#` are ignored.

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

Column lookup uses a normalised name matching that accepts common aliases: `bodypart` for `part`; `u`, `v` for `x`, `y`; `time`, `timestamp` for `t`.

### Template CSV

`BAmotiontrack` expects a centered rigid template with this header:

```csv
part,X,Y,Z
nose,23.2,16.5,-2.0
left ear,4.1,9.0,15.0
```

The template coordinates are in the object frame, not world coordinates. The `head center` part must be present and should be at the object origin (i.e., the template must be head-centered).

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

The BA program loads the camera matrices, loads the three observation CSVs, groups them by frame, and then reads the rigid template.

### 2. Estimate segmented translation seeds

The observation window is divided into segments of 150 frames each. For every segment the program finds the first frame that contains a triangulatable `head center` observation (≥2 cameras) and uses its world-space position as the translation seed for that segment.

When initialising the pose for any given frame, the nearest seed by frame number is used. This is substantially more robust than a single global seed for long recordings or recordings with large head-position drift across the session.

### 3. Gap-frame detection and pose infill (path B)

Before running BA, the program infers the regular frame step from the observation data and identifies all *gap frames* — frames that fall within the recording range but have no observations (e.g., frames dropped by the tracking software).

The initialisation sequence is:

1. Build poses for all observed frames (with optional warm start).
2. Run natural cubic spline interpolation over the six pose components independently to generate initial poses for every gap frame.
3. Merge observed-frame and gap-frame poses into one sorted list.
4. Run global BA over the full merged list. Gap frames have no reprojection residuals; they are constrained only by the motion-track term (when enabled).

### 4. Per-frame pose initialization (warm start)

Each processed frame has one 6-parameter pose block:

- `rx, ry, rz`: angle-axis rotation
- `tx, ty, tz`: translation

If `--warm-start` is enabled, the program copies the previous optimized pose forward and runs a short single-frame Ceres solve (up to 5 iterations, `DENSE_QR`, `HuberLoss(1.0)`) for each observed frame before global BA. Frames with fewer than three distinct template parts keep the seed pose.

### 5. Global bundle adjustment

Global BA is solved in one `ceres::Problem` with all frames — observed and gap — included simultaneously.

The objective contains:

- reprojection residuals for all observed frames,
- optional motion-track residuals when `--motion-track` is enabled and at least 5 frames exist.

Gap frames contribute no reprojection residuals but are connected to neighboring frames by motion-track residuals, which pull their poses toward a smooth trajectory.

The solver uses `SPARSE_NORMAL_CHOLESKY` with up to 200 iterations and all available hardware threads.

#### Reprojection term

For each observation:

```text
X_world = R_t X_obj + t_t
u_pred  = project(P_cam, X_world)
residual = u_pred - u_obs
```

Wrapped in `HuberLoss(1.0)`.

#### Motion-track term

For each center frame `t`, the code predicts the center pose from frames `t−2`, `t−1`, `t+1`, `t+2` using four-point cubic Lagrange interpolation with fixed weights:

```text
−1/6,  2/3,  2/3,  −1/6
```

It then evaluates the actual and predicted poses on a virtual `3×3×3` grid in object space and penalises the 3D mismatch, normalised by the head-to-tail-base body length.

### 6. Final 3D output

Final 3D points are reconstructed from the fixed template plus the optimized pose:

```text
X_world = R_t X_obj + t_t
```

No independent 3D point variables are optimized.

## Output files

### `generate_rigid_template`

- `mouse_model_obj.csv`: centered rigid template

### `BAmotiontrack`

Final evaluation outputs always go into `results/` (or `--outdir`).

If motion-track is OFF:

- `results/pose_no_motion.csv`
- `results/xyz_no_motion.csv`
- `results/xyz_restored_no_motion.csv`

If motion-track is ON:

- `results/pose_with_motion.csv`
- `results/xyz_with_motion.csv`
- `results/xyz_restored_with_motion.csv`

Optional init analysis, only when `--save-init` is used:

- `results/debug/init_pose.csv`
- `results/debug/init_xyz.csv`

Optional warm-start analysis, only when `--warm-start --save-init` is used:

- `results/debug/warm_start_pose.csv`
- `results/debug/warm_start_xyz.csv`

Optional final reprojection diagnostics, only when `--save-reproj` is used:

- `results/debug/reproj_frame.csv`
- `results/debug/reproj_frame_cam.csv`
- `results/debug/reproj_frame_part.csv`

Optional motion-track debug output, only when `--motion-track --save-reproj` is used:

- `results/debug/track_frame.csv`

## `pose_*.csv` format

Header:

```csv
frame,t,rx,ry,rz,tx,ty,tz,motion_track
```

The `motion_track` column records whether the motion-track constraint was active during BA (`1`) or not (`0`). This lets downstream scripts distinguish the two conditions even when both result files are combined.

## `xyz_restored_*.csv` format

Header:

```csv
frame,t,part,X,Y,Z,restored,num_obs
```

For every frame in the optimised sequence and every part present in the template, the program:

1. reconstructs the final 3D point from the fixed template and final pose,
2. counts how many original 2D observations that `frame–part` pair had,
3. sets `restored` to indicate the quality of the underlying pose estimate.

`restored` values:

| Value | Meaning |
|-------|---------|
| `0` | All cameras observed this part (full constraint) |
| `1` | Fewer than `num_cams` observations; pose still supported by reprojection residuals from other parts |
| `2` | Gap frame — no observations at all; pose obtained by cubic spline interpolation |

## `track_frame.csv` format

Header:

```csv
frame,t,track_rmse_mm,track_mean_abs_mm,track_max_abs_mm
```

Track errors are in millimetres (object-space units), matching the template coordinate units.

## Implementation notes

- `PoseReprojCost` stores the projection matrix as a plain `double[12]` array to avoid heap allocations and `cv::Mat` overhead inside the Ceres autodiff template.
- `find_column_index()` provides a single normalised-name CSV column lookup used by both `load_obs_csv` and `load_template`.
- `compute_error_stats()` consolidates the RMSE / mean / max calculation shared by all three reprojection-error writers.
- The BA solver thread count is set to `std::thread::hardware_concurrency()` at runtime.

## Current implementation assumptions

- exactly three fixed cameras
- one rigid object pose per processed frame (observed or gap-filled)
- camera matrices are fixed during BA
- the rigid template is fixed during BA
- template generation aggregates raw triangulated world-space samples directly, without per-frame pose alignment
- warm start is optional and off by default
- motion-track is optional and off by default
- reprojection diagnostics are auxiliary and written only when explicitly requested

## Suggested experiments

You can now compare different template strategies while keeping the BA stage unchanged:

- single-frame template
- multi-frame mean template
- multi-frame median template

Typical metrics to compare:

- reprojection RMSE
- 3D RMSE
- missing-point restoration quality (including `restored=2` gap frames)

## License

No license file is included in the cleaned project snapshot.
