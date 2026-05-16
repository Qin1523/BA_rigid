# Rigid Template Multi-View Bundle Adjustment

This repository contains a compact C++ workflow for rigid-template reconstruction and pose-only bundle adjustment with three fixed cameras.

The project is now split into three executables:

- `generate_rigid_template`: builds a rigid template from multi-frame triangulated samples.
- `BAmotiontrack`: loads a saved template and optimizes one 6-DoF rigid pose per processed frame.
- `evaluation_3d`: audits GT against up to four prediction variants: rigid/restored crossed with no-motion/with-motion.

The current code keeps the camera matrices fixed and keeps the template fixed during BA. The optimization variables are only the per-frame rigid poses.

## Minimal project layout

```text
BA_rigid/
├── BAmotiontrack.cpp
├── CMakeLists.txt
├── README.md
├── evaluation_3d.cpp
├── generate_rigid_template.cpp
└── 3cam/
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

The `CMakeLists.txt` links the same dependency set for all executables.

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

Step 3: evaluate final 3D trajectories

```bash
./build/evaluation_3d
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
- `--adaptive-k <value>`: MAD multiplier for adaptive reprojection filtering, default `2.5`
- `--angle-thresh <value>`: minimum triangulation angle in deg, default `10.0`
- `--outlier-mult <value>`: per-part spatial outlier multiplier, default `3.0`
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
- `--save-init`: save `init_*` analysis and optional `warm_start_*` analysis under `<outdir>/debug/`
- `--save-reproj`: save final reprojection diagnostics under `<outdir>/debug/`
- `--lambda <value>`: motion-track weight, default `1.0`
- `--grid-step <value>` / `--grid_step <value>`: virtual grid spacing in mm, default `10.0`
- `--mt-center <part>`: optional body part used as the motion-track grid center
- `--stride <value>`: process every `N`th grouped frame, default `1`
- `--outdir <path>`: output directory, default `results`
- `--help` / `-h`: show help

### `evaluation_3d`

- `--gt <path>`: GT trajectory CSV, default `xyz_gt.csv`
- `--rigid-no-motion <path>`: rigid no-motion trajectory CSV, default `results/xyz_rigid_no_motion.csv`
- `--rigid-with-motion <path>`: rigid with-motion trajectory CSV, default `results/xyz_rigid_with_motion.csv`
- `--restored-no-motion <path>`: restored no-motion trajectory CSV, default `results/xyz_restored_no_motion.csv`
- `--restored-with-motion <path>`: restored with-motion trajectory CSV, default `results/xyz_restored_with_motion.csv`
- `--no-motion <path>`: compatibility alias for `--restored-no-motion`
- `--motion <path>` / `--with-motion <path>`: compatibility aliases for `--restored-with-motion`
- `--outdir <path>` / `--output-dir <path>`: output directory for evaluation CSVs and plots, default `results`
- `--obs-front <path>`: front-camera observation CSV for observation-missing visualization, default `3cam/obs_de_front.csv`
- `--obs-side <path>`: side-camera observation CSV for observation-missing visualization, default `3cam/obs_de_side.csv`
- `--obs-top <path>`: top-camera observation CSV for observation-missing visualization, default `3cam/obs_de_top.csv`
- `--fps <value>`: FPS used when observation CSVs need `t -> frame` conversion, default `30.0`
- `--missing-start <frame>`: optional missing-window start frame
- `--missing-end <frame>`: optional missing-window end frame
- `--help` / `-h`: show help

The evaluator is GT-driven:

- it iterates over GT `(frame, part)` keys,
- records `matched`, `missing_prediction`, and diagnostic `extra_prediction` cases,
- reports `matched_rmse_3d`, `coverage`, `missing_rate`, and `penalized_rmse_3d`,
- computes the missing-point penalty automatically as `0.7 * GT body scale` from the GT bounding box,
- uses a robust frame high-error threshold `max(median + 3 * MAD, 10mm)`,
- and writes point/frame/segment audit tables for error analysis, including segment-level part summaries with separate missing/error counts.

In addition, when the three observation CSVs are available, it writes a separate observation-level missing visualization module:

- `obs_camera_summary.csv`: per-frame front/side/top coverage and missing-part counts
- `obs_missing_per_part.csv`: per-frame, per-camera, per-part missing mask
- `camera_coverage_vs_time.png`: camera-level coverage curves with bottom `Time (s)` and top `Frame`
- `camera_missing_count_vs_time.png`: camera-level missing-count curves with the same dual x-axis
- `front_missing_heatmap.png`, `side_missing_heatmap.png`, `top_missing_heatmap.png`: per-camera missing heatmaps across frames and body parts

This observation module is diagnostic only. It does not change the 3D `coverage` metric reported by the GT-driven evaluator.

It accepts the requested `frame,t,part,X,Y,Z` layout and the current repository GT header `frame,time,part,x_r,y_r,z_r`.
If `results/xyz_restored_with_motion.csv` is absent, the evaluator also accepts the legacy path `results/xyz_restored_motion.csv`.

## Input format

### `cams.txt`

The code reads the first three valid lines. Each valid line must contain 12 numbers describing one `3x4` projection matrix. Empty lines and lines starting with `#` are ignored.

### Observation CSV

Each camera CSV must contain at least `t`, `part`, `x`, and `y`.

Expected format:

```csv
t,part,x,y
0.033333,nose,245.67,189.34
0.033333,left_ear,198.23,156.78
```

Fields:

- `t`: timestamp in seconds
- `part`: body-part name
- `x`, `y`: 2D image coordinates

### Template CSV

`BAmotiontrack` expects a head-centered, body-axis aligned rigid template with this header:

```csv
part,X,Y,Z
nose,23.2,16.5,-2.0
left ear,4.1,9.0,15.0
```

The template coordinates are in the object frame, not world coordinates.

## How `generate_rigid_template` works

### 1. Load and group observations

The program loads the three camera matrices, loads all observation CSVs, merges them, and groups them by frame using:

```text
frame = round(t * fps)
```

### 2. Collect triangulated samples across frames

The sample collection is split into two filtering passes.

First pass, for each frame and part:

- group observations by `part`,
- collect all available camera views for that part,
- require at least two views,
- triangulate one world-space 3D point with linear DLT,
- reject the sample if its maximum triangulation angle across participating camera pairs is below the default threshold (`10.0` deg),
- keep the surviving 3D points and their reprojection errors temporarily in world coordinates.

After that first pass, the code estimates an adaptive reprojection threshold from all geometry-filtered candidates:

```text
sigma_hat = MAD(reproj_error - median(reproj_error)) / 0.6745
reproj_thresh = median(reproj_error) + adaptive_k * sigma_hat
```

Here:

- `MAD` is the median absolute deviation of the reprojection errors around their median.
- `0.6745` is the Gaussian consistency constant. For a normal distribution, `MAD = 0.67448975... * sigma`, so dividing by `0.6745` converts `MAD` into a robust standard-deviation-scale estimate `sigma_hat`.
- `adaptive_k` is a tunable threshold factor, not a fixed statistical constant. The default `2.5` is an engineering tradeoff between over-rejecting valid samples and letting high-error samples through.
- smaller `adaptive_k` values such as `2.0` are stricter and reject more samples,
- `adaptive_k = 2.5` is the current default and a balanced setting,
- larger values such as `3.0` are looser and keep more high-error samples.

If the MAD estimate degenerates, the code falls back to a median-based threshold instead of forcing an over-tight cutoff.

Second pass, per frame:

- reject points whose mean reprojection error is above the adaptive threshold,
- require `head center` and `tail base`,
- choose a left-right reference from ears first, then front paws, then hind paws,
- set the frame origin to `head center`,
- define the body y-axis as `head center -> tail base`,
- orthogonalize the left-right reference to obtain a consistent x-axis,
- compute the z-axis from the cross product,
- transform every accepted point into the frame-aligned body-local coordinates:

```text
origin = head_center
y_axis = normalize(tail_base - head_center)
X_local = [
  dot(X_world - origin, x_axis),
  dot(X_world - origin, y_axis),
  dot(X_world - origin, z_axis)
]
```

The program stores these body-aligned local 3D candidates in:

```text
part_samples[part] -> [X_1, X_2, ..., X_K]
```

### 3. Aggregate each part robustly

For each part, the program first applies a spatial outlier filter in the body-aligned local sample cloud:

- compute a provisional coordinate-wise median center,
- compute the Euclidean distance from every sample to that center,
- compute the median of those distances,
- estimate a robust spread from the distance MAD,
- keep samples within `median_distance + 3.0 * sigma_distance`,
- if the MAD estimate degenerates, keep the original samples instead of over-filtering.

After that filtering stage, it computes the final coordinate-wise median:

```text
X_final = median({X_t})
Y_final = median({Y_t})
Z_final = median({Z_t})
```

This produces one aggregated body-aligned local point per part.

### 4. Final template cleanup

No final centroid recenter is applied. The template remains head-centered and body-axis aligned.

### 5. Save the template

The resulting template is written to `mouse_model_obj.csv` by default, and rows are saved in the fixed `part_order` derived from the input observations.

The program also writes an interactive Plotly skeleton view with the same basename and a `.html` extension, for example:

- `mouse_model_obj.csv`
- `mouse_model_obj.html`

## How `BAmotiontrack` works

### 1. Load cameras, observations, and the saved template

The BA program loads the camera matrices, loads the three observation CSVs, groups them by frame, and then reads `mouse_model_obj.csv` as a fixed rigid template.

### 2. Estimate an initial translation seed

Because the template origin is `head center`, the BA triangulates `head center` in the earliest grouped frame and uses that world-space position as the initial translation seed.

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

- `mouse_model_obj.csv`: head-centered, body-axis aligned rigid template
- `mouse_model_obj.html`: interactive Plotly 3D skeleton view of the final template

### `BAmotiontrack`

Final evaluation outputs always go into `results/`.

If motion-track is OFF:

- `results/pose_rigid_no_motion.csv`
- `results/xyz_rigid_no_motion.csv`
- `results/xyz_restored_no_motion.csv`

If motion-track is ON:

- `results/pose_rigid_with_motion.csv`
- `results/xyz_rigid_with_motion.csv`
- `results/xyz_restored_motion.csv`

Optional initialization analysis, when `--save-init` is used:

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

### `evaluation_3d`

Default evaluation outputs are written under `results/`:

- `results/eval_3d_overall.csv`
- `results/eval_3d_frame.csv`
- `results/eval_3d_frame_audit.csv`
- `results/eval_3d_part.csv`
- `results/eval_3d_point_audit.csv`
- `results/eval_3d_segment.csv`
- `results/eval_3d_segment_audit.csv`
- `results/rmse_3d_vs_time_rigid.png`
- `results/rmse_3d_vs_time_restored.png`
- `results/coverage_vs_time_rigid.png`
- `results/coverage_vs_time_restored.png`
- `results/penalized_rmse_3d_vs_time_rigid.png`
- `results/penalized_rmse_3d_vs_time_restored.png`
- `results/rmse_3d_per_part.png`
- `results/segment_summary_rigid.png`
- `results/segment_summary_restored.png`
- `results/obs_camera_summary.csv`
- `results/obs_missing_per_part.csv`
- `results/camera_coverage_vs_time.png`
- `results/camera_missing_count_vs_time.png`
- `results/front_missing_heatmap.png`
- `results/side_missing_heatmap.png`
- `results/top_missing_heatmap.png`

## `xyz_restored_*.csv`

Header:

```csv
frame,t,part,X,Y,Z,restored,num_obs
```

For every processed frame and every part present in the template, the program writes the same restored CSV format for both no-motion and motion-track runs:

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
- template generation uses multi-frame triangulation with reprojection filtering, triangulation-angle filtering, head-centered frame-wise body-axis alignment, and per-part spatial outlier filtering
- by default the exported template remains head-centered and body-axis aligned, without final centroid recenter
- the BA initial translation seed is estimated from `head center` only
- motion-track uses the template centroid as its default grid center, unless `--mt-center` is provided
- warm start is optional and off by default unless `--warm-start` is provided
- motion-track is optional and off by default unless `--motion-track` is provided
- reprojection diagnostics are auxiliary and are only written when explicitly requested

Because template generation uses a head-centered body frame with a head-to-tail principal axis and a left-right reference, it removes translation and a large part of the pose variation without requiring iterative rigid registration. It is still a single-frame analytic alignment rather than a full template-registration pipeline.

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
