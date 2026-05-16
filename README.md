# Rigid Template Multi-View Bundle Adjustment

This repository contains a compact C++ workflow for rigid-template reconstruction and pose-only bundle adjustment with three fixed cameras.

The project is now split into three executables:

- `generate_rigid_template`: builds a rigid template from multi-frame triangulated samples.
- `BAmotiontrack`: loads a saved template and optimizes one 6-DoF rigid pose per processed frame, with optional gap-frame infill via cubic spline interpolation.
- `evaluation_3d`: audits GT against up to four prediction variants: rigid/restored crossed with no-motion/with-motion.


Camera matrices and the template are kept fixed throughout BA. The only optimization variables are the per-frame rigid poses.

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
- A C++17 compiler (uses `<numeric>`, `<thread>`, `<unordered_set>`)
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

The code reads the first three valid lines. Each valid line must contain 12 numbers describing one `3×4` projection matrix. Empty lines and lines starting with `#` are ignored.

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

Column lookup uses a normalised name matching that accepts common aliases: `bodypart` for `part`; `u`, `v` for `x`, `y`; `time`, `timestamp` for `t`.

### Template CSV

`BAmotiontrack` expects a head-centered, body-axis aligned rigid template with this header:

```csv
part,X,Y,Z
nose,23.2,16.5,-2.0
left ear,4.1,9.0,15.0
```

The template coordinates are in the object frame, not world coordinates. The `head center` part must be present and should be at the object origin (i.e., the template must be head-centered).

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

- `mouse_model_obj.csv`: head-centered, body-axis aligned rigid template
- `mouse_model_obj.html`: interactive Plotly 3D skeleton view of the final template

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
- missing-point restoration quality (including `restored=2` gap frames)

## License

No license file is included in the cleaned project snapshot.
