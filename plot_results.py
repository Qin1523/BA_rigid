#!/usr/bin/env python3
"""
plot_results.py  --  独立绘图脚本，读取 evaluation_3d 和 BAmotiontrack 的 CSV 输出绘图。

用法：
  python plot_results.py --results_dir results --out_dir figures/
  python plot_results.py --only observation ...
  python plot_results.py --only evaluation ...
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def parse_int(value):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


method_order = [
    "rigid_no_motion",
    "rigid_with_motion",
    "restored_no_motion",
    "restored_with_motion",
]


def active_methods(rows):
    present = {row["method"] for row in rows if row.get("method")}
    return [method for method in method_order if method in present]


def save_empty_plot(path, title, message):
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_frame_time_map(rows):
    mapping = {}
    for row in rows:
        frame = parse_int(row.get("frame"))
        if frame is None:
            frame = parse_int(row.get("frame_id"))
        t = parse_float(row.get("t"))
        if t is None:
            t = parse_float(row.get("time_s"))
        if frame is None:
            continue
        if t is None:
            t = float(frame)
        mapping[frame] = t
    return mapping


def frames_to_times(x, mapping):
    arr = np.asarray(x, dtype=float)
    if not mapping:
        return arr
    keys = np.array(sorted(mapping.keys()), dtype=float)
    vals = np.array([mapping[int(k)] for k in keys], dtype=float)
    return np.interp(arr, keys, vals)


def times_to_frames(t, mapping):
    arr = np.asarray(t, dtype=float)
    if not mapping:
        return arr
    items = sorted(mapping.items())
    frames = np.array([float(k) for k, _ in items], dtype=float)
    times = np.array([float(v) for _, v in items], dtype=float)
    keep = np.concatenate(([True], np.diff(times) > 0))
    times = times[keep]
    frames = frames[keep]
    if times.size == 0:
        return arr
    if times.size == 1:
        return np.full_like(arr, frames[0], dtype=float)
    return np.interp(arr, times, frames)


def add_frame_axis(ax, frame_time_map):
    if not frame_time_map:
        return
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda t: times_to_frames(t, frame_time_map),
            lambda f: frames_to_times(f, frame_time_map),
        ),
    )
    secax.set_xlabel("Frame")


def choose_tick_positions(n):
    if n <= 0:
        return []
    if n <= 8:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=8, dtype=int)
    return sorted(set(int(x) for x in raw))


def frame_to_time(x, mapping):
    if not mapping:
        return x
    keys = sorted(mapping.keys())
    if x <= keys[0]:
        return mapping[keys[0]]
    if x >= keys[-1]:
        return mapping[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= x <= keys[i + 1]:
            f0, f1 = keys[i], keys[i + 1]
            t0, t1 = mapping[f0], mapping[f1]
            if f1 == f0:
                return t0
            alpha = (x - f0) / (f1 - f0)
            return t0 + alpha * (t1 - t0)
    return x


def infer_full_frames(frame_rows, anomaly_rows):
    frames_in_order = []
    seen_frames = set()
    for row in frame_rows:
        frame = parse_int(row.get("frame"))
        if frame is None or frame in seen_frames:
            continue
        seen_frames.add(frame)
        frames_in_order.append(frame)
    if frames_in_order:
        return frames_in_order

    anomaly_frames = sorted(
        {
            parse_int(row.get("frame_id"))
            for row in anomaly_rows
            if parse_int(row.get("frame_id")) is not None
        }
    )
    if not anomaly_frames:
        return []
    if len(anomaly_frames) == 1:
        return anomaly_frames

    diffs = [
        anomaly_frames[i + 1] - anomaly_frames[i]
        for i in range(len(anomaly_frames) - 1)
        if anomaly_frames[i + 1] > anomaly_frames[i]
    ]
    frame_step = min(diffs) if diffs else 1
    return list(range(anomaly_frames[0], anomaly_frames[-1] + frame_step, frame_step))


def complete_frame_time_map(frames_in_order, base_mapping):
    if not frames_in_order:
        return {}
    if not base_mapping:
        return {frame: float(frame) for frame in frames_in_order}
    completed = {}
    interpolated = frames_to_times(frames_in_order, base_mapping)
    for frame, t in zip(frames_in_order, interpolated):
        completed[frame] = float(t)
    return completed


labels = {
    "rigid_no_motion": "Rigid No Motion",
    "rigid_with_motion": "Rigid With Motion",
    "restored_no_motion": "Restored No Motion",
    "restored_with_motion": "Restored With Motion",
}
colors = {
    "rigid_no_motion": "#1f77b4",
    "rigid_with_motion": "#ff7f0e",
    "restored_no_motion": "#7F8C8D",
    "restored_with_motion": "#D64933",
}
restored_hatches = {0: "///", 1: "xx", 2: ".."}


def plot_evaluation(results_dir, out_dir):
    def build_restored_spans(frame_rows, mapping):
        frames_by_method = {}
        for row in frame_rows:
            method = row.get("method")
            frame = parse_int(row.get("frame"))
            restored = parse_int(row.get("restored"))
            if method is None or frame is None or restored != 2:
                continue
            frames_by_method.setdefault(method, set()).add(frame)

        spans = {}
        for method, frame_set in frames_by_method.items():
            ordered = sorted(frame_set)
            if not ordered:
                continue
            merged = []
            start = ordered[0]
            prev = ordered[0]
            for frame in ordered[1:]:
                if frame == prev + 1:
                    prev = frame
                    continue
                merged.append((start, prev))
                start = frame
                prev = frame
            merged.append((start, prev))
            spans[method] = [
                (frame_to_time(start - 0.5, mapping), frame_to_time(end + 0.5, mapping))
                for start, end in merged
            ]
        return spans

    def add_restored_overlays(ax, methods, restored_spans):
        for method in methods:
            for start_t, end_t in restored_spans.get(method, []):
                ax.axvspan(
                    start_t,
                    end_t,
                    color=colors.get(method, "#999999"),
                    alpha=0.08,
                    linewidth=0,
                    zorder=0,
                )

    frame_rows = read_csv(os.path.join(results_dir, "eval_3d_frame_audit.csv"))
    frame_time_map = build_frame_time_map(frame_rows)
    restored_spans = build_restored_spans(frame_rows, frame_time_map)
    methods = active_methods(frame_rows)

    if not methods:
        methods = method_order

    matched_rmse_data = {method: [] for method in methods}
    coverage_data = {method: [] for method in methods}
    for row in frame_rows:
        method = row.get("method")
        if method not in matched_rmse_data:
            continue
        t = parse_float(row.get("t"))
        matched_rmse = parse_float(row.get("matched_rmse_3d"))
        coverage = parse_float(row.get("coverage"))
        if t is not None and matched_rmse is not None:
            matched_rmse_data[method].append((t, matched_rmse))
        if t is not None and coverage is not None:
            coverage_data[method].append((t, coverage))

    def plot_time_series_overlay(data_map, ylabel, title, output_name):
        method_draw_order = [
            "rigid_no_motion",
            "rigid_with_motion",
            "restored_no_motion",
            "restored_with_motion",
        ]
        has_data = any(data_map.get(method) for method in method_draw_order)
        if not has_data:
            save_empty_plot(os.path.join(out_dir, output_name), title, "No data available")
            return

        plt.figure(figsize=(10.5, 4.8))
        ax = plt.gca()
        add_restored_overlays(ax, method_draw_order, restored_spans)
        style_map = {
            "rigid_no_motion": {"linestyle": "--", "marker": "o", "linewidth": 1.5, "markersize": 2.6, "alpha": 0.72},
            "rigid_with_motion": {"linestyle": "-", "marker": "s", "linewidth": 1.5, "markersize": 2.6, "alpha": 0.72},
            "restored_no_motion": {"linestyle": "--", "linewidth": 1.8, "alpha": 0.7},
            "restored_with_motion": {"linestyle": "-", "linewidth": 1.0, "alpha": 0.9},
        }

        for zorder, method in enumerate(method_draw_order, start=1):
            if method not in data_map:
                continue
            series = sorted(data_map[method])
            if not series:
                continue
            times = [item[0] for item in series]
            values = [item[1] for item in series]
            ax.plot(
                times,
                values,
                label=labels.get(method, method),
                color=colors.get(method, "#888888"),
                zorder=zorder,
                **style_map.get(method, {}),
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        add_frame_axis(ax, frame_time_map)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, output_name), dpi=200)
        plt.close()

    def plot_time_series_split(data_map, ylabel, title_prefix, output_prefix):
        groups = {
            "no_motion": ["rigid_no_motion", "restored_no_motion"],
            "with_motion": ["rigid_with_motion", "restored_with_motion"],
        }

        for group_name, group_methods in groups.items():
            has_data = any(data_map.get(m) for m in group_methods)
            output_name = f"{output_prefix}_{group_name}.png"
            title = f"{title_prefix} ({group_name.replace('_', ' ').title()})"

            if not has_data:
                save_empty_plot(os.path.join(out_dir, output_name), title, "No data available")
                continue

            plt.figure(figsize=(10, 4.5))
            ax = plt.gca()
            add_restored_overlays(ax, group_methods, restored_spans)
            for method in group_methods:
                if method not in data_map:
                    continue
                series = sorted(data_map[method])
                if not series:
                    continue
                times = [item[0] for item in series]
                values = [item[1] for item in series]
                ax.plot(
                    times,
                    values,
                    marker="o",
                    linewidth=1.7,
                    markersize=3.0,
                    label=labels[method],
                    color=colors[method],
                    alpha=0.85,
                )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            add_frame_axis(ax, frame_time_map)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, output_name), dpi=200)
            plt.close()

    plot_time_series_overlay(
        matched_rmse_data,
        "Matched 3D RMSE",
        "Matched 3D RMSE vs Time",
        "rmse_3d_vs_time.png",
    )
    plot_time_series_split(coverage_data, "Coverage", "Coverage vs Time", "coverage_vs_time")

    part_rows = read_csv(os.path.join(results_dir, "eval_3d_part.csv"))
    part_methods = active_methods(part_rows)
    if not part_methods:
        part_methods = methods

    part_order = []
    seen_parts = set()
    part_values = {method: {} for method in part_methods}
    layer_values = {
        method: {
            0: {},
            1: {},
            2: {},
        }
        for method in part_methods
    }

    for row in part_rows:
        method = row.get("method")
        part = row.get("part")
        rmse = parse_float(row.get("matched_rmse_3d"))
        if not part:
            continue
        if part not in seen_parts:
            seen_parts.add(part)
            part_order.append(part)
        if method in part_values and rmse is not None:
            part_values[method][part] = rmse
        if method in layer_values:
            for restored in (0, 1, 2):
                layer_rmse = parse_float(row.get(f"matched_rmse_restored{restored}"))
                if layer_rmse is not None:
                    layer_values[method][restored][part] = layer_rmse

    if not part_order:
        save_empty_plot(
            os.path.join(out_dir, "rmse_3d_per_part.png"),
            "Matched 3D RMSE per Part",
            "No part data available",
        )
        save_empty_plot(
            os.path.join(out_dir, "rmse_3d_per_part_restored_category.png"),
            "Matched 3D RMSE per Part(restored category)",
            "No part data available",
        )
    else:
        x = list(range(len(part_order)))
        width = 0.8 / max(len(part_methods), 1)
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        for idx, method in enumerate(part_methods):
            values = [part_values[method].get(part, math.nan) for part in part_order]
            offset = (idx - (len(part_methods) - 1) / 2.0) * width
            positions = [item + offset for item in x]
            ax.bar(
                positions,
                values,
                width=width,
                label=labels.get(method, method),
                color=colors.get(method, "#888888"),
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(part_order, rotation=35, ha="right")
        ax.set_ylabel("Matched 3D RMSE")
        ax.set_title("Matched 3D RMSE per Part")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_3d_per_part.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(13, 5.6))
        ax = plt.gca()
        subgroup_width = 0.82 / max(len(part_methods), 1)
        category_width = subgroup_width / 3.0
        layer_labels = {0: "restored=0", 1: "restored=1", 2: "restored=2"}
        for method_idx, method in enumerate(part_methods):
            method_offset = (method_idx - (len(part_methods) - 1) / 2.0) * subgroup_width
            for restored in (0, 1, 2):
                values = [layer_values[method][restored].get(part, math.nan) for part in part_order]
                offset = method_offset + (restored - 1) * category_width
                positions = [item + offset for item in x]
                ax.bar(
                    positions,
                    values,
                    width=category_width,
                    label=f"{labels.get(method, method)} {layer_labels[restored]}",
                    color=colors.get(method, "#888888"),
                    alpha=0.78,
                    hatch=restored_hatches[restored],
                    edgecolor="#444444",
                    linewidth=0.7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(part_order, rotation=35, ha="right")
        ax.set_ylabel("Matched 3D RMSE")
        ax.set_title("Matched 3D RMSE per Part(restored category)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "rmse_3d_per_part_restored_category.png"),
            dpi=200,
        )
        plt.close()


camera_order = ["front", "side", "top"]
camera_labels = {"front": "Front Camera", "side": "Side Camera", "top": "Top Camera"}
camera_colors = {
    "front": "#6A9EC9",
    "side":  "#E67F65",
    "top":   "#5A9E6E",
}
heatmap_cmap = ListedColormap(["#FFFFFF", "#C0392B"])

_OBS_RESULTS_DIR = "./results"


def plot_observation(out_dir):
    anomaly_rows = read_csv(os.path.join(_OBS_RESULTS_DIR, "anomaly_missing_obs.csv"))
    frame_rows = read_csv(os.path.join(_OBS_RESULTS_DIR, "eval_3d_frame_audit.csv"))

    base_frame_time_map = build_frame_time_map(frame_rows)
    if not base_frame_time_map:
        base_frame_time_map = build_frame_time_map(anomaly_rows)

    frames_in_order = infer_full_frames(frame_rows, anomaly_rows)
    frame_time_map = complete_frame_time_map(frames_in_order, base_frame_time_map)

    missing_count_output = os.path.join(out_dir, "camera_missing_count_vs_time.png")
    heatmap_output = os.path.join(out_dir, "missing_heatmap_all_cameras.png")

    if not frames_in_order and not anomaly_rows:
        save_empty_plot(
            missing_count_output,
            "Camera Missing Count vs Time",
            "No observation summary available",
        )
        save_empty_plot(
            heatmap_output,
            "Camera Missing Heatmaps",
            "No missing-per-part data available",
        )
        return

    missing_count = defaultdict(lambda: defaultdict(int))
    for row in anomaly_rows:
        camera = row.get("camera")
        frame = parse_int(row.get("frame_id"))
        if camera is None or frame is None:
            continue
        missing_count[camera][frame] += 1

    missing_count_data = {camera: [] for camera in camera_order}
    for camera in camera_order:
        for frame in frames_in_order:
            t = frame_time_map.get(frame)
            if t is None:
                t = float(frame)
            missing_count_data[camera].append((t, missing_count[camera].get(frame, 0)))

    fig, axes = plt.subplots(
        nrows=len(camera_order),
        ncols=1,
        figsize=(10, 6),
        sharex=False,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, camera in zip(axes, camera_order):
        series = missing_count_data[camera]
        values = []
        if series:
            times = [item[0] for item in series]
            values = [item[1] for item in series]
            ax.plot(
                times,
                values,
                linewidth=0.7,
                color=camera_colors[camera],
                alpha=1.0,
            )
        ax.set_title(camera_labels[camera])
        ax.set_ylabel("Missing Parts")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        y_max = max(values) if series else 10
        y_ticks = np.arange(0, int(y_max) + 2, 2)
        ax.set_yticks(y_ticks)

        add_frame_axis(ax, frame_time_map)

    fig.savefig(missing_count_output, dpi=200)
    plt.close(fig)

    parts_in_order = []
    seen_parts = set()
    for row in anomaly_rows:
        part = (row.get("part_name") or "").strip()
        if part and part not in seen_parts:
            seen_parts.add(part)
            parts_in_order.append(part)

    if not frames_in_order or not parts_in_order:
        save_empty_plot(heatmap_output, "Camera Missing Heatmaps", "No missing-per-part data available")
        return

    frame_index = {frame: idx for idx, frame in enumerate(frames_in_order)}
    part_index = {part: idx for idx, part in enumerate(parts_in_order)}
    matrices = {
        camera: np.zeros((len(parts_in_order), len(frames_in_order)), dtype=float)
        for camera in camera_order
    }

    for row in anomaly_rows:
        cam = row.get("camera")
        part = (row.get("part_name") or "").strip()
        frame = parse_int(row.get("frame_id"))
        if cam == "front" and part in part_index and frame in frame_index:
            matrices[cam][part_index[part], frame_index[frame]] = 1.0
        if cam == "side" and part in part_index and frame in frame_index:
            matrices[cam][part_index[part], frame_index[frame]] = 1.0
        if cam == "top" and part in part_index and frame in frame_index:
            matrices[cam][part_index[part], frame_index[frame]] = 1.0

    tick_positions = choose_tick_positions(len(frames_in_order))
    bottom_labels = []
    for pos in tick_positions:
        frame = frames_in_order[pos]
        t = frame_time_map.get(frame)
        if t is None:
            bottom_labels.append(str(frame))
        else:
            bottom_labels.append(f"{t:.2f}")

    fig_height = max(9.0, 1.35 * len(parts_in_order))
    fig, axes = plt.subplots(
        nrows=len(camera_order),
        ncols=1,
        figsize=(12, fig_height),
        sharex=False,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    image = None
    for ax, camera in zip(axes, camera_order):
        image = ax.imshow(
            matrices[camera],
            aspect="auto",
            interpolation="nearest",
            cmap=heatmap_cmap,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{camera_labels[camera]} Missing Heatmap")
        ax.set_ylabel("Body Part")
        ax.set_xlabel("Time (s)")
        ax.set_yticks(np.arange(len(parts_in_order)))
        ax.set_yticklabels(parts_in_order)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(bottom_labels, rotation=0)

        top_ax = ax.secondary_xaxis("top")
        top_ax.set_xlabel("Frame")
        top_ax.set_xticks(tick_positions)
        top_ax.set_xticklabels([str(frames_in_order[pos]) for pos in tick_positions])

    cbar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Observed", "Missing"])

    fig.savefig(heatmap_output, dpi=200)
    plt.close(fig)


def main():
    global _OBS_RESULTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--out_dir", default="./results/figures")
    parser.add_argument("--only", choices=["observation", "evaluation"], default=None)
    args = parser.parse_args()

    _OBS_RESULTS_DIR = args.results_dir
    os.makedirs(args.out_dir, exist_ok=True)

    if args.only != "evaluation":
        plot_observation(args.out_dir)
    if args.only != "observation":
        plot_evaluation(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
