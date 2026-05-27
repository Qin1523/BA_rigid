#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <array>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <limits>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <limits.h>
#include <unistd.h>

using namespace std;

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR ""
#endif

struct Point3Row {
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    string part;
    double x = numeric_limits<double>::quiet_NaN();
    double y = numeric_limits<double>::quiet_NaN();
    double z = numeric_limits<double>::quiet_NaN();
    int restored = 0;
};

struct AggregateAccumulator {
    size_t gt_points = 0;
    size_t matched_points = 0;
    double matched_sum_sq = 0.0;
    array<size_t, 3> restored_counts = {0, 0, 0};
    array<double, 3> restored_sum_sq = {0.0, 0.0, 0.0};
};

struct PointAuditRow {
    string method;
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    string part;
    string status;
    double gt_x = numeric_limits<double>::quiet_NaN();
    double gt_y = numeric_limits<double>::quiet_NaN();
    double gt_z = numeric_limits<double>::quiet_NaN();
    double pred_x = numeric_limits<double>::quiet_NaN();
    double pred_y = numeric_limits<double>::quiet_NaN();
    double pred_z = numeric_limits<double>::quiet_NaN();
    double error_3d = numeric_limits<double>::quiet_NaN();
    int restored = 0;
    bool is_large_error = false;
};

struct FrameSummaryRow {
    string method;
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    size_t gt_num_parts = 0;
    size_t matched_num_parts = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
    int restored = 0;
};

struct PartSummaryRow {
    string method;
    string part;
    size_t gt_points = 0;
    size_t matched_points = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
    double matched_rmse_restored0 = numeric_limits<double>::quiet_NaN();
    double matched_rmse_restored1 = numeric_limits<double>::quiet_NaN();
    double matched_rmse_restored2 = numeric_limits<double>::quiet_NaN();
    size_t count_restored0 = 0;
    size_t count_restored1 = 0;
    size_t count_restored2 = 0;
};

struct OverallSummaryRow {
    string method;
    size_t gt_total_points = 0;
    size_t matched_points = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
};

struct MethodSpec {
    string name;
    string path;
    bool explicit_path = false;
    vector<string> fallback_paths;
};

struct ObservationInputSpec {
    string camera;
    string path;
    bool explicit_path = false;
};

struct ObservationRow {
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    string part;
};

struct ObservationAnalysisResult {
    vector<string> camera_order;
    vector<string> part_order;
    vector<long long> frames;
    map<long long, double> frame_times;
    map<long long, map<string, set<string>>> obs_map;
    map<string, string> resolved_paths;
    map<string, size_t> row_counts;
};

struct ObservationOutputArtifacts {
    bool csvs_generated = false;
    bool plots_generated = false;
    string camera_summary_csv_path;
    string missing_per_part_csv_path;
};

struct MethodEvaluation {
    string method;
    string input_path;
    vector<PointAuditRow> point_rows;
    vector<FrameSummaryRow> frame_rows;
    vector<PartSummaryRow> part_rows;
    OverallSummaryRow overall_row;
};

using TrajectoryMap = map<pair<long long, string>, Point3Row>;

static const vector<string> kMethodOrder = {
    "rigid_no_motion",
    "rigid_with_motion",
    "restored_no_motion",
    "restored_with_motion"
};

string remove_bom(const string& str) {
    if (str.size() >= 3 &&
        static_cast<unsigned char>(str[0]) == 0xEF &&
        static_cast<unsigned char>(str[1]) == 0xBB &&
        static_cast<unsigned char>(str[2]) == 0xBF) {
        return str.substr(3);
    }
    return str;
}

bool file_exists(const string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

bool directory_exists(const string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool is_absolute_path(const string& path) {
    return !path.empty() && path[0] == '/';
}

string parent_dir(const string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

string join_path(const string& base, const string& child) {
    if (base.empty() || base == ".") return child;
    if (child.empty()) return base;
    if (base.back() == '/') return base + child;
    return base + "/" + child;
}

string trim_copy(const string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

vector<string> split_csv_line(const string& line) {
    vector<string> fields;
    string field;
    stringstream ss(line);
    while (getline(ss, field, ',')) {
        fields.push_back(trim_copy(field));
    }
    return fields;
}

string normalize_header_name(string name) {
    name = trim_copy(remove_bom(name));
    string normalized;
    normalized.reserve(name.size());
    for (unsigned char ch : name) {
        if (isalnum(ch)) {
            normalized.push_back(static_cast<char>(tolower(ch)));
        }
    }
    return normalized;
}

string get_executable_dir(const char* argv0) {
    if (argv0 == nullptr) {
        return ".";
    }

    char resolved_path[PATH_MAX];
    if (realpath(argv0, resolved_path) != nullptr) {
        return parent_dir(resolved_path);
    }

    return ".";
}

string resolve_input_path(const string& path, const string& executable_dir) {
    if (path.empty()) {
        return path;
    }

    if (file_exists(path) || is_absolute_path(path)) {
        return path;
    }

    vector<string> search_roots;
    search_roots.push_back(executable_dir);

    string current = executable_dir;
    for (int depth = 0; depth < 3; depth++) {
        string parent = parent_dir(current);
        if (parent == current) {
            break;
        }
        search_roots.push_back(parent);
        current = parent;
    }

    string source_root = PROJECT_SOURCE_DIR;
    if (!source_root.empty()) {
        search_roots.push_back(source_root);
    }

    for (const string& root : search_roots) {
        string candidate = join_path(root, path);
        if (file_exists(candidate)) {
            return candidate;
        }
    }

    return path;
}

bool ensure_directory(const string& path) {
    if (path.empty()) {
        return false;
    }

    if (directory_exists(path)) {
        return true;
    }

    if (mkdir(path.c_str(), 0755) == 0) {
        return true;
    }

    if (errno == EEXIST && directory_exists(path)) {
        return true;
    }

    cerr << "ERROR: Failed to create directory " << path << endl;
    return false;
}

bool is_x_header(const string& normalized_name) {
    return normalized_name == "x" || normalized_name == "xr" || normalized_name == "xgt";
}

bool is_y_header(const string& normalized_name) {
    return normalized_name == "y" || normalized_name == "yr" || normalized_name == "ygt";
}

bool is_obs_x_header(const string& normalized_name) {
    return is_x_header(normalized_name) || normalized_name == "u";
}

bool is_obs_y_header(const string& normalized_name) {
    return is_y_header(normalized_name) || normalized_name == "v";
}

bool is_z_header(const string& normalized_name) {
    return normalized_name == "z" || normalized_name == "zr" || normalized_name == "zgt";
}

int find_column_index(const vector<string>& header_fields,
                      const vector<string>& candidate_names) {
    for (size_t i = 0; i < header_fields.size(); i++) {
        const string normalized = normalize_header_name(header_fields[i]);
        for (const string& candidate : candidate_names) {
            if (normalized == normalize_header_name(candidate)) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

int normalize_restored_value(int restored) {
    if (restored == 1 || restored == 2) {
        return restored;
    }
    return 0;
}

TrajectoryMap load_trajectory_csv(const string& path) {
    TrajectoryMap trajectory;
    ifstream ifs(path);
    if (!ifs.is_open()) {
        cerr << "ERROR: Cannot open " << path << endl;
        return trajectory;
    }

    string line;
    if (!getline(ifs, line)) {
        cerr << "ERROR: Empty CSV file " << path << endl;
        return trajectory;
    }

    vector<string> header_fields = split_csv_line(remove_bom(line));
    int frame_col = -1;
    int t_col = -1;
    int part_col = -1;
    int x_col = -1;
    int y_col = -1;
    int z_col = -1;
    const int restored_col = find_column_index(header_fields, {"restored"});

    for (size_t i = 0; i < header_fields.size(); i++) {
        string name = normalize_header_name(header_fields[i]);
        if (name == "frame") {
            frame_col = static_cast<int>(i);
        } else if (name == "t" || name == "time" || name == "timestamp") {
            t_col = static_cast<int>(i);
        } else if (name == "part" || name == "bodypart") {
            part_col = static_cast<int>(i);
        } else if (is_x_header(name)) {
            x_col = static_cast<int>(i);
        } else if (is_y_header(name)) {
            y_col = static_cast<int>(i);
        } else if (is_z_header(name)) {
            z_col = static_cast<int>(i);
        }
    }

    if (frame_col < 0 || part_col < 0 || x_col < 0 || y_col < 0 || z_col < 0) {
        cerr << "ERROR: CSV header in " << path
             << " must contain frame, part, and XYZ columns" << endl;
        return trajectory;
    }

    size_t duplicate_count = 0;
    while (getline(ifs, line)) {
        line = remove_bom(line);
        if (trim_copy(line).empty()) continue;

        vector<string> fields = split_csv_line(line);
        const int required_col = max({frame_col, part_col, x_col, y_col, z_col, t_col});
        if (required_col >= static_cast<int>(fields.size())) {
            continue;
        }

        try {
            Point3Row row;
            row.frame = stoll(fields[frame_col]);
            if (t_col >= 0 && !fields[t_col].empty()) {
                row.t = stod(fields[t_col]);
            }
            row.part = fields[part_col];
            row.x = stod(fields[x_col]);
            row.y = stod(fields[y_col]);
            row.z = stod(fields[z_col]);
            if (restored_col >= 0 &&
                restored_col < static_cast<int>(fields.size()) &&
                !fields[restored_col].empty()) {
                row.restored = normalize_restored_value(stoi(fields[restored_col]));
            }

            pair<long long, string> key = make_pair(row.frame, row.part);
            if (trajectory.count(key)) {
                duplicate_count++;
            }
            trajectory[key] = row;
        } catch (...) {
            continue;
        }
    }

    cout << "Loaded " << trajectory.size() << " unique 3D points from " << path;
    if (duplicate_count > 0) {
        cout << " (" << duplicate_count << " duplicate keys overwritten)";
    }
    cout << endl;
    return trajectory;
}

vector<ObservationRow> load_observation_csv(const string& path,
                                            const string& camera,
                                            double fps,
                                            size_t& duplicate_count) {
    vector<ObservationRow> rows;
    duplicate_count = 0;

    ifstream ifs(path);
    if (!ifs.is_open()) {
        cerr << "ERROR: Cannot open observation CSV " << path << endl;
        return rows;
    }

    string line;
    if (!getline(ifs, line)) {
        cerr << "ERROR: Empty observation CSV " << path << endl;
        return rows;
    }

    vector<string> header_fields = split_csv_line(remove_bom(line));
    int frame_col = -1;
    int t_col = -1;
    int part_col = -1;
    int x_col = -1;
    int y_col = -1;

    for (size_t i = 0; i < header_fields.size(); i++) {
        const string name = normalize_header_name(header_fields[i]);
        if (name == "frame") {
            frame_col = static_cast<int>(i);
        } else if (name == "t" || name == "time" || name == "timestamp") {
            t_col = static_cast<int>(i);
        } else if (name == "part" || name == "bodypart") {
            part_col = static_cast<int>(i);
        } else if (is_obs_x_header(name)) {
            x_col = static_cast<int>(i);
        } else if (is_obs_y_header(name)) {
            y_col = static_cast<int>(i);
        }
    }

    if (part_col < 0 || x_col < 0 || y_col < 0 || (frame_col < 0 && t_col < 0)) {
        cerr << "ERROR: Observation CSV header in " << path
             << " must contain part, x/u, y/v, and frame or t columns" << endl;
        return rows;
    }

    map<pair<long long, string>, ObservationRow> unique_rows;
    while (getline(ifs, line)) {
        line = remove_bom(line);
        if (trim_copy(line).empty()) continue;

        vector<string> fields = split_csv_line(line);
        const int required_col = max({frame_col, t_col, part_col, x_col, y_col});
        if (required_col >= static_cast<int>(fields.size())) {
            continue;
        }

        try {
            ObservationRow row;
            row.part = fields[part_col];

            bool has_frame = false;
            if (frame_col >= 0 && !fields[frame_col].empty()) {
                row.frame = stoll(fields[frame_col]);
                has_frame = true;
            }

            double t_value = numeric_limits<double>::quiet_NaN();
            if (t_col >= 0 && !fields[t_col].empty()) {
                t_value = stod(fields[t_col]);
            }

            if (!has_frame) {
                if (!(fps > 0.0) || !std::isfinite(t_value)) {
                    continue;
                }
                row.frame = llround(t_value * fps);
            }

            if (std::isfinite(t_value)) {
                row.t = t_value;
            } else if (fps > 0.0) {
                row.t = static_cast<double>(row.frame) / fps;
            }

            // Parse the 2D coordinates so malformed rows are ignored.
            (void)stod(fields[x_col]);
            (void)stod(fields[y_col]);

            const pair<long long, string> key = make_pair(row.frame, row.part);
            if (unique_rows.count(key)) {
                duplicate_count++;
            }
            unique_rows[key] = row;
        } catch (...) {
            continue;
        }
    }

    rows.reserve(unique_rows.size());
    for (const auto& kv : unique_rows) {
        rows.push_back(kv.second);
    }

    cout << "Loaded " << rows.size() << " unique " << camera
         << " observation rows from " << path;
    if (duplicate_count > 0) {
        cout << " (" << duplicate_count << " duplicate frame-part keys overwritten)";
    }
    cout << endl;
    return rows;
}

bool build_observation_analysis(const vector<ObservationInputSpec>& specs,
                                double fps,
                                ObservationAnalysisResult& result) {
    result = ObservationAnalysisResult{};

    set<string> seen_parts;
    bool has_any_rows = false;
    long long min_frame = numeric_limits<long long>::max();
    long long max_frame = numeric_limits<long long>::min();

    for (const auto& spec : specs) {
        result.camera_order.push_back(spec.camera);
        result.resolved_paths[spec.camera] = spec.path;

        size_t duplicate_count = 0;
        const vector<ObservationRow> rows =
            load_observation_csv(spec.path, spec.camera, fps, duplicate_count);
        result.row_counts[spec.camera] = rows.size();

        for (const auto& row : rows) {
            has_any_rows = true;
            min_frame = min(min_frame, row.frame);
            max_frame = max(max_frame, row.frame);
            result.obs_map[row.frame][spec.camera].insert(row.part);

            auto time_it = result.frame_times.find(row.frame);
            if (time_it == result.frame_times.end() ||
                (!std::isfinite(time_it->second) && std::isfinite(row.t))) {
                result.frame_times[row.frame] = row.t;
            }

            if (seen_parts.insert(row.part).second) {
                result.part_order.push_back(row.part);
            }
        }
    }

    if (!has_any_rows || result.part_order.empty()) {
        return false;
    }

    for (long long frame = min_frame; frame <= max_frame; frame++) {
        result.frames.push_back(frame);
        auto time_it = result.frame_times.find(frame);
        if (time_it == result.frame_times.end() || !std::isfinite(time_it->second)) {
            result.frame_times[frame] = (fps > 0.0)
                ? static_cast<double>(frame) / fps
                : numeric_limits<double>::quiet_NaN();
        }
    }

    return true;
}

double compute_distance_3d(const Point3Row& a, const Point3Row& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

double safe_rmse(double sum_sq, size_t count) {
    if (count == 0) {
        return numeric_limits<double>::quiet_NaN();
    }
    return sqrt(sum_sq / static_cast<double>(count));
}

double safe_ratio(size_t numerator, size_t denominator) {
    if (denominator == 0) {
        return numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

string format_double_or_empty(double value) {
    if (std::isnan(value)) {
        return "";
    }

    ostringstream oss;
    oss << value;
    return oss.str();
}

double choose_time_value(double gt_t, double pred_t) {
    if (!std::isnan(gt_t)) {
        return gt_t;
    }
    return pred_t;
}

int method_sort_rank(const string& method) {
    for (size_t i = 0; i < kMethodOrder.size(); i++) {
        if (kMethodOrder[i] == method) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(kMethodOrder.size());
}

int status_sort_rank(const string& status) {
    if (status == "matched") return 0;
    if (status == "extra_prediction") return 1;
    return 2;
}

double compute_median(vector<double> values) {
    if (values.empty()) {
        return numeric_limits<double>::quiet_NaN();
    }

    sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

double compute_frame_high_error_threshold(const vector<FrameSummaryRow>& frame_rows) {
    vector<double> finite_values;
    for (const auto& row : frame_rows) {
        if (!std::isnan(row.matched_rmse_3d)) {
            finite_values.push_back(row.matched_rmse_3d);
        }
    }

    if (finite_values.empty()) {
        return numeric_limits<double>::quiet_NaN();
    }

    const double median = compute_median(finite_values);
    vector<double> abs_deviations;
    abs_deviations.reserve(finite_values.size());
    for (double value : finite_values) {
        abs_deviations.push_back(fabs(value - median));
    }

    const double mad = compute_median(abs_deviations);
    return median + 3.0 * mad;
}

string resolve_method_input_path(const MethodSpec& spec,
                                 const string& executable_dir,
                                 bool& used_fallback) {
    used_fallback = false;
    const string resolved_primary = resolve_input_path(spec.path, executable_dir);
    if (file_exists(resolved_primary)) {
        return resolved_primary;
    }

    if (!spec.explicit_path) {
        for (const string& fallback_path : spec.fallback_paths) {
            const string resolved_fallback = resolve_input_path(fallback_path, executable_dir);
            if (file_exists(resolved_fallback)) {
                used_fallback = true;
                return resolved_fallback;
            }
        }
    }

    return resolved_primary;
}

bool validate_prediction_alignment(const string& method,
                                   const string& input_path,
                                   const TrajectoryMap& gt,
                                   const TrajectoryMap& pred) {
    for (const auto& kv : gt) {
        if (pred.find(kv.first) == pred.end()) {
            cerr << "ERROR: Prediction trajectory for " << method
                 << " is missing GT-aligned key (frame=" << kv.first.first
                 << ", part=" << kv.first.second << ") in " << input_path << endl;
            return false;
        }
    }
    return true;
}

MethodEvaluation evaluate_method(const string& method,
                                 const string& input_path,
                                 const TrajectoryMap& gt,
                                 const TrajectoryMap& pred) {
    MethodEvaluation evaluation;
    evaluation.method = method;
    evaluation.input_path = input_path;

    map<long long, AggregateAccumulator> frame_accumulators;
    map<long long, double> frame_times;
    map<long long, int> frame_restored;
    map<string, AggregateAccumulator> part_accumulators;
    AggregateAccumulator overall_accumulator;

    for (const auto& kv : gt) {
        const Point3Row& gt_row = kv.second;
        PointAuditRow row;
        row.method = method;
        row.frame = gt_row.frame;
        row.t = gt_row.t;
        row.part = gt_row.part;
        row.gt_x = gt_row.x;
        row.gt_y = gt_row.y;
        row.gt_z = gt_row.z;

        AggregateAccumulator& frame_acc = frame_accumulators[gt_row.frame];
        AggregateAccumulator& part_acc = part_accumulators[gt_row.part];
        frame_acc.gt_points++;
        part_acc.gt_points++;
        overall_accumulator.gt_points++;

        auto frame_time_it = frame_times.find(gt_row.frame);
        if (frame_time_it == frame_times.end() || std::isnan(frame_time_it->second)) {
            frame_times[gt_row.frame] = gt_row.t;
        }

        const Point3Row& pred_row = pred.at(kv.first);
        const double error_3d = compute_distance_3d(gt_row, pred_row);
        const double error_sq = error_3d * error_3d;

        row.status = "matched";
        row.t = choose_time_value(gt_row.t, pred_row.t);
        row.pred_x = pred_row.x;
        row.pred_y = pred_row.y;
        row.pred_z = pred_row.z;
        row.error_3d = error_3d;
        row.restored = normalize_restored_value(pred_row.restored);

        frame_acc.matched_points++;
        frame_acc.matched_sum_sq += error_sq;
        frame_restored[gt_row.frame] = max(frame_restored[gt_row.frame], row.restored);

        part_acc.matched_points++;
        part_acc.matched_sum_sq += error_sq;
        part_acc.restored_counts[row.restored]++;
        part_acc.restored_sum_sq[row.restored] += error_sq;

        overall_accumulator.matched_points++;
        overall_accumulator.matched_sum_sq += error_sq;

        evaluation.point_rows.push_back(row);
    }

    for (const auto& kv : pred) {
        if (gt.count(kv.first)) {
            continue;
        }

        const Point3Row& pred_row = kv.second;
        PointAuditRow row;
        row.method = method;
        row.frame = pred_row.frame;
        row.t = pred_row.t;
        row.part = pred_row.part;
        row.status = "extra_prediction";
        row.pred_x = pred_row.x;
        row.pred_y = pred_row.y;
        row.pred_z = pred_row.z;
        row.restored = normalize_restored_value(pred_row.restored);
        evaluation.point_rows.push_back(row);
    }

    evaluation.overall_row.method = method;
    evaluation.overall_row.gt_total_points = overall_accumulator.gt_points;
    evaluation.overall_row.matched_points = overall_accumulator.matched_points;
    evaluation.overall_row.coverage = safe_ratio(
        overall_accumulator.matched_points, overall_accumulator.gt_points);
    evaluation.overall_row.matched_rmse_3d = safe_rmse(
        overall_accumulator.matched_sum_sq, overall_accumulator.matched_points);

    for (const auto& kv : frame_accumulators) {
        FrameSummaryRow row;
        row.method = method;
        row.frame = kv.first;
        auto time_it = frame_times.find(kv.first);
        if (time_it != frame_times.end()) {
            row.t = time_it->second;
        }
        row.gt_num_parts = kv.second.gt_points;
        row.matched_num_parts = kv.second.matched_points;
        row.coverage = safe_ratio(kv.second.matched_points, kv.second.gt_points);
        row.matched_rmse_3d = safe_rmse(kv.second.matched_sum_sq, kv.second.matched_points);
        auto restored_it = frame_restored.find(kv.first);
        if (restored_it != frame_restored.end()) {
            row.restored = restored_it->second;
        }
        evaluation.frame_rows.push_back(row);
    }

    for (const auto& kv : part_accumulators) {
        PartSummaryRow row;
        row.method = method;
        row.part = kv.first;
        row.gt_points = kv.second.gt_points;
        row.matched_points = kv.second.matched_points;
        row.coverage = safe_ratio(kv.second.matched_points, kv.second.gt_points);
        row.matched_rmse_3d = safe_rmse(kv.second.matched_sum_sq, kv.second.matched_points);
        row.count_restored0 = kv.second.restored_counts[0];
        row.count_restored1 = kv.second.restored_counts[1];
        row.count_restored2 = kv.second.restored_counts[2];
        row.matched_rmse_restored0 = safe_rmse(
            kv.second.restored_sum_sq[0], kv.second.restored_counts[0]);
        row.matched_rmse_restored1 = safe_rmse(
            kv.second.restored_sum_sq[1], kv.second.restored_counts[1]);
        row.matched_rmse_restored2 = safe_rmse(
            kv.second.restored_sum_sq[2], kv.second.restored_counts[2]);
        evaluation.part_rows.push_back(row);
    }

    sort(evaluation.part_rows.begin(), evaluation.part_rows.end(),
         [](const PartSummaryRow& a, const PartSummaryRow& b) {
             return a.part < b.part;
         });

    const double high_error_threshold = compute_frame_high_error_threshold(evaluation.frame_rows);
    if (std::isfinite(high_error_threshold)) {
        cout << evaluation.method << " frame high-error threshold (mm): "
             << high_error_threshold << endl;
    }

    for (auto& row : evaluation.point_rows) {
        row.is_large_error =
            row.status == "matched" &&
            !std::isnan(row.error_3d) &&
            std::isfinite(high_error_threshold) &&
            row.error_3d > high_error_threshold;
    }

    sort(evaluation.point_rows.begin(), evaluation.point_rows.end(),
         [](const PointAuditRow& a, const PointAuditRow& b) {
             if (a.frame != b.frame) return a.frame < b.frame;
             if (a.part != b.part) return a.part < b.part;
             return status_sort_rank(a.status) < status_sort_rank(b.status);
         });

    return evaluation;
}

void save_overall_csv(const string& path,
                      const vector<OverallSummaryRow>& overall_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,gt_total_points,matched_points,coverage,matched_rmse_3d\n";
    for (const auto& row : overall_rows) {
        ofs << row.method << ","
            << row.gt_total_points << ","
            << row.matched_points << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << "\n";
    }

    cout << "Saved overall 3D evaluation to " << path << endl;
}

void save_frame_audit_csv(const string& path,
                          const vector<FrameSummaryRow>& frame_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,frame,t,gt_num_parts,matched_num_parts,coverage,matched_rmse_3d,restored\n";
    for (const auto& row : frame_rows) {
        ofs << row.method << ","
            << row.frame << ","
            << format_double_or_empty(row.t) << ","
            << row.gt_num_parts << ","
            << row.matched_num_parts << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << row.restored << "\n";
    }

    cout << "Saved per-frame audit to " << path << endl;
}

void save_part_csv(const string& path,
                   const vector<PartSummaryRow>& part_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,part,gt_points,matched_points,coverage,matched_rmse_3d,"
           "matched_rmse_restored0,matched_rmse_restored1,matched_rmse_restored2,"
           "count_restored0,count_restored1,count_restored2\n";
    for (const auto& row : part_rows) {
        ofs << row.method << ","
            << row.part << ","
            << row.gt_points << ","
            << row.matched_points << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.matched_rmse_restored0) << ","
            << format_double_or_empty(row.matched_rmse_restored1) << ","
            << format_double_or_empty(row.matched_rmse_restored2) << ","
            << row.count_restored0 << ","
            << row.count_restored1 << ","
            << row.count_restored2 << "\n";
    }

    cout << "Saved per-part 3D summary to " << path << endl;
}

void save_point_audit_csv(const string& path,
                          const vector<PointAuditRow>& point_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,frame,t,part,status,gt_x,gt_y,gt_z,pred_x,pred_y,pred_z,error_3d,restored,is_large_error\n";
    for (const auto& row : point_rows) {
        ofs << row.method << ","
            << row.frame << ","
            << format_double_or_empty(row.t) << ","
            << row.part << ","
            << row.status << ","
            << format_double_or_empty(row.gt_x) << ","
            << format_double_or_empty(row.gt_y) << ","
            << format_double_or_empty(row.gt_z) << ","
            << format_double_or_empty(row.pred_x) << ","
            << format_double_or_empty(row.pred_y) << ","
            << format_double_or_empty(row.pred_z) << ","
            << format_double_or_empty(row.error_3d) << ","
            << row.restored << ","
            << (row.is_large_error ? 1 : 0) << "\n";
    }

    cout << "Saved point-level audit to " << path << endl;
}

void save_observation_camera_summary_csv(const string& path,
                                         const ObservationAnalysisResult& analysis) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t";
    for (const auto& camera : analysis.camera_order) {
        ofs << "," << camera << "_coverage";
    }
    for (const auto& camera : analysis.camera_order) {
        ofs << "," << camera << "_missing";
    }
    ofs << "\n";

    const size_t total_parts = analysis.part_order.size();
    for (long long frame : analysis.frames) {
        ofs << frame << "," << format_double_or_empty(analysis.frame_times.at(frame));

        for (const auto& camera : analysis.camera_order) {
            size_t observed_count = 0;
            auto frame_it = analysis.obs_map.find(frame);
            if (frame_it != analysis.obs_map.end()) {
                auto camera_it = frame_it->second.find(camera);
                if (camera_it != frame_it->second.end()) {
                    observed_count = camera_it->second.size();
                }
            }
            ofs << "," << format_double_or_empty(safe_ratio(observed_count, total_parts));
        }

        for (const auto& camera : analysis.camera_order) {
            size_t observed_count = 0;
            auto frame_it = analysis.obs_map.find(frame);
            if (frame_it != analysis.obs_map.end()) {
                auto camera_it = frame_it->second.find(camera);
                if (camera_it != frame_it->second.end()) {
                    observed_count = camera_it->second.size();
                }
            }
            const size_t missing_count =
                (observed_count <= total_parts) ? (total_parts - observed_count) : 0;
            ofs << "," << missing_count;
        }

        ofs << "\n";
    }

    cout << "Saved observation camera summary to " << path << endl;
}

void save_observation_missing_per_part_csv(const string& path,
                                           const ObservationAnalysisResult& analysis) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,camera,part,is_missing\n";
    for (long long frame : analysis.frames) {
        const double t = analysis.frame_times.at(frame);
        auto frame_it = analysis.obs_map.find(frame);
        for (const auto& camera : analysis.camera_order) {
            const set<string>* observed_parts = nullptr;
            if (frame_it != analysis.obs_map.end()) {
                auto camera_it = frame_it->second.find(camera);
                if (camera_it != frame_it->second.end()) {
                    observed_parts = &camera_it->second;
                }
            }

            for (const auto& part : analysis.part_order) {
                const bool is_missing =
                    (observed_parts == nullptr || observed_parts->count(part) == 0);
                ofs << frame << ","
                    << format_double_or_empty(t) << ","
                    << camera << ","
                    << part << ","
                    << (is_missing ? 1 : 0) << "\n";
            }
        }
    }

    cout << "Saved observation missing audit to " << path << endl;
}

bool generate_plots(const string& output_dir) {
    const string script_path = join_path(output_dir, "plot_eval_3d.py");
    const string mpl_cache_dir = join_path(output_dir, ".mplcache");
    const string xdg_cache_dir = join_path(output_dir, ".cache");
    if (!ensure_directory(mpl_cache_dir) || !ensure_directory(xdg_cache_dir)) {
        return false;
    }

    static const char* script = R"PY(
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

output_dir = sys.argv[1]

method_order = [
    "rigid_no_motion",
    "rigid_with_motion",
    "restored_no_motion",
    "restored_with_motion",
]
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
restored_hatches = {
    0: "///",
    1: "xx",
    2: "..",
}

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

def build_frame_time_map(frame_rows):
    mapping = {}
    for row in frame_rows:
        frame = parse_int(row.get("frame"))
        t = parse_float(row.get("t"))
        if frame is None:
            continue
        if t is None:
            t = float(frame)
        mapping[frame] = t
    return mapping

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

def add_frame_axis(ax, mapping):
    if not mapping:
        return
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda t: times_to_frames(t, mapping),
            lambda f: frames_to_times(f, mapping),
        ),
    )
    secax.set_xlabel("Frame")

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

frame_rows = read_csv(os.path.join(output_dir, "eval_3d_frame_audit.csv"))
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
        save_empty_plot(os.path.join(output_dir, output_name), title, "No data available")
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
    plt.savefig(os.path.join(output_dir, output_name), dpi=200)
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
            save_empty_plot(os.path.join(output_dir, output_name), title, "No data available")
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
        plt.savefig(os.path.join(output_dir, output_name), dpi=200)
        plt.close()

plot_time_series_overlay(
    matched_rmse_data,
    "Matched 3D RMSE",
    "Matched 3D RMSE vs Time",
    "rmse_3d_vs_time.png",
)
plot_time_series_split(coverage_data, "Coverage", "Coverage vs Time", "coverage_vs_time")

part_rows = read_csv(os.path.join(output_dir, "eval_3d_part.csv"))
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
        os.path.join(output_dir, "rmse_3d_per_part.png"),
        "Matched 3D RMSE per Part",
        "No part data available",
    )
    save_empty_plot(
        os.path.join(output_dir, "rmse_3d_per_part_restored_category.png"),
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
    plt.savefig(os.path.join(output_dir, "rmse_3d_per_part.png"), dpi=200)
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
        os.path.join(output_dir, "rmse_3d_per_part_restored_category.png"),
        dpi=200,
    )
    plt.close()
)PY";

    ofstream ofs(script_path);
    if (!ofs.is_open()) {
        cerr << "WARNING: Cannot write plot script " << script_path << endl;
        return false;
    }
    ofs << script;
    ofs.close();

    string command = "XDG_CACHE_HOME=\"" + xdg_cache_dir + "\" "
                     "MPLCONFIGDIR=\"" + mpl_cache_dir + "\" "
                     "python3 \"" + script_path + "\" \"" + output_dir + "\"";
    int rc = system(command.c_str());
    std::remove(script_path.c_str());

    if (rc != 0) {
        cerr << "WARNING: Plot generation failed" << endl;
        return false;
    }

    cout << "Saved plots to " << output_dir << endl;
    return true;
}

bool generate_observation_plots(const string& output_dir) {
    const string script_path = join_path(output_dir, "plot_obs_missing.py");
    const string mpl_cache_dir = join_path(output_dir, ".mplcache");
    const string xdg_cache_dir = join_path(output_dir, ".cache");
    if (!ensure_directory(mpl_cache_dir) || !ensure_directory(xdg_cache_dir)) {
        return false;
    }

    static const char* script = R"PY(
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

output_dir = sys.argv[1]
camera_order = ["front", "side", "top"]
camera_labels = {
    "front": "Front Camera",
    "side": "Side Camera",
    "top": "Top Camera",
}
camera_colors = {
    "front": "#6A9EC9",
    "side": "#E67F65",
    "top": "#5A9E6E",
}

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

def build_frame_time_map(rows):
    mapping = {}
    for row in rows:
        frame = parse_int(row.get("frame"))
        t = parse_float(row.get("t"))
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
    frames = frames[keep]
    times = times[keep]
    if times.size == 0:
        return arr
    if times.size == 1:
        return np.full_like(arr, frames[0], dtype=float)
    return np.interp(arr, times, frames)

def save_empty_plot(path, title, message):
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def add_frame_axis(ax, mapping):
    if not mapping:
        return
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda t: times_to_frames(t, mapping),
            lambda f: frames_to_times(f, mapping),
        ),
    )
    secax.set_xlabel("Frame")

def choose_tick_positions(count, max_ticks=8):
    if count <= 0:
        return []
    if count <= max_ticks:
        return list(range(count))
    raw = np.linspace(0, count - 1, num=max_ticks, dtype=int)
    return sorted(set(int(x) for x in raw))

summary_rows = read_csv(os.path.join(output_dir, "obs_camera_summary.csv"))
missing_rows = read_csv(os.path.join(output_dir, "obs_missing_per_part.csv"))
frame_time_map = build_frame_time_map(summary_rows)

missing_count_output = os.path.join(output_dir, "camera_missing_count_vs_time.png")
heatmap_output = os.path.join(output_dir, "missing_heatmap_all_cameras.png")

if not summary_rows:
    save_empty_plot(missing_count_output, "Camera Missing Count vs Time", "No observation summary available")
    save_empty_plot(heatmap_output, "Camera Missing Heatmaps", "No missing-per-part data available")
else:
    missing_count_data = {camera: [] for camera in camera_order}

    for row in summary_rows:
        t = parse_float(row.get("t"))
        frame = parse_int(row.get("frame"))
        if t is None and frame is not None:
            t = frame_time_map.get(frame)
        if t is None:
            continue

        for camera in camera_order:
            missing_count = parse_float(row.get(f"{camera}_missing"))
            if missing_count is not None:
                missing_count_data[camera].append((t, missing_count))

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
        if series:
            times = [item[0] for item in series]
            values = [item[1] for item in series]
            ax.plot(
                times,
                values,
                linewidth=1.2,
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

frames_in_order = []
seen_frames = set()
for row in summary_rows:
    frame = parse_int(row.get("frame"))
    if frame is None or frame in seen_frames:
        continue
    seen_frames.add(frame)
    frames_in_order.append(frame)

parts_in_order = []
seen_parts = set()
for row in missing_rows:
    part = (row.get("part") or "").strip()
    if part and part not in seen_parts:
        seen_parts.add(part)
        parts_in_order.append(part)

heatmap_cmap = ListedColormap(["#f7f7f7", "#b22222"])

if not frames_in_order or not parts_in_order:
    save_empty_plot(heatmap_output, "Camera Missing Heatmaps", "No missing-per-part data available")
else:
    frame_to_index = {frame: idx for idx, frame in enumerate(frames_in_order)}
    part_to_index = {part: idx for idx, part in enumerate(parts_in_order)}
    matrices = {
        camera: np.zeros((len(parts_in_order), len(frames_in_order)), dtype=float)
        for camera in camera_order
    }

    for row in missing_rows:
        camera = row.get("camera")
        if camera not in matrices:
            continue
        frame = parse_int(row.get("frame"))
        part = (row.get("part") or "").strip()
        is_missing = parse_int(row.get("is_missing"))
        if frame not in frame_to_index or part not in part_to_index or is_missing is None:
            continue
        matrices[camera][part_to_index[part], frame_to_index[frame]] = float(is_missing)

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
)PY";

    ofstream ofs(script_path);
    if (!ofs.is_open()) {
        cerr << "WARNING: Cannot write observation plot script " << script_path << endl;
        return false;
    }
    ofs << script;
    ofs.close();

    string command = "XDG_CACHE_HOME=\"" + xdg_cache_dir + "\" "
                     "MPLCONFIGDIR=\"" + mpl_cache_dir + "\" "
                     "python3 \"" + script_path + "\" \"" + output_dir + "\"";
    int rc = system(command.c_str());
    std::remove(script_path.c_str());

    if (rc != 0) {
        cerr << "WARNING: Observation plot generation failed" << endl;
        return false;
    }

    cout << "Saved observation plots to " << output_dir << endl;
    return true;
}

int main(int argc, char** argv) {
    string gt_path = "xyz_gt.csv";
    vector<MethodSpec> method_specs = {
        {"rigid_no_motion", "./results/xyz_no_motion.csv", false, {}},
        {"rigid_with_motion", "./results/xyz_with_motion.csv", false, {}},
        {"restored_no_motion", "./results/xyz_restored_no_motion.csv", false, {}},
        {"restored_with_motion", "./results/xyz_restored_with_motion.csv", false,
         {"./results/xyz_restored_motion.csv"}}
    };
    vector<ObservationInputSpec> obs_specs = {
        {"front", "3cam/obs_de_front.csv", false},
        {"side", "3cam/obs_de_side.csv", false},
        {"top", "3cam/obs_de_top.csv", false}
    };
    string output_dir = "./results/eval";
    double fps = 30.0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--gt" && i + 1 < argc) {
            gt_path = argv[++i];
        } else if (arg == "--rigid-no-motion" && i + 1 < argc) {
            method_specs[0].path = argv[++i];
            method_specs[0].explicit_path = true;
        } else if (arg == "--rigid-with-motion" && i + 1 < argc) {
            method_specs[1].path = argv[++i];
            method_specs[1].explicit_path = true;
        } else if (arg == "--restored-no-motion" && i + 1 < argc) {
            method_specs[2].path = argv[++i];
            method_specs[2].explicit_path = true;
        } else if (arg == "--restored-with-motion" && i + 1 < argc) {
            method_specs[3].path = argv[++i];
            method_specs[3].explicit_path = true;
        } else if (arg == "--no-motion" && i + 1 < argc) {
            method_specs[2].path = argv[++i];
            method_specs[2].explicit_path = true;
        } else if ((arg == "--motion" || arg == "--with-motion") && i + 1 < argc) {
            method_specs[3].path = argv[++i];
            method_specs[3].explicit_path = true;
        } else if ((arg == "--outdir" || arg == "--output-dir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--front" && i + 1 < argc) {
            obs_specs[0].path = argv[++i];
            obs_specs[0].explicit_path = true;
        } else if (arg == "--side" && i + 1 < argc) {
            obs_specs[1].path = argv[++i];
            obs_specs[1].explicit_path = true;
        } else if (arg == "--top" && i + 1 < argc) {
            obs_specs[2].path = argv[++i];
            obs_specs[2].explicit_path = true;
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --gt <path>                   GT trajectory CSV (default: xyz_gt.csv)\n";
            cout << "  --rigid-no-motion <path>      rigid no-motion CSV (default: ./results/xyz_no_motion.csv)\n";
            cout << "  --rigid-with-motion <path>    rigid with-motion CSV (default: ./results/xyz_with_motion.csv)\n";
            cout << "  --restored-no-motion <path>   restored no-motion CSV (default: ./results/xyz_restored_no_motion.csv)\n";
            cout << "  --restored-with-motion <path> restored with-motion CSV (default: ./results/xyz_restored_with_motion.csv)\n";
            cout << "  --no-motion <path>            Compatibility alias for --restored-no-motion\n";
            cout << "  --motion <path>               Compatibility alias for --restored-with-motion\n";
            cout << "  --with-motion <path>          Compatibility alias for --restored-with-motion\n";
            cout << "  --outdir <path>               Output directory for CSVs and PNGs (default: ./results/eval)\n";
            cout << "  --output-dir <path>           Alias for --outdir\n";
            cout << "  --front <path>                Front camera CSV (default: 3cam/obs_de_front.csv)\n";
            cout << "  --side <path>                 Side camera CSV (default: 3cam/obs_de_side.csv)\n";
            cout << "  --top <path>                  Top camera CSV (default: 3cam/obs_de_top.csv)\n";
            cout << "  --fps <value>                 FPS used when observation CSVs need t<->frame conversion (default: 30.0)\n";
            cout << "  --help, -h                    Show this help\n";
            return 0;
        }
    }

    if (!(fps > 0.0) || !std::isfinite(fps)) {
        cerr << "ERROR: --fps must be a positive finite value" << endl;
        return -1;
    }

    const string executable_dir = get_executable_dir(argv[0]);
    gt_path = resolve_input_path(gt_path, executable_dir);
    cout << "\n=== 3D Evaluation Audit ===\n" << endl;
    cout << "Resolved GT: " << gt_path << endl;
    cout << "Output directory: " << output_dir << endl;
    cout << "Observation FPS: " << fps << endl;

    if (!file_exists(gt_path)) {
        cerr << "ERROR: GT file does not exist: " << gt_path << endl;
        return -1;
    }

    if (!ensure_directory(output_dir)) {
        return -1;
    }

    TrajectoryMap gt = load_trajectory_csv(gt_path);
    if (gt.empty()) {
        cerr << "ERROR: Failed to load GT trajectory" << endl;
        return -1;
    }

    vector<MethodEvaluation> evaluations;
    for (const auto& spec : method_specs) {
        bool used_fallback = false;
        const string resolved_path = resolve_method_input_path(spec, executable_dir, used_fallback);
        if (!file_exists(resolved_path)) {
            if (spec.explicit_path) {
                cerr << "ERROR: Requested input for " << spec.name
                     << " does not exist: " << resolved_path << endl;
                return -1;
            }
            cout << "Skipping " << spec.name << " because no input file was found at "
                 << resolved_path << endl;
            continue;
        }

        if (used_fallback) {
            cout << "Resolved " << spec.name << " via compatibility fallback: "
                 << resolved_path << endl;
        } else {
            cout << "Resolved " << spec.name << ": " << resolved_path << endl;
        }

        TrajectoryMap pred = load_trajectory_csv(resolved_path);
        if (pred.empty()) {
            cerr << "ERROR: Failed to load prediction trajectory for " << spec.name << endl;
            return -1;
        }
        if (!validate_prediction_alignment(spec.name, resolved_path, gt, pred)) {
            return -1;
        }

        evaluations.push_back(evaluate_method(spec.name, resolved_path, gt, pred));
    }

    if (evaluations.empty()) {
        cerr << "ERROR: No prediction inputs were available. Provide at least one rigid/restored trajectory." << endl;
        return -1;
    }

    vector<PointAuditRow> all_point_rows;
    vector<FrameSummaryRow> all_frame_rows;
    vector<PartSummaryRow> all_part_rows;
    vector<OverallSummaryRow> all_overall_rows;

    for (const auto& evaluation : evaluations) {
        cout << evaluation.method
             << ": gt_points=" << evaluation.overall_row.gt_total_points
             << ", matched=" << evaluation.overall_row.matched_points
             << ", coverage=" << format_double_or_empty(evaluation.overall_row.coverage)
             << ", matched_rmse_3d=" << format_double_or_empty(evaluation.overall_row.matched_rmse_3d)
             << endl;

        all_point_rows.insert(all_point_rows.end(),
                              evaluation.point_rows.begin(), evaluation.point_rows.end());
        all_frame_rows.insert(all_frame_rows.end(),
                              evaluation.frame_rows.begin(), evaluation.frame_rows.end());
        all_part_rows.insert(all_part_rows.end(),
                             evaluation.part_rows.begin(), evaluation.part_rows.end());
        all_overall_rows.push_back(evaluation.overall_row);
    }

    sort(all_overall_rows.begin(), all_overall_rows.end(),
         [](const OverallSummaryRow& a, const OverallSummaryRow& b) {
             return method_sort_rank(a.method) < method_sort_rank(b.method);
         });

    sort(all_frame_rows.begin(), all_frame_rows.end(),
         [](const FrameSummaryRow& a, const FrameSummaryRow& b) {
             if (method_sort_rank(a.method) != method_sort_rank(b.method)) {
                 return method_sort_rank(a.method) < method_sort_rank(b.method);
             }
             return a.frame < b.frame;
         });

    sort(all_part_rows.begin(), all_part_rows.end(),
         [](const PartSummaryRow& a, const PartSummaryRow& b) {
             if (method_sort_rank(a.method) != method_sort_rank(b.method)) {
                 return method_sort_rank(a.method) < method_sort_rank(b.method);
             }
             return a.part < b.part;
         });

    const string overall_csv_path = join_path(output_dir, "eval_3d_overall.csv");
    const string frame_audit_csv_path = join_path(output_dir, "eval_3d_frame_audit.csv");
    const string part_csv_path = join_path(output_dir, "eval_3d_part.csv");
    const string point_audit_csv_path = join_path(output_dir, "eval_3d_point_audit.csv");

    save_overall_csv(overall_csv_path, all_overall_rows);
    save_frame_audit_csv(frame_audit_csv_path, all_frame_rows);
    save_part_csv(part_csv_path, all_part_rows);
    save_point_audit_csv(point_audit_csv_path, all_point_rows);

    const bool plots_ok = generate_plots(output_dir);
    ObservationOutputArtifacts observation_outputs;

    vector<ObservationInputSpec> resolved_obs_specs = obs_specs;
    bool observation_inputs_available = true;
    for (auto& spec : resolved_obs_specs) {
        spec.path = resolve_input_path(spec.path, executable_dir);
        if (!file_exists(spec.path)) {
            if (spec.explicit_path) {
                cerr << "ERROR: Requested observation input for " << spec.camera
                     << " does not exist: " << spec.path << endl;
                return -1;
            }
            observation_inputs_available = false;
        }
    }

    if (!observation_inputs_available) {
        cout << "Skipping observation missing visualization because one or more observation CSVs are unavailable." << endl;
    } else {
        cout << "\n=== Observation Missing Visualization ===" << endl;
        for (const auto& spec : resolved_obs_specs) {
            cout << "Resolved " << spec.camera << " observation CSV: " << spec.path << endl;
        }

        ObservationAnalysisResult observation_analysis;
        if (build_observation_analysis(resolved_obs_specs, fps, observation_analysis)) {
            cout << "Observation frames: " << observation_analysis.frames.size()
                 << ", body parts: " << observation_analysis.part_order.size() << endl;
            for (const auto& camera : observation_analysis.camera_order) {
                cout << "  " << camera << " rows: "
                     << observation_analysis.row_counts[camera] << endl;
            }
            observation_outputs.camera_summary_csv_path =
                join_path(output_dir, "obs_camera_summary.csv");
            observation_outputs.missing_per_part_csv_path =
                join_path(output_dir, "obs_missing_per_part.csv");

            save_observation_camera_summary_csv(
                observation_outputs.camera_summary_csv_path, observation_analysis);
            save_observation_missing_per_part_csv(
                observation_outputs.missing_per_part_csv_path, observation_analysis);
            observation_outputs.csvs_generated = true;
            observation_outputs.plots_generated = generate_observation_plots(output_dir);
        } else {
            cout << "Skipping observation missing visualization because no valid observation rows were loaded." << endl;
        }
    }

    cout << "\nGenerated files:" << endl;
    cout << "  - " << overall_csv_path << endl;
    cout << "  - " << frame_audit_csv_path << endl;
    cout << "  - " << part_csv_path << endl;
    cout << "  - " << point_audit_csv_path << endl;
    if (plots_ok) {
        cout << "  - " << join_path(output_dir, "rmse_3d_vs_time.png") << endl;
        cout << "  - " << join_path(output_dir, "coverage_vs_time_no_motion.png") << endl;
        cout << "  - " << join_path(output_dir, "coverage_vs_time_with_motion.png") << endl;
        cout << "  - " << join_path(output_dir, "rmse_3d_per_part.png") << endl;
        cout << "  - " << join_path(output_dir, "rmse_3d_per_part_restored_category.png") << endl;
    }
    if (observation_outputs.csvs_generated) {
        cout << "  - " << observation_outputs.camera_summary_csv_path << endl;
        cout << "  - " << observation_outputs.missing_per_part_csv_path << endl;
    }
    if (observation_outputs.plots_generated) {
        cout << "  - " << join_path(output_dir, "camera_missing_count_vs_time.png") << endl;
        cout << "  - " << join_path(output_dir, "missing_heatmap_all_cameras.png") << endl;
    }

    return 0;
}
