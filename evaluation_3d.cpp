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
};

struct AggregateAccumulator {
    size_t gt_points = 0;
    size_t matched_points = 0;
    size_t missing_points = 0;
    double matched_sum_sq = 0.0;
    double penalized_sum_sq = 0.0;
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
    bool is_large_error = false;
    string segment = "normal";
};

struct FrameSummaryRow {
    string method;
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    size_t gt_num_parts = 0;
    size_t matched_num_parts = 0;
    size_t missing_num_parts = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
    double penalized_rmse_3d = numeric_limits<double>::quiet_NaN();
    string frame_status = "complete";
    string segment = "normal";
    bool high_error = false;
};

struct PartSummaryRow {
    string method;
    string part;
    size_t gt_points = 0;
    size_t matched_points = 0;
    size_t missing_points = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
    double penalized_rmse_3d = numeric_limits<double>::quiet_NaN();
};

struct OverallSummaryRow {
    string method;
    size_t gt_total_points = 0;
    size_t matched_points = 0;
    size_t missing_points = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double missing_rate = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
    double penalized_rmse_3d = numeric_limits<double>::quiet_NaN();
};

struct SegmentAuditRow {
    string method;
    int segment_id = 0;
    long long start_frame = 0;
    long long end_frame = 0;
    size_t length = 0;
    string segment_type;
    string source;
    size_t matched_points = 0;
    size_t missing_points = 0;
    double coverage = numeric_limits<double>::quiet_NaN();
    double matched_rmse_3d = numeric_limits<double>::quiet_NaN();
    double penalized_rmse_3d = numeric_limits<double>::quiet_NaN();
    string main_parts_affected;
};

struct MissingWindow {
    bool valid = false;
    long long start = 0;
    long long end = 0;
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
    vector<SegmentAuditRow> segment_rows;
};

using TrajectoryMap = map<pair<long long, string>, Point3Row>;

static const vector<string> kMethodOrder = {
    "rigid_no_motion",
    "rigid_with_motion",
    "restored_no_motion",
    "restored_with_motion"
};

static const double kMinFrameHighErrorThresholdMm = 10.0;
static const double kMissingPenaltyScaleMultiplier = 0.7;

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

double compute_body_scale(const TrajectoryMap& gt) {
    bool has_point = false;
    double min_x = 0.0, max_x = 0.0;
    double min_y = 0.0, max_y = 0.0;
    double min_z = 0.0, max_z = 0.0;

    for (const auto& kv : gt) {
        const Point3Row& row = kv.second;
        if (!std::isfinite(row.x) || !std::isfinite(row.y) || !std::isfinite(row.z)) {
            continue;
        }

        if (!has_point) {
            min_x = max_x = row.x;
            min_y = max_y = row.y;
            min_z = max_z = row.z;
            has_point = true;
            continue;
        }

        min_x = min(min_x, row.x);
        max_x = max(max_x, row.x);
        min_y = min(min_y, row.y);
        max_y = max(max_y, row.y);
        min_z = min(min_z, row.z);
        max_z = max(max_z, row.z);
    }

    if (!has_point) {
        return numeric_limits<double>::quiet_NaN();
    }

    return max({max_x - min_x, max_y - min_y, max_z - min_z});
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

string format_size_value(size_t value) {
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

string join_strings(const vector<string>& items, const string& delimiter) {
    if (items.empty()) {
        return "";
    }

    ostringstream oss;
    for (size_t i = 0; i < items.size(); i++) {
        if (i > 0) {
            oss << delimiter;
        }
        oss << items[i];
    }
    return oss.str();
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
    if (status == "missing_prediction") return 1;
    if (status == "extra_prediction") return 2;
    return 3;
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
        return kMinFrameHighErrorThresholdMm;
    }

    const double median = compute_median(finite_values);
    vector<double> abs_deviations;
    abs_deviations.reserve(finite_values.size());
    for (double value : finite_values) {
        abs_deviations.push_back(fabs(value - median));
    }

    const double mad = compute_median(abs_deviations);
    const double threshold_stat = median + 3.0 * mad;
    return max(threshold_stat, kMinFrameHighErrorThresholdMm);
}

string detect_auto_segment_type(const FrameSummaryRow& row) {
    const bool has_missing = row.missing_num_parts > 0;
    if (has_missing && row.high_error) {
        return "mixed_window";
    }
    if (has_missing) {
        return "missing_window";
    }
    if (row.high_error) {
        return "high_error_window";
    }
    return "";
}

string build_main_parts_affected(const vector<PointAuditRow>& point_rows,
                                 const string& method,
                                 long long start_frame,
                                 long long end_frame) {
    map<string, size_t> missing_counts;
    map<string, size_t> error_counts;
    map<string, size_t> total_counts;

    for (const auto& row : point_rows) {
        if (row.method != method) {
            continue;
        }
        if (row.frame < start_frame || row.frame > end_frame) {
            continue;
        }
        if (row.status == "extra_prediction") {
            continue;
        }

        if (row.status == "missing_prediction") {
            missing_counts[row.part]++;
            total_counts[row.part]++;
        } else if (row.is_large_error) {
            error_counts[row.part]++;
            total_counts[row.part]++;
        }
    }

    vector<pair<string, size_t>> sorted_counts(total_counts.begin(), total_counts.end());
    sort(sorted_counts.begin(), sorted_counts.end(),
         [&](const pair<string, size_t>& a, const pair<string, size_t>& b) {
             if (a.second != b.second) return a.second > b.second;
             const size_t a_missing = missing_counts[a.first];
             const size_t b_missing = missing_counts[b.first];
             if (a_missing != b_missing) return a_missing > b_missing;
             return a.first < b.first;
         });


    vector<string> top_parts;
    for (size_t i = 0; i < sorted_counts.size() && i < 5; i++) {
        const string& part = sorted_counts[i].first;
        top_parts.push_back(
            part + "(missing=" + format_size_value(missing_counts[part]) +
            ",error=" + format_size_value(error_counts[part]) + ")");
    }
    return join_strings(top_parts, "; ");
}

SegmentAuditRow build_segment_row(const string& method,
                                  int segment_id,
                                  const string& segment_type,
                                  const string& source,
                                  size_t start_index,
                                  size_t end_index,
                                  const vector<FrameSummaryRow>& frame_rows,
                                  const vector<PointAuditRow>& point_rows) {
    SegmentAuditRow row;
    row.method = method;
    row.segment_id = segment_id;
    row.segment_type = segment_type;
    row.source = source;
    row.start_frame = frame_rows[start_index].frame;
    row.end_frame = frame_rows[end_index].frame;
    row.length = end_index - start_index + 1;

    double matched_sum_sq = 0.0;
    double penalized_sum_sq = 0.0;
    size_t gt_points = 0;
    for (size_t i = start_index; i <= end_index; i++) {
        row.matched_points += frame_rows[i].matched_num_parts;
        row.missing_points += frame_rows[i].missing_num_parts;
        gt_points += frame_rows[i].gt_num_parts;

        if (!std::isnan(frame_rows[i].matched_rmse_3d)) {
            matched_sum_sq += frame_rows[i].matched_rmse_3d * frame_rows[i].matched_rmse_3d *
                              static_cast<double>(frame_rows[i].matched_num_parts);
        }
        if (!std::isnan(frame_rows[i].penalized_rmse_3d)) {
            penalized_sum_sq += frame_rows[i].penalized_rmse_3d * frame_rows[i].penalized_rmse_3d *
                                static_cast<double>(frame_rows[i].gt_num_parts);
        }
    }

    row.coverage = safe_ratio(row.matched_points, gt_points);
    row.matched_rmse_3d = safe_rmse(matched_sum_sq, row.matched_points);
    row.penalized_rmse_3d = safe_rmse(penalized_sum_sq, gt_points);
    row.main_parts_affected =
        build_main_parts_affected(point_rows, method, row.start_frame, row.end_frame);
    return row;
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

MethodEvaluation evaluate_method(const string& method,
                                 const string& input_path,
                                 const TrajectoryMap& gt,
                                 const TrajectoryMap& pred,
                                 const MissingWindow& manual_missing_window,
                                 double missing_penalty) {
    MethodEvaluation evaluation;
    evaluation.method = method;
    evaluation.input_path = input_path;

    map<long long, AggregateAccumulator> frame_accumulators;
    map<long long, double> frame_times;
    map<string, AggregateAccumulator> part_accumulators;
    AggregateAccumulator overall_accumulator;

    const double penalty_sq = missing_penalty * missing_penalty;

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

        auto pred_it = pred.find(kv.first);
        if (pred_it != pred.end()) {
            const Point3Row& pred_row = pred_it->second;
            const double error_3d = compute_distance_3d(gt_row, pred_row);
            const double error_sq = error_3d * error_3d;

            row.status = "matched";
            row.t = choose_time_value(gt_row.t, pred_row.t);
            row.pred_x = pred_row.x;
            row.pred_y = pred_row.y;
            row.pred_z = pred_row.z;
            row.error_3d = error_3d;

            frame_acc.matched_points++;
            frame_acc.matched_sum_sq += error_sq;
            frame_acc.penalized_sum_sq += error_sq;

            part_acc.matched_points++;
            part_acc.matched_sum_sq += error_sq;
            part_acc.penalized_sum_sq += error_sq;

            overall_accumulator.matched_points++;
            overall_accumulator.matched_sum_sq += error_sq;
            overall_accumulator.penalized_sum_sq += error_sq;
        } else {
            row.status = "missing_prediction";

            frame_acc.missing_points++;
            frame_acc.penalized_sum_sq += penalty_sq;

            part_acc.missing_points++;
            part_acc.penalized_sum_sq += penalty_sq;

            overall_accumulator.missing_points++;
            overall_accumulator.penalized_sum_sq += penalty_sq;
        }

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
        row.segment = "extra_prediction";
        evaluation.point_rows.push_back(row);
    }

    evaluation.overall_row.method = method;
    evaluation.overall_row.gt_total_points = overall_accumulator.gt_points;
    evaluation.overall_row.matched_points = overall_accumulator.matched_points;
    evaluation.overall_row.missing_points = overall_accumulator.missing_points;
    evaluation.overall_row.coverage = safe_ratio(
        overall_accumulator.matched_points, overall_accumulator.gt_points);
    evaluation.overall_row.missing_rate = safe_ratio(
        overall_accumulator.missing_points, overall_accumulator.gt_points);
    evaluation.overall_row.matched_rmse_3d = safe_rmse(
        overall_accumulator.matched_sum_sq, overall_accumulator.matched_points);
    evaluation.overall_row.penalized_rmse_3d = safe_rmse(
        overall_accumulator.penalized_sum_sq, overall_accumulator.gt_points);

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
        row.missing_num_parts = kv.second.missing_points;
        row.coverage = safe_ratio(kv.second.matched_points, kv.second.gt_points);
        row.matched_rmse_3d = safe_rmse(kv.second.matched_sum_sq, kv.second.matched_points);
        row.penalized_rmse_3d = safe_rmse(kv.second.penalized_sum_sq, kv.second.gt_points);
        evaluation.frame_rows.push_back(row);
    }

    for (const auto& kv : part_accumulators) {
        PartSummaryRow row;
        row.method = method;
        row.part = kv.first;
        row.gt_points = kv.second.gt_points;
        row.matched_points = kv.second.matched_points;
        row.missing_points = kv.second.missing_points;
        row.coverage = safe_ratio(kv.second.matched_points, kv.second.gt_points);
        row.matched_rmse_3d = safe_rmse(kv.second.matched_sum_sq, kv.second.matched_points);
        row.penalized_rmse_3d = safe_rmse(kv.second.penalized_sum_sq, kv.second.gt_points);
        evaluation.part_rows.push_back(row);
    }

    sort(evaluation.part_rows.begin(), evaluation.part_rows.end(),
         [](const PartSummaryRow& a, const PartSummaryRow& b) {
             return a.part < b.part;
         });

    const double high_error_threshold = compute_frame_high_error_threshold(evaluation.frame_rows);
    cout << evaluation.method << " frame high-error threshold (mm): "
         << high_error_threshold << endl;

    for (auto& row : evaluation.point_rows) {
        row.is_large_error =
            row.status == "matched" &&
            !std::isnan(row.error_3d) &&
            row.error_3d > high_error_threshold;
    }

    map<long long, vector<string>> frame_segment_labels;

    for (auto& row : evaluation.frame_rows) {
        row.high_error =
            !std::isnan(row.matched_rmse_3d) && row.matched_rmse_3d > high_error_threshold;

        const bool has_missing = row.missing_num_parts > 0;
        if (row.matched_num_parts == 0) {
            row.frame_status = "all_missing";
        } else if (has_missing && row.high_error) {
            row.frame_status = "mixed_abnormal";
        } else if (has_missing) {
            row.frame_status = "partial_missing";
        } else if (row.high_error) {
            row.frame_status = "large_error";
        } else {
            row.frame_status = "complete";
        }
    }

    int next_segment_id = 1;
    size_t segment_start = 0;
    string current_segment_type;
    for (size_t i = 0; i < evaluation.frame_rows.size(); i++) {
        const string row_segment_type = detect_auto_segment_type(evaluation.frame_rows[i]);
        const bool extend_current =
            !current_segment_type.empty() &&
            row_segment_type == current_segment_type &&
            i > 0 &&
            evaluation.frame_rows[i].frame == evaluation.frame_rows[i - 1].frame + 1;

        if (extend_current) {
            continue;
        }

        if (!current_segment_type.empty()) {
            const size_t segment_end = i - 1;
            evaluation.segment_rows.push_back(build_segment_row(
                method, next_segment_id++, current_segment_type, "auto",
                segment_start, segment_end, evaluation.frame_rows, evaluation.point_rows));
            for (size_t j = segment_start; j <= segment_end; j++) {
                frame_segment_labels[evaluation.frame_rows[j].frame].push_back(current_segment_type);
            }
        }

        if (!row_segment_type.empty()) {
            segment_start = i;
            current_segment_type = row_segment_type;
        } else {
            current_segment_type.clear();
        }
    }

    if (!current_segment_type.empty() && !evaluation.frame_rows.empty()) {
        const size_t segment_end = evaluation.frame_rows.size() - 1;
        evaluation.segment_rows.push_back(build_segment_row(
            method, next_segment_id++, current_segment_type, "auto",
            segment_start, segment_end, evaluation.frame_rows, evaluation.point_rows));
        for (size_t j = segment_start; j <= segment_end; j++) {
            frame_segment_labels[evaluation.frame_rows[j].frame].push_back(current_segment_type);
        }
    }

    if (manual_missing_window.valid && !evaluation.frame_rows.empty()) {
        size_t manual_start_index = evaluation.frame_rows.size();
        size_t manual_end_index = evaluation.frame_rows.size();
        for (size_t i = 0; i < evaluation.frame_rows.size(); i++) {
            if (evaluation.frame_rows[i].frame >= manual_missing_window.start &&
                evaluation.frame_rows[i].frame <= manual_missing_window.end) {
                manual_start_index = min(manual_start_index, i);
                manual_end_index = max(manual_end_index, i);
                frame_segment_labels[evaluation.frame_rows[i].frame].push_back(
                    "manual_missing_window");
            }
        }

        if (manual_start_index < evaluation.frame_rows.size()) {
            evaluation.segment_rows.push_back(build_segment_row(
                method, next_segment_id++, "missing_window", "manual",
                manual_start_index, manual_end_index,
                evaluation.frame_rows, evaluation.point_rows));
        }
    }

    for (auto& row : evaluation.frame_rows) {
        auto labels_it = frame_segment_labels.find(row.frame);
        if (labels_it == frame_segment_labels.end() || labels_it->second.empty()) {
            row.segment = "normal";
            continue;
        }

        vector<string> labels = labels_it->second;
        sort(labels.begin(), labels.end());
        labels.erase(unique(labels.begin(), labels.end()), labels.end());
        row.segment = join_strings(labels, "|");
    }

    for (auto& row : evaluation.point_rows) {
        if (row.status == "extra_prediction") {
            auto labels_it = frame_segment_labels.find(row.frame);
            if (labels_it == frame_segment_labels.end() || labels_it->second.empty()) {
                row.segment = "extra_prediction";
            } else {
                vector<string> labels = labels_it->second;
                sort(labels.begin(), labels.end());
                labels.erase(unique(labels.begin(), labels.end()), labels.end());
                labels.push_back("extra_prediction");
                row.segment = join_strings(labels, "|");
            }
            continue;
        }

        auto labels_it = frame_segment_labels.find(row.frame);
        if (labels_it == frame_segment_labels.end() || labels_it->second.empty()) {
            row.segment = "normal";
        } else {
            vector<string> labels = labels_it->second;
            sort(labels.begin(), labels.end());
            labels.erase(unique(labels.begin(), labels.end()), labels.end());
            row.segment = join_strings(labels, "|");
        }
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

    ofs << "method,gt_total_points,matched_points,missing_points,coverage,missing_rate,matched_rmse_3d,penalized_rmse_3d\n";
    for (const auto& row : overall_rows) {
        ofs << row.method << ","
            << row.gt_total_points << ","
            << row.matched_points << ","
            << row.missing_points << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.missing_rate) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.penalized_rmse_3d) << "\n";
    }

    cout << "Saved overall 3D evaluation to " << path << endl;
}

void save_frame_csv(const string& path,
                    const vector<FrameSummaryRow>& frame_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,frame,t,gt_num_parts,matched_num_parts,missing_num_parts,coverage,matched_rmse_3d,penalized_rmse_3d\n";
    for (const auto& row : frame_rows) {
        ofs << row.method << ","
            << row.frame << ","
            << format_double_or_empty(row.t) << ","
            << row.gt_num_parts << ","
            << row.matched_num_parts << ","
            << row.missing_num_parts << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.penalized_rmse_3d) << "\n";
    }

    cout << "Saved per-frame 3D summary to " << path << endl;
}

void save_frame_audit_csv(const string& path,
                          const vector<FrameSummaryRow>& frame_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,frame,t,gt_num_parts,matched_num_parts,missing_num_parts,coverage,matched_rmse_3d,penalized_rmse_3d,frame_status,segment\n";
    for (const auto& row : frame_rows) {
        ofs << row.method << ","
            << row.frame << ","
            << format_double_or_empty(row.t) << ","
            << row.gt_num_parts << ","
            << row.matched_num_parts << ","
            << row.missing_num_parts << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.penalized_rmse_3d) << ","
            << row.frame_status << ","
            << row.segment << "\n";
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

    ofs << "method,part,gt_points,matched_points,missing_points,coverage,matched_rmse_3d,penalized_rmse_3d\n";
    for (const auto& row : part_rows) {
        ofs << row.method << ","
            << row.part << ","
            << row.gt_points << ","
            << row.matched_points << ","
            << row.missing_points << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.penalized_rmse_3d) << "\n";
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

    ofs << "method,frame,t,part,status,gt_x,gt_y,gt_z,pred_x,pred_y,pred_z,error_3d,is_large_error,segment\n";
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
            << (row.is_large_error ? 1 : 0) << ","
            << row.segment << "\n";
    }

    cout << "Saved point-level audit to " << path << endl;
}

void save_segment_csv(const string& path,
                      const vector<SegmentAuditRow>& segment_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,segment_id,start_frame,end_frame,length,segment_type,source,matched_points,missing_points,coverage,matched_rmse_3d,penalized_rmse_3d\n";
    for (const auto& row : segment_rows) {
        ofs << row.method << ","
            << row.segment_id << ","
            << row.start_frame << ","
            << row.end_frame << ","
            << row.length << ","
            << row.segment_type << ","
            << row.source << ","
            << row.matched_points << ","
            << row.missing_points << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.penalized_rmse_3d) << "\n";
    }

    cout << "Saved segment summary to " << path << endl;
}

void save_segment_audit_csv(const string& path,
                            const vector<SegmentAuditRow>& segment_rows) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "method,segment_id,start_frame,end_frame,length,segment_type,source,matched_points,missing_points,coverage,matched_rmse_3d,penalized_rmse_3d,main_parts_affected\n";
    for (const auto& row : segment_rows) {
        ofs << row.method << ","
            << row.segment_id << ","
            << row.start_frame << ","
            << row.end_frame << ","
            << row.length << ","
            << row.segment_type << ","
            << row.source << ","
            << row.matched_points << ","
            << row.missing_points << ","
            << format_double_or_empty(row.coverage) << ","
            << format_double_or_empty(row.matched_rmse_3d) << ","
            << format_double_or_empty(row.penalized_rmse_3d) << ","
            << row.main_parts_affected << "\n";
    }

    cout << "Saved segment audit to " << path << endl;
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

def build_frame_time_map(frame_rows):
    mapping = {}
    for row in frame_rows:
        try:
            frame = int(row["frame"])
            t = parse_float(row.get("t"))
            if t is not None:
                mapping[frame] = t
        except:
            continue
    return mapping

def frame_to_time(x, mapping):
    if not mapping:
        return x
    keys = sorted(mapping.keys())
    if x <= keys[0]:
        return mapping[keys[0]]
    if x >= keys[-1]:
        return mapping[keys[-1]]
    for i in range(len(keys)-1):
        if keys[i] <= x <= keys[i+1]:
            f0, f1 = keys[i], keys[i+1]
            t0, t1 = mapping[f0], mapping[f1]
            if f1 == f0:
                return t0
            alpha = (x - f0) / (f1 - f0)
            return t0 + alpha * (t1 - t0)
    return x

# ----- New helper functions for vectorized secondary axis -----
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

    # Guard against duplicate or non-monotonic time values.
    keep = np.concatenate(([True], np.diff(times) > 0))
    times = times[keep]
    frames = frames[keep]
    if times.size == 0:
        return arr
    if times.size == 1:
        return np.full_like(arr, frames[0], dtype=float)

    return np.interp(arr, times, frames)

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
    "restored_no_motion": "#2ca02c",
    "restored_with_motion": "#d62728",
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

frame_rows = read_csv(os.path.join(output_dir, "eval_3d_frame.csv"))
frame_time_map = build_frame_time_map(frame_rows)
methods = active_methods(frame_rows)

if not methods:
    methods = method_order

matched_rmse_data = {method: [] for method in methods}
coverage_data = {method: [] for method in methods}
penalized_rmse_data = {method: [] for method in methods}
for row in frame_rows:
    method = row["method"]
    if method not in matched_rmse_data:
        continue
    t = parse_float(row["t"])
    matched_rmse = parse_float(row["matched_rmse_3d"])
    coverage = parse_float(row["coverage"])
    penalized_rmse = parse_float(row["penalized_rmse_3d"])
    if t is not None and matched_rmse is not None:
        matched_rmse_data[method].append((t, matched_rmse))
    if t is not None and coverage is not None:
        coverage_data[method].append((t, coverage))
    if t is not None and penalized_rmse is not None:
        penalized_rmse_data[method].append((t, penalized_rmse))

def plot_time_series_split(data_map, ylabel, title_prefix, output_prefix):
    groups = {
        "rigid": ["rigid_no_motion", "rigid_with_motion"],
        "restored": ["restored_no_motion", "restored_with_motion"],
    }

    for group_name, group_methods in groups.items():
        has_data = any(data_map.get(m) for m in group_methods)
        output_name = f"{output_prefix}_{group_name}.png"
        title = f"{title_prefix} ({group_name.capitalize()})"

        if not has_data:
            save_empty_plot(os.path.join(output_dir, output_name), title, "No data available")
            continue

        plt.figure(figsize=(10, 4.5))
        for method in group_methods:
            if method not in data_map:
                continue
            series = sorted(data_map[method])
            if not series:
                continue
            times = [item[0] for item in series]
            values = [item[1] for item in series]
            plt.plot(
                times,
                values,
                marker="o",
                linewidth=1.5,
                markersize=3.0,
                label=labels[method],
                color=colors[method],
            )

        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        # Add secondary x-axis for frame number if mapping available
        ax = plt.gca()
        if frame_time_map:
            secax = ax.secondary_xaxis(
                'top',
                functions=(
                    lambda t: times_to_frames(t, frame_time_map),
                    lambda f: frames_to_times(f, frame_time_map),
                ),
            )
            secax.set_xlabel("Frame")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_name), dpi=200)
        plt.close()

plot_time_series_split(matched_rmse_data, "Matched 3D RMSE", "Matched 3D RMSE vs Time", "rmse_3d_vs_time")
plot_time_series_split(coverage_data, "Coverage", "Coverage vs Time", "coverage_vs_time")
plot_time_series_split(penalized_rmse_data, "Penalized 3D RMSE", "Penalized 3D RMSE vs Time", "penalized_rmse_3d_vs_time")

part_rows = read_csv(os.path.join(output_dir, "eval_3d_part.csv"))
part_methods = active_methods(part_rows)
if not part_methods:
    part_methods = methods

part_order = []
seen_parts = set()
part_values = {method: {} for method in part_methods}
for row in part_rows:
    method = row["method"]
    part = row["part"]
    rmse = parse_float(row["matched_rmse_3d"])
    if part not in seen_parts:
        seen_parts.add(part)
        part_order.append(part)
    if method in part_values and rmse is not None:
        part_values[method][part] = rmse

if not part_order:
    save_empty_plot(os.path.join(output_dir, "rmse_3d_per_part.png"),
                    "Matched 3D RMSE per Part", "No part data available")
else:
    x = list(range(len(part_order)))
    width = 0.8 / max(len(part_methods), 1)
    plt.figure(figsize=(12, 5))
    for idx, method in enumerate(part_methods):
        values = [part_values[method].get(part, math.nan) for part in part_order]
        offset = (idx - (len(part_methods) - 1) / 2.0) * width
        positions = [item + offset for item in x]
        plt.bar(positions, values, width=width, label=labels.get(method, method),
                color=colors.get(method, "#888888"), alpha=0.85)
    plt.xticks(x, part_order, rotation=35, ha="right")
    plt.ylabel("Matched 3D RMSE")
    plt.title("Matched 3D RMSE per Part")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_3d_per_part.png"), dpi=200)
    plt.close()

segment_rows = read_csv(os.path.join(output_dir, "eval_3d_segment_audit.csv"))
segment_methods = active_methods(segment_rows)
if not segment_methods:
    segment_methods = methods


def plot_segment_summary_split(segment_rows):
    groups = {
        "rigid": ["rigid_no_motion", "rigid_with_motion"],
        "restored": ["restored_no_motion", "restored_with_motion"],
    }

    for group_name, group_methods in groups.items():
        subset = [row for row in segment_rows if row.get("method") in group_methods]
        title = f"Segment Summary ({group_name.capitalize()})"
        output_name = f"segment_summary_{group_name}.png"

        if not subset:
            save_empty_plot(os.path.join(output_dir, output_name), title, "No abnormal segments detected")
            continue

        subset.sort(key=lambda row: (
            group_methods.index(row["method"]),
            int(row["start_frame"]),
            int(row["end_frame"]),
            int(row["segment_id"]),
        ))

        plt.figure(figsize=(12, 3.2))
        y_positions = {method: idx for idx, method in enumerate(group_methods)}

        for row in subset:
            method = row["method"]
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            length = max(1, end - start + 1)
            y = y_positions[method]
            plt.broken_barh(
                [(start, length)],
                (y - 0.35, 0.7),
                facecolors=colors.get(method, "#888888"),
                edgecolors="none",
                alpha=0.9,
            )

        handles = []
        present_methods = {row["method"] for row in subset}
        for method in group_methods:
            if method in present_methods:
                handles.append(
                    plt.Line2D([0], [0], color=colors.get(method, "#888888"), lw=6,
                               label=labels.get(method, method))
                )

        plt.yticks([y_positions[m] for m in group_methods],
                   [labels.get(m, m) for m in group_methods])
        plt.xlabel("Frame")
        plt.title(title)
        plt.grid(True, axis="x", alpha=0.25)
        if handles:
            plt.legend(handles=handles, loc="upper right")
        # Add secondary x-axis for time if mapping available
        ax = plt.gca()
        if frame_time_map:
            secax = ax.secondary_xaxis(
                'top',
                functions=(
                    lambda f: frames_to_times(f, frame_time_map),
                    lambda t: times_to_frames(t, frame_time_map),
                ),
            )
            secax.set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_name), dpi=200)
        plt.close()


plot_segment_summary_split(segment_rows)
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
    "front": "#1f77b4",
    "side": "#ff7f0e",
    "top": "#2ca02c",
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

coverage_output = os.path.join(output_dir, "camera_coverage_vs_time.png")
missing_count_output = os.path.join(output_dir, "camera_missing_count_vs_time.png")

if not summary_rows:
    save_empty_plot(coverage_output, "Camera Coverage vs Time", "No observation summary available")
    save_empty_plot(missing_count_output, "Camera Missing Count vs Time", "No observation summary available")
else:
    coverage_data = {camera: [] for camera in camera_order}
    missing_count_data = {camera: [] for camera in camera_order}

    for row in summary_rows:
        t = parse_float(row.get("t"))
        frame = parse_int(row.get("frame"))
        if t is None and frame is not None:
            t = frame_time_map.get(frame)
        if t is None:
            continue

        for camera in camera_order:
            coverage = parse_float(row.get(f"{camera}_coverage"))
            missing_count = parse_float(row.get(f"{camera}_missing"))
            if coverage is not None:
                coverage_data[camera].append((t, coverage))
            if missing_count is not None:
                missing_count_data[camera].append((t, missing_count))

    plt.figure(figsize=(10, 4.5))
    for camera in camera_order:
        series = coverage_data[camera]
        if not series:
            continue
        times = [item[0] for item in series]
        values = [item[1] for item in series]
        plt.plot(times, values, linewidth=1.8, label=camera_labels[camera],
                 color=camera_colors[camera])
    plt.xlabel("Time (s)")
    plt.ylabel("Coverage")
    plt.title("Camera Coverage vs Time")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    add_frame_axis(plt.gca(), frame_time_map)
    plt.tight_layout()
    plt.savefig(coverage_output, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    for camera in camera_order:
        series = missing_count_data[camera]
        if not series:
            continue
        times = [item[0] for item in series]
        values = [item[1] for item in series]
        plt.plot(times, values, linewidth=1.8, label=camera_labels[camera],
                 color=camera_colors[camera])
    plt.xlabel("Time (s)")
    plt.ylabel("Missing Parts")
    plt.title("Camera Missing Count vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    add_frame_axis(plt.gca(), frame_time_map)
    plt.tight_layout()
    plt.savefig(missing_count_output, dpi=200)
    plt.close()

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

for camera in camera_order:
    output_path = os.path.join(output_dir, f"missing_heatmap_{camera}.png")
    title = f"{camera_labels[camera]} Missing Heatmap"
    if not frames_in_order or not parts_in_order:
        save_empty_plot(output_path, title, "No missing-per-part data available")
        continue

    frame_to_index = {frame: idx for idx, frame in enumerate(frames_in_order)}
    part_to_index = {part: idx for idx, part in enumerate(parts_in_order)}
    matrix = np.zeros((len(parts_in_order), len(frames_in_order)), dtype=float)

    for row in missing_rows:
        if row.get("camera") != camera:
            continue
        frame = parse_int(row.get("frame"))
        part = (row.get("part") or "").strip()
        is_missing = parse_int(row.get("is_missing"))
        if frame not in frame_to_index or part not in part_to_index or is_missing is None:
            continue
        matrix[part_to_index[part], frame_to_index[frame]] = float(is_missing)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(parts_in_order))))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                      cmap=heatmap_cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_ylabel("Body Part")
    ax.set_xlabel("Time (s)")
    ax.set_yticks(np.arange(len(parts_in_order)))
    ax.set_yticklabels(parts_in_order)

    tick_positions = choose_tick_positions(len(frames_in_order))
    bottom_labels = []
    for pos in tick_positions:
        frame = frames_in_order[pos]
        t = frame_time_map.get(frame)
        if t is None:
            bottom_labels.append(str(frame))
        else:
            bottom_labels.append(f"{t:.2f}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(bottom_labels, rotation=0)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xlabel("Frame")
    top_ax.set_xticks(tick_positions)
    top_ax.set_xticklabels([str(frames_in_order[pos]) for pos in tick_positions])

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Observed", "Missing"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
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
    string output_dir = "./results";
    double fps = 30.0;
    MissingWindow missing_window;
    bool missing_start_set = false;
    bool missing_end_set = false;

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
        } else if (arg == "--missing-start" && i + 1 < argc) {
            missing_window.start = stoll(argv[++i]);
            missing_start_set = true;
        } else if (arg == "--missing-end" && i + 1 < argc) {
            missing_window.end = stoll(argv[++i]);
            missing_end_set = true;
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
            cout << "  --outdir <path>               Output directory for CSVs and PNGs (default: ./results)\n";
            cout << "  --output-dir <path>           Alias for --outdir\n";
            cout << "  --front <path>                Front camera CSV (default: 3cam/obs_de_front.csv)\n";
            cout << "  --side <path>                 Side camera CSV (default: 3cam/obs_de_side.csv)\n";
            cout << "  --top <path>                  Top camera CSV (default: 3cam/obs_de_top.csv)\n";
            cout << "  --fps <value>                 FPS used when observation CSVs need t<->frame conversion (default: 30.0)\n";
            cout << "  --missing-start <frame>       Optional manual missing-window start frame\n";
            cout << "  --missing-end <frame>         Optional manual missing-window end frame\n";
            cout << "  --help, -h                    Show this help\n";
            return 0;
        }
    }

    if (missing_start_set != missing_end_set) {
        cerr << "ERROR: --missing-start and --missing-end must be provided together" << endl;
        return -1;
    }
    if (missing_start_set && missing_window.end < missing_window.start) {
        cerr << "ERROR: missing window end must be >= start" << endl;
        return -1;
    }
    if (missing_start_set) {
        missing_window.valid = true;
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
    cout << "Frame high-error geometric floor (mm): "
         << kMinFrameHighErrorThresholdMm << endl;

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

    const double body_scale = compute_body_scale(gt);
    if (!(body_scale > 0.0) || !std::isfinite(body_scale)) {
        cerr << "ERROR: Could not compute a valid GT body scale from the GT bounding box" << endl;
        return -1;
    }

    const double missing_penalty = kMissingPenaltyScaleMultiplier * body_scale;
    cout << "GT body scale (mm): " << body_scale << endl;
    cout << "Auto missing penalty (mm): " << missing_penalty << endl;

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

        evaluations.push_back(evaluate_method(
            spec.name, resolved_path, gt, pred, missing_window, missing_penalty));
    }

    if (evaluations.empty()) {
        cerr << "ERROR: No prediction inputs were available. Provide at least one rigid/restored trajectory." << endl;
        return -1;
    }

    if (missing_window.valid) {
        cout << "Using manual missing window: [" << missing_window.start
             << ", " << missing_window.end << "]" << endl;
    } else {
        cout << "No manual missing window provided; abnormal segments will be auto-detected from frame-level coverage and error." << endl;
    }

    vector<PointAuditRow> all_point_rows;
    vector<FrameSummaryRow> all_frame_rows;
    vector<PartSummaryRow> all_part_rows;
    vector<OverallSummaryRow> all_overall_rows;
    vector<SegmentAuditRow> all_segment_rows;

    for (const auto& evaluation : evaluations) {
        cout << evaluation.method
             << ": gt_points=" << evaluation.overall_row.gt_total_points
             << ", matched=" << evaluation.overall_row.matched_points
             << ", missing=" << evaluation.overall_row.missing_points
             << ", coverage=" << format_double_or_empty(evaluation.overall_row.coverage)
             << ", matched_rmse_3d=" << format_double_or_empty(evaluation.overall_row.matched_rmse_3d)
             << ", penalized_rmse_3d=" << format_double_or_empty(evaluation.overall_row.penalized_rmse_3d)
             << endl;

        all_point_rows.insert(all_point_rows.end(),
                              evaluation.point_rows.begin(), evaluation.point_rows.end());
        all_frame_rows.insert(all_frame_rows.end(),
                              evaluation.frame_rows.begin(), evaluation.frame_rows.end());
        all_part_rows.insert(all_part_rows.end(),
                             evaluation.part_rows.begin(), evaluation.part_rows.end());
        all_overall_rows.push_back(evaluation.overall_row);
        all_segment_rows.insert(all_segment_rows.end(),
                                evaluation.segment_rows.begin(), evaluation.segment_rows.end());
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

    sort(all_segment_rows.begin(), all_segment_rows.end(),
         [](const SegmentAuditRow& a, const SegmentAuditRow& b) {
             if (method_sort_rank(a.method) != method_sort_rank(b.method)) {
                 return method_sort_rank(a.method) < method_sort_rank(b.method);
             }
             return a.segment_id < b.segment_id;
         });

    const string overall_csv_path = join_path(output_dir, "eval_3d_overall.csv");
    const string frame_csv_path = join_path(output_dir, "eval_3d_frame.csv");
    const string frame_audit_csv_path = join_path(output_dir, "eval_3d_frame_audit.csv");
    const string part_csv_path = join_path(output_dir, "eval_3d_part.csv");
    const string point_audit_csv_path = join_path(output_dir, "eval_3d_point_audit.csv");
    const string segment_csv_path = join_path(output_dir, "eval_3d_segment.csv");
    const string segment_audit_csv_path = join_path(output_dir, "eval_3d_segment_audit.csv");

    save_overall_csv(overall_csv_path, all_overall_rows);
    save_frame_csv(frame_csv_path, all_frame_rows);
    save_frame_audit_csv(frame_audit_csv_path, all_frame_rows);
    save_part_csv(part_csv_path, all_part_rows);
    save_point_audit_csv(point_audit_csv_path, all_point_rows);
    save_segment_csv(segment_csv_path, all_segment_rows);
    save_segment_audit_csv(segment_audit_csv_path, all_segment_rows);

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
    cout << "  - " << frame_csv_path << endl;
    cout << "  - " << frame_audit_csv_path << endl;
    cout << "  - " << part_csv_path << endl;
    cout << "  - " << point_audit_csv_path << endl;
    cout << "  - " << segment_csv_path << endl;
    cout << "  - " << segment_audit_csv_path << endl;
    if (plots_ok) {
        cout << "  - " << join_path(output_dir, "rmse_3d_vs_time_rigid.png") << endl;
        cout << "  - " << join_path(output_dir, "rmse_3d_vs_time_restored.png") << endl;
        cout << "  - " << join_path(output_dir, "coverage_vs_time_rigid.png") << endl;
        cout << "  - " << join_path(output_dir, "coverage_vs_time_restored.png") << endl;
        cout << "  - " << join_path(output_dir, "penalized_rmse_3d_vs_time_rigid.png") << endl;
        cout << "  - " << join_path(output_dir, "penalized_rmse_3d_vs_time_restored.png") << endl;
        cout << "  - " << join_path(output_dir, "rmse_3d_per_part.png") << endl;
        cout << "  - " << join_path(output_dir, "segment_summary_rigid.png") << endl;
        cout << "  - " << join_path(output_dir, "segment_summary_restored.png") << endl;
    }
    if (observation_outputs.csvs_generated) {
        cout << "  - " << observation_outputs.camera_summary_csv_path << endl;
        cout << "  - " << observation_outputs.missing_per_part_csv_path << endl;
    }
    if (observation_outputs.plots_generated) {
        cout << "  - " << join_path(output_dir, "camera_coverage_vs_time.png") << endl;
        cout << "  - " << join_path(output_dir, "camera_missing_count_vs_time.png") << endl;
        cout << "  - " << join_path(output_dir, "missing_heatmap_front.png") << endl;
        cout << "  - " << join_path(output_dir, "missing_heatmap_side.png") << endl;
        cout << "  - " << join_path(output_dir, "missing_heatmap_top.png") << endl;
    }

    return 0;
}
