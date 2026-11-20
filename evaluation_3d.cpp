#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    size_t pred_points = 0;
    double sum_sq = 0.0;
    array<size_t, 3> restored_counts = {0, 0, 0};
    array<double, 3> restored_sum_sq = {0.0, 0.0, 0.0};
};

struct PointAuditRow {
    string method;
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    string part;
    double gt_x = numeric_limits<double>::quiet_NaN();
    double gt_y = numeric_limits<double>::quiet_NaN();
    double gt_z = numeric_limits<double>::quiet_NaN();
    double pred_x = numeric_limits<double>::quiet_NaN();
    double pred_y = numeric_limits<double>::quiet_NaN();
    double pred_z = numeric_limits<double>::quiet_NaN();
    double error_3d = numeric_limits<double>::quiet_NaN();
    int restored = 0;
};

struct FrameSummaryRow {
    string method;
    long long frame = 0;
    double t = numeric_limits<double>::quiet_NaN();
    size_t gt_num_parts = 0;
    double rmse_3d = numeric_limits<double>::quiet_NaN();
    int restored = 0;
    int has_missing_camera = 0;
    bool is_high_error = false;
};

struct PartSummaryRow {
    string method;
    string part;
    size_t gt_points = 0;
    double rmse_3d = numeric_limits<double>::quiet_NaN();
    double rmse_restored0 = numeric_limits<double>::quiet_NaN();
    double rmse_restored1 = numeric_limits<double>::quiet_NaN();
    double rmse_restored2 = numeric_limits<double>::quiet_NaN();
    size_t count_restored0 = 0;
    size_t count_restored1 = 0;
    size_t count_restored2 = 0;
};

struct OverallSummaryRow {
    string method;
    size_t gt_total_points = 0;
    size_t matched_points = 0;
    double rmse_3d = numeric_limits<double>::quiet_NaN();
};

struct MissingObs {
    long long frame_id = 0;
    double time_s = numeric_limits<double>::quiet_NaN();
    int cam = -1;
    string part_name;
    int is_camera_missing = 0;
};

struct MethodSpec {
    string name;
    string path;
    bool explicit_path = false;
    vector<string> fallback_paths;
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

vector<MissingObs> load_observation_audit(const string& path) {
    vector<MissingObs> result;
    ifstream ifs(path);
    if (!ifs.is_open()) {
        return result;
    }

    static const unordered_map<string, int> cam_index =
        {{"front", 0}, {"side", 1}, {"top", 2}};

    string line;
    if (!getline(ifs, line)) return result;

    while (getline(ifs, line)) {
        line = remove_bom(line);
        if (trim_copy(line).empty()) continue;
        vector<string> fields = split_csv_line(line);
        if (fields.size() < 5) continue;
        try {
            MissingObs m;
            m.frame_id = stoll(fields[0]);
            m.time_s = stod(fields[1]);
            auto it = cam_index.find(fields[2]);
            m.cam = (it != cam_index.end()) ? it->second : -1;
            m.part_name = fields[3];
            m.is_camera_missing = stoi(fields[4]);
            result.push_back(m);
        } catch (...) {
            continue;
        }
    }
    return result;
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
        if (!std::isnan(row.rmse_3d)) {
            finite_values.push_back(row.rmse_3d);
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
                                 const TrajectoryMap& pred,
                                 const unordered_set<long long>& missing_cam_frames) {
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

        row.t = choose_time_value(gt_row.t, pred_row.t);
        row.pred_x = pred_row.x;
        row.pred_y = pred_row.y;
        row.pred_z = pred_row.z;
        row.error_3d = error_3d;
        row.restored = normalize_restored_value(pred_row.restored);

        frame_acc.pred_points++;
        frame_acc.sum_sq += error_sq;
        frame_restored[gt_row.frame] = max(frame_restored[gt_row.frame], row.restored);

        part_acc.pred_points++;
        part_acc.sum_sq += error_sq;
        part_acc.restored_counts[row.restored]++;
        part_acc.restored_sum_sq[row.restored] += error_sq;

        overall_accumulator.pred_points++;
        overall_accumulator.sum_sq += error_sq;

        evaluation.point_rows.push_back(row);
    }

    evaluation.overall_row.method = method;
    evaluation.overall_row.gt_total_points = overall_accumulator.gt_points;
    evaluation.overall_row.matched_points = overall_accumulator.pred_points;
    evaluation.overall_row.rmse_3d = safe_rmse(
        overall_accumulator.sum_sq, overall_accumulator.pred_points);

    for (const auto& kv : frame_accumulators) {
        FrameSummaryRow row;
        row.method = method;
        row.frame = kv.first;
        auto time_it = frame_times.find(kv.first);
        if (time_it != frame_times.end()) {
            row.t = time_it->second;
        }
        row.gt_num_parts = kv.second.gt_points;
        row.rmse_3d = safe_rmse(kv.second.sum_sq, kv.second.pred_points);
        auto restored_it = frame_restored.find(kv.first);
        if (restored_it != frame_restored.end()) {
            row.restored = restored_it->second;
        }
        row.has_missing_camera = missing_cam_frames.count(kv.first) ? 1 : 0;
        evaluation.frame_rows.push_back(row);
    }

    for (const auto& kv : part_accumulators) {
        PartSummaryRow row;
        row.method = method;
        row.part = kv.first;
        row.gt_points = kv.second.gt_points;
        row.rmse_3d = safe_rmse(kv.second.sum_sq, kv.second.pred_points);
        row.count_restored0 = kv.second.restored_counts[0];
        row.count_restored1 = kv.second.restored_counts[1];
        row.count_restored2 = kv.second.restored_counts[2];
        row.rmse_restored0 = safe_rmse(
            kv.second.restored_sum_sq[0], kv.second.restored_counts[0]);
        row.rmse_restored1 = safe_rmse(
            kv.second.restored_sum_sq[1], kv.second.restored_counts[1]);
        row.rmse_restored2 = safe_rmse(
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

    for (auto& row : evaluation.frame_rows) {
        row.is_high_error =
            !std::isnan(row.rmse_3d) &&
            std::isfinite(high_error_threshold) &&
            row.rmse_3d > high_error_threshold;
    }

    sort(evaluation.point_rows.begin(), evaluation.point_rows.end(),
         [](const PointAuditRow& a, const PointAuditRow& b) {
             if (a.frame != b.frame) return a.frame < b.frame;
             return a.part < b.part;
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

    ofs << "method,gt_total_points,matched_points,rmse_3d\n";
    for (const auto& row : overall_rows) {
        ofs << row.method << ","
            << row.gt_total_points << ","
            << row.matched_points << ","
            << format_double_or_empty(row.rmse_3d) << "\n";
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

    ofs << "method,frame,t,gt_num_parts,rmse_3d,restored,has_missing_camera,is_high_error\n";
    for (const auto& row : frame_rows) {
        ofs << row.method << ","
            << row.frame << ","
            << format_double_or_empty(row.t) << ","
            << row.gt_num_parts << ","
            << format_double_or_empty(row.rmse_3d) << ","
            << row.restored << ","
            << row.has_missing_camera << ","
            << (row.is_high_error ? 1 : 0) << "\n";
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

    ofs << "method,part,gt_points,rmse_3d,"
           "rmse_restored0,rmse_restored1,rmse_restored2,"
           "count_restored0,count_restored1,count_restored2\n";
    for (const auto& row : part_rows) {
        ofs << row.method << ","
            << row.part << ","
            << row.gt_points << ","
            << format_double_or_empty(row.rmse_3d) << ","
            << format_double_or_empty(row.rmse_restored0) << ","
            << format_double_or_empty(row.rmse_restored1) << ","
            << format_double_or_empty(row.rmse_restored2) << ","
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

    ofs << "method,frame,t,part,gt_x,gt_y,gt_z,pred_x,pred_y,pred_z,error_3d,restored\n";
    for (const auto& row : point_rows) {
        ofs << row.method << ","
            << row.frame << ","
            << format_double_or_empty(row.t) << ","
            << row.part << ","
            << format_double_or_empty(row.gt_x) << ","
            << format_double_or_empty(row.gt_y) << ","
            << format_double_or_empty(row.gt_z) << ","
            << format_double_or_empty(row.pred_x) << ","
            << format_double_or_empty(row.pred_y) << ","
            << format_double_or_empty(row.pred_z) << ","
            << format_double_or_empty(row.error_3d) << ","
            << row.restored << "\n";
    }

    cout << "Saved point-level audit to " << path << endl;
}

int main(int argc, char** argv) {
    string gt_path = "xyz_gt.csv";
    vector<MethodSpec> method_specs = {
        {"restored_no_motion", "./results/xyz_restored_no_motion.csv", false, {}},
        {"restored_with_motion", "./results/xyz_restored_with_motion.csv", false,
         {"./results/xyz_restored_motion.csv"}}
    };
    string output_dir = "./results";

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--gt" && i + 1 < argc) {
            gt_path = argv[++i];
        } else if (arg == "--restored-no-motion" && i + 1 < argc) {
            method_specs[0].path = argv[++i];
            method_specs[0].explicit_path = true;
        } else if (arg == "--restored-with-motion" && i + 1 < argc) {
            method_specs[1].path = argv[++i];
            method_specs[1].explicit_path = true;
        } else if (arg == "--no-motion" && i + 1 < argc) {
            method_specs[0].path = argv[++i];
            method_specs[0].explicit_path = true;
        } else if ((arg == "--motion" || arg == "--with-motion") && i + 1 < argc) {
            method_specs[1].path = argv[++i];
            method_specs[1].explicit_path = true;
        } else if ((arg == "--outdir" || arg == "--output-dir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --gt <path>                   GT trajectory CSV (default: xyz_gt.csv)\n";
            cout << "  --restored-no-motion <path>   restored no-motion CSV (default: ./results/xyz_restored_no_motion.csv)\n";
            cout << "  --restored-with-motion <path> restored with-motion CSV (default: ./results/xyz_restored_with_motion.csv)\n";
            cout << "  --no-motion <path>            Compatibility alias for --restored-no-motion\n";
            cout << "  --motion <path>               Compatibility alias for --restored-with-motion\n";
            cout << "  --with-motion <path>          Compatibility alias for --restored-with-motion\n";
            cout << "  --outdir <path>               Output directory for CSVs (default: ./results/eval)\n";
            cout << "  --output-dir <path>           Alias for --outdir\n";
            cout << "  --help, -h                    Show this help\n";
            return 0;
        }
    }

    const string executable_dir = get_executable_dir(argv[0]);
    gt_path = resolve_input_path(gt_path, executable_dir);
    cout << "\n=== 3D Evaluation Audit ===\n" << endl;
    cout << "Resolved GT: " << gt_path << endl;
    cout << "Output directory: " << output_dir << endl;

    if (!file_exists(gt_path)) {
        cerr << "ERROR: GT file does not exist: " << gt_path << endl;
        return -1;
    }

    if (!ensure_directory(output_dir)) {
        return -1;
    }

    unordered_set<long long> missing_cam_frames;
    {
        // const string audit_path_primary = resolve_input_path(
        //     "results/diagnostics/anomaly_missing_obs.csv", executable_dir);
        // const string audit_path_fallback = resolve_input_path(
        //     "results/anomaly_missing_obs.csv", executable_dir);
        

        const string audit_path_primary = join_path(
            join_path(output_dir, "diagnostics"), "anomaly_missing_obs.csv");
        const string audit_path_fallback = join_path(
            output_dir, "anomaly_missing_obs.csv");

        const string audit_path = file_exists(audit_path_primary)
            ? audit_path_primary
            : audit_path_fallback;
        if (file_exists(audit_path)) {
            vector<MissingObs> audit = load_observation_audit(audit_path);
            for (const auto& m : audit) {
                if (m.is_camera_missing) {
                    missing_cam_frames.insert(m.frame_id);
                }
            }
            cout << "Loaded observation audit: "
                 << missing_cam_frames.size()
                 << " frames with camera-level missing detections." << endl;
        } else {
            cout << "No observation audit found at " << audit_path
                 << "; has_missing_camera will be 0 for all frames." << endl;
        }
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

        evaluations.push_back(evaluate_method(spec.name, resolved_path, gt, pred,
                                              missing_cam_frames));
    }

    if (evaluations.empty()) {
        cerr << "ERROR: No prediction inputs were available. Provide at least one rigid/restored trajectory." << endl;
        return -1;
    }

    // ── init基线评估(可选、自动探测,不需要CLI参数) ──
    // init不是一种"求解方法",是BA求解前的起点;若diagnostics/xyz_init.csv存在就顺手
    // 评估一次,单独存到diagnostics/,不参与method_sort_rank排序,不写入主三张表。
    {
        const string init_xyz_candidate = resolve_input_path(
            "results/diagnostics/xyz_init.csv", executable_dir);
        const string init_xyz_fallback = join_path(
            join_path(output_dir, "diagnostics"), "xyz_init.csv");
        const string resolved_init_path = file_exists(init_xyz_candidate)
            ? init_xyz_candidate
            : init_xyz_fallback;

        if (!file_exists(resolved_init_path)) {
            cout << "No xyz_init.csv found at " << resolved_init_path
                 << "; skipping init baseline evaluation." << endl;
        } else {
            TrajectoryMap init_pred = load_trajectory_csv(resolved_init_path);
            if (init_pred.empty()) {
                cout << "WARNING: Failed to load xyz_init.csv; skipping init evaluation." << endl;
            } else if (!validate_prediction_alignment("init", resolved_init_path, gt, init_pred)) {
                cout << "WARNING: xyz_init.csv failed alignment check; skipping init evaluation." << endl;
            } else {
                MethodEvaluation init_eval = evaluate_method(
                    "init", resolved_init_path, gt, init_pred, missing_cam_frames);

                cout << "init: gt_points=" << init_eval.overall_row.gt_total_points
                     << ", rmse_3d=" << format_double_or_empty(init_eval.overall_row.rmse_3d)
                     << "  (reference baseline, not part of method comparison)" << endl;

                const string diag_dir = join_path(output_dir, "diagnostics");
                ensure_directory(diag_dir);

                vector<OverallSummaryRow> init_overall_rows = {init_eval.overall_row};
                vector<FrameSummaryRow> init_frame_rows = init_eval.frame_rows;
                sort(init_frame_rows.begin(), init_frame_rows.end(),
                     [](const FrameSummaryRow& a, const FrameSummaryRow& b) {
                         return a.frame < b.frame;
                     });
                vector<PartSummaryRow> init_part_rows = init_eval.part_rows;
                sort(init_part_rows.begin(), init_part_rows.end(),
                     [](const PartSummaryRow& a, const PartSummaryRow& b) {
                         return a.part < b.part;
                     });

                save_overall_csv(join_path(diag_dir, "eval_3d_init_overall.csv"), init_overall_rows);
                save_frame_audit_csv(join_path(diag_dir, "eval_3d_init_frame.csv"), init_frame_rows);
                save_part_csv(join_path(diag_dir, "eval_3d_init_part.csv"), init_part_rows);

                cout << "Saved init baseline evaluation to " << diag_dir
                     << "/eval_3d_init_{overall,frame,part}.csv" << endl;
            }
        }
    } 

    vector<PointAuditRow> all_point_rows;
    vector<FrameSummaryRow> all_frame_rows;
    vector<PartSummaryRow> all_part_rows;
    vector<OverallSummaryRow> all_overall_rows;

    for (const auto& evaluation : evaluations) {
        cout << evaluation.method
             << ": gt_points=" << evaluation.overall_row.gt_total_points
             << ", rmse_3d=" << format_double_or_empty(evaluation.overall_row.rmse_3d)
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

    cout << "\n=== All done ===" << endl;
    return 0;
}
