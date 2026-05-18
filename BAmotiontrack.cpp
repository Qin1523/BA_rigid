#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <cctype>
#include <cerrno>
#include <sys/stat.h>
#include <limits.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR ""
#endif

// ============================================================================
// Data structures
// ============================================================================

// A single 2D observation.
struct Obs2 {
    int cam;        // Camera index: 0 / 1 / 2
    string part;    // Body-part label
    double u, v;    // 2D image coordinates
    double t;       // Timestamp in seconds
    long long frame; // Explicit frame index if present, otherwise -1
};

// ============================================================================
// 1) Input loading and preprocessing
// ============================================================================

string remove_bom(const string& str);

bool file_exists(const string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
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
        if (std::isalnum(ch)) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return normalized;
}

// Return the first column index whose normalised name matches any key; -1 if not found.
int find_column_index(const vector<string>& header_fields,
                      const vector<string>& keys) {
    for (size_t i = 0; i < header_fields.size(); i++) {
        string name = normalize_header_name(header_fields[i]);
        for (const auto& key : keys) {
            if (name == key) return static_cast<int>(i);
        }
    }
    return -1;
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

bool directory_exists(const string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
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

// 1.1 Load the first three camera projection matrices.
vector<cv::Mat> load_Ps_txt_first3(const string& cams_path) {
    vector<cv::Mat> Ps;
    ifstream ifs(cams_path);
    if (!ifs.is_open()) {
        cerr << "ERROR: Cannot open " << cams_path << endl;
        return Ps;
    }
    
    string line;
    int count = 0;
    int line_num = 0;
    
    while (getline(ifs, line) && count < 3) {
        line_num++;
        
        // Trim leading and trailing whitespace.
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comment lines.
        if (line.empty() || line[0] == '#') {
            cout << "Skipping line " << line_num << ": " 
                 << (line.empty() ? "(empty)" : line) << endl;
            continue;
        }
        
        stringstream ss(line);
        vector<double> vals;
        double val;
        while (ss >> val) {
            vals.push_back(val);
        }
        
        if (vals.size() >= 12) {
            cv::Mat P(3, 4, CV_64F);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    P.at<double>(i, j) = vals[i * 4 + j];
                }
            }
            Ps.push_back(P);
            count++;
            
            cout << "Loaded camera " << count << " from line " << line_num 
                 << " (" << vals.size() << " values)" << endl;
        } else {
            cerr << "WARNING: Line " << line_num << " has only " << vals.size() 
                 << " values (need 12), skipping" << endl;
        }
    }
    
    cout << "\nSuccessfully loaded " << Ps.size() << " camera projection matrices" << endl;
    
    if (Ps.size() < 3) {
        cerr << "\nERROR: Only found " << Ps.size() << " valid cameras, need 3" << endl;
        cerr << "Please check your cams.txt format:" << endl;
        cerr << "  - Each camera should be on one line" << endl;
        cerr << "  - Each line should have 12 numbers (3x4 projection matrix)" << endl;
        cerr << "  - Lines starting with # are treated as comments" << endl;
    }
    
    return Ps;
}

// Remove a UTF-8 BOM if present.
string remove_bom(const string& str) {
    if (str.size() >= 3 && 
        (unsigned char)str[0] == 0xEF && 
        (unsigned char)str[1] == 0xBB && 
        (unsigned char)str[2] == 0xBF) {
        return str.substr(3);
    }
    return str;
}

// 1.2 Load 2D observations for one camera.
vector<Obs2> load_obs_csv(const string& path, int cam_id) {
    vector<Obs2> obs;
    ifstream ifs(path);
    if (!ifs.is_open()) {
        cerr << "WARNING: Cannot open " << path << endl;
        return obs;
    }
    
    string line;
    if (!getline(ifs, line)) {
        cout << "Loaded 0 observations from cam " << cam_id << endl;
        return obs;
    }

    vector<string> header_fields = split_csv_line(remove_bom(line));
    int frame_col = find_column_index(header_fields, {"frame"});
    int t_col     = find_column_index(header_fields, {"t", "time", "timestamp"});
    int part_col  = find_column_index(header_fields, {"part", "bodypart"});
    int x_col     = find_column_index(header_fields, {"x", "u"});
    int y_col     = find_column_index(header_fields, {"y", "v"});

    if (t_col < 0 || part_col < 0 || x_col < 0 || y_col < 0) {
        cerr << "ERROR: CSV header in " << path
             << " must contain at least t, part, x, y columns" << endl;
        return obs;
    }

    while (getline(ifs, line)) {
        line = remove_bom(line);
        if (trim_copy(line).empty()) continue;

        vector<string> fields = split_csv_line(line);
        const int required_col = max({frame_col, t_col, part_col, x_col, y_col});
        if (required_col >= static_cast<int>(fields.size())) {
            continue;
        }

        try {
            long long frame = -1;
            if (frame_col >= 0 && !fields[frame_col].empty()) {
                frame = stoll(fields[frame_col]);
            }

            double t = stod(fields[t_col]);
            const string& part = fields[part_col];
            double x = stod(fields[x_col]);
            double y = stod(fields[y_col]);

            obs.push_back({cam_id, part, x, y, t, frame});
        } catch (...) {
            // Skip malformed rows.
            continue;
        }
    }
    
    cout << "Loaded " << obs.size() << " observations from cam " << cam_id << endl;
    return obs;
}

// Group observations by frame and record the first-seen part order.
map<long long, vector<Obs2>> group_obs_by_frame(const vector<Obs2>& obsAll, double fps,
                                                  vector<string>& part_order) {
    map<long long, vector<Obs2>> obs_by_frame;
    unordered_map<string, bool> part_seen;
    
    for (const auto& ob : obsAll) {
        long long frame = (ob.frame >= 0) ? ob.frame : (long long)round(ob.t * fps);
        obs_by_frame[frame].push_back(ob);
        
        // Preserve the first appearance order for stable CSV output.
        if (!part_seen[ob.part]) {
            part_seen[ob.part] = true;
            part_order.push_back(ob.part);
        }
    }
    
    cout << "Grouped into " << obs_by_frame.size() << " frames" << endl;
    cout << "Part order (" << part_order.size() << " parts): ";
    for (size_t i = 0; i < part_order.size(); i++) {
        cout << part_order[i];
        if (i < part_order.size() - 1) cout << ", ";
    }
    cout << endl;
    
    return obs_by_frame;
}

// Infer the frame step from consecutive frame-number differences.
long long infer_frame_step(const map<long long, vector<Obs2>>& obs_by_frame) {
    if (obs_by_frame.size() < 2) return 1;
    long long min_step = LLONG_MAX;
    auto it = obs_by_frame.begin();
    auto prev_it = it++;
    for (; it != obs_by_frame.end(); ++it, ++prev_it) {
        long long diff = it->first - prev_it->first;
        if (diff > 0 && diff < min_step) min_step = diff;
    }
    return (min_step == LLONG_MAX) ? 1 : min_step;
}

// Return the set of frame numbers that are absent from obs_by_frame but lie
// within [first, last] on the inferred regular grid.
vector<long long> find_gap_frames(
    const map<long long, vector<Obs2>>& obs_by_frame,
    long long frame_step) {

    vector<long long> gaps;
    if (obs_by_frame.empty()) return gaps;

    long long first = obs_by_frame.begin()->first;
    long long last  = obs_by_frame.rbegin()->first;

    for (long long fr = first; fr <= last; fr += frame_step) {
        if (obs_by_frame.find(fr) == obs_by_frame.end())
            gaps.push_back(fr);
    }
    return gaps;
}

// ============================================================================
// 2) Triangulation and template I/O helpers
// ============================================================================

// Triangulate one 3D point with linear DLT.
cv::Point3d triangulate_DLT(const vector<pair<int, cv::Point2d>>& views, 
                             const vector<cv::Mat>& Ps) {
    int n = views.size();
    if (n < 2) {
        return cv::Point3d(0, 0, 0);
    }
    
    cv::Mat A(2 * n, 4, CV_64F);
    
    for (int i = 0; i < n; i++) {
        int cam = views[i].first;
        double u = views[i].second.x;
        double v = views[i].second.y;
        
        const cv::Mat& P = Ps[cam];
        
        // u * P3 - P1
        for (int j = 0; j < 4; j++) {
            A.at<double>(2*i, j) = u * P.at<double>(2, j) - P.at<double>(0, j);
        }

        // v * P3 - P2
        for (int j = 0; j < 4; j++) {
            A.at<double>(2*i+1, j) = v * P.at<double>(2, j) - P.at<double>(1, j);
        }
    }
    
    cv::Mat w, u_mat, vt;
    cv::SVD::compute(A, w, u_mat, vt);
    
    cv::Mat X_homo = vt.row(3).t();
    double W = X_homo.at<double>(3, 0);
    
    if (fabs(W) < 1e-10) {
        return cv::Point3d(0, 0, 0);
    }
    
    return cv::Point3d(
        X_homo.at<double>(0, 0) / W,
        X_homo.at<double>(1, 0) / W,
        X_homo.at<double>(2, 0) / W
    );
}

unordered_map<string, array<double, 3>> load_template(const string& path) {
    unordered_map<string, array<double, 3>> model_obj;
    ifstream ifs(path);
    if (!ifs.is_open()) {
        cerr << "ERROR: Cannot open template file " << path << endl;
        return model_obj;
    }

    string line;
    if (!getline(ifs, line)) {
        cerr << "ERROR: Empty template file " << path << endl;
        return model_obj;
    }

    vector<string> header_fields = split_csv_line(remove_bom(line));
    int part_col = find_column_index(header_fields, {"part", "bodypart"});
    int x_col    = find_column_index(header_fields, {"x"});
    int y_col    = find_column_index(header_fields, {"y"});
    int z_col    = find_column_index(header_fields, {"z"});

    if (part_col < 0 || x_col < 0 || y_col < 0 || z_col < 0) {
        cerr << "ERROR: Template header in " << path
             << " must contain part, X, Y, Z columns" << endl;
        return model_obj;
    }

    while (getline(ifs, line)) {
        line = remove_bom(line);
        if (trim_copy(line).empty()) continue;

        vector<string> fields = split_csv_line(line);
        const int required_col = max({part_col, x_col, y_col, z_col});
        if (required_col >= static_cast<int>(fields.size())) {
            continue;
        }

        try {
            const string& part = fields[part_col];
            double x = stod(fields[x_col]);
            double y = stod(fields[y_col]);
            double z = stod(fields[z_col]);
            model_obj[part] = {x, y, z};
        } catch (...) {
            continue;
        }
    }

    cout << "Loaded " << model_obj.size() << " template points from " << path << endl;
    return model_obj;
}

double distance_3d(const array<double, 3>& a,
                   const array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Build one translation seed per segment of segment_length consecutive frames.
bool estimate_initial_translation_from_sequence(
    const map<long long, vector<Obs2>>& obs_by_frame,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<cv::Mat>& Ps,
    map<long long, array<double, 3>>& seed_map,
    int segment_length = 150) {

    seed_map.clear();

    if (obs_by_frame.empty()) {
        cerr << "ERROR: No frames available for translation seed estimation" << endl;
        return false;
    }
    if (!model_obj.count("head center")) {
        cerr << "ERROR: Template does not contain 'head center'" << endl;
        return false;
    }

    long long first_frame = obs_by_frame.begin()->first;
    long long last_frame  = obs_by_frame.rbegin()->first;
    long long total_frames = last_frame - first_frame + 1;
    int num_segments = static_cast<int>(
        ceil(static_cast<double>(total_frames) / segment_length));

    for (int seg = 0; seg < num_segments; seg++) {
        long long seg_start = first_frame + seg * segment_length;
        long long seg_end   = seg_start + segment_length - 1;

        for (const auto& kv : obs_by_frame) {
            if (kv.first < seg_start || kv.first > seg_end) continue;

            vector<pair<int, cv::Point2d>> hc_views;
            for (const auto& ob : kv.second) {
                if (ob.part == "head center")
                    hc_views.push_back({ob.cam, cv::Point2d(ob.u, ob.v)});
            }
            if (hc_views.size() < 2) continue;

            cv::Point3d X_hc = triangulate_DLT(hc_views, Ps);
            if (!std::isfinite(X_hc.x) || !std::isfinite(X_hc.y) || !std::isfinite(X_hc.z))
                continue;

            seed_map[seg_start] = {X_hc.x, X_hc.y, X_hc.z};
            cout << "Segment [" << seg_start << ", " << seg_end
                 << "] seed from frame " << kv.first << ": ("
                 << X_hc.x << ", " << X_hc.y << ", " << X_hc.z << ")" << endl;
            break;
        }
    }

    if (seed_map.empty()) {
        cerr << "ERROR: Could not find any valid head center for translation seeding" << endl;
        return false;
    }
    cout << "Total translation seeds: " << seed_map.size()
         << " (out of " << num_segments << " segments)" << endl;
    return true;
}

// Look up the nearest translation seed for a given frame number.
array<double, 3> lookup_translation_seed(
    long long frame,
    const map<long long, array<double, 3>>& seed_map) {

    if (seed_map.empty()) return {0.0, 0.0, 0.0};

    auto it = seed_map.lower_bound(frame);

    if (it == seed_map.begin()) return it->second;
    if (it == seed_map.end())   return prev(it)->second;

    auto prev_it = prev(it);
    long long dist_prev = frame - prev_it->first;
    long long dist_next = it->first - frame;
    return (dist_prev <= dist_next) ? prev_it->second : it->second;
}

array<double, 3> compute_template_centroid_obj(
    const unordered_map<string, array<double, 3>>& model_obj) {

    array<double, 3> centroid = {0.0, 0.0, 0.0};
    if (model_obj.empty()) {
        return centroid;
    }

    for (const auto& kv : model_obj) {
        centroid[0] += kv.second[0];
        centroid[1] += kv.second[1];
        centroid[2] += kv.second[2];
    }

    const double inv_count = 1.0 / static_cast<double>(model_obj.size());
    centroid[0] *= inv_count;
    centroid[1] *= inv_count;
    centroid[2] *= inv_count;
    return centroid;
}

bool compute_template_body_scale_stats(
    const unordered_map<string, array<double, 3>>& model_obj,
    double& head_to_tail_base,
    double& tail_base_to_tail_root) {

    head_to_tail_base = -1.0;
    tail_base_to_tail_root = -1.0;
    bool has_any_stat = false;

    auto head_center_it = model_obj.find("head center");
    auto tail_root_it = model_obj.find("tail root");
    auto tail_base_it = model_obj.find("tail base");

    if (head_center_it != model_obj.end() && tail_base_it != model_obj.end()) {
        head_to_tail_base = distance_3d(head_center_it->second, tail_base_it->second);
        has_any_stat = true;
    }

    if (tail_root_it != model_obj.end() && tail_base_it != model_obj.end()) {
        tail_base_to_tail_root = distance_3d(tail_base_it->second,tail_root_it->second);
        has_any_stat = true;
    }

    return has_any_stat;
}

// ============================================================================
// 3) Ceres cost functions
// ============================================================================

// Reprojection residual for one observation with a fixed template point.
struct PoseReprojCost {
    PoseReprojCost(const cv::Mat& P, const array<double, 3>& X_obj,
                   double u_obs, double v_obs)
        : X_obj_(X_obj), u_obs_(u_obs), v_obs_(v_obs) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                P_[i * 4 + j] = P.at<double>(i, j);
    }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // pose = [angle_axis(3), translation(3)]
        const T* aa = pose;
        const T* trans = pose + 3;

        // Cast the template point to the autodiff scalar type.
        T X_obj[3];
        X_obj[0] = T(X_obj_[0]);
        X_obj[1] = T(X_obj_[1]);
        X_obj[2] = T(X_obj_[2]);

        // Apply rotation.
        T X_world[3];
        ceres::AngleAxisRotatePoint(aa, X_obj, X_world);

        // Apply translation.
        X_world[0] += trans[0];
        X_world[1] += trans[1];
        X_world[2] += trans[2];

        // Project to the image plane.
        T X_homo[4] = {X_world[0], X_world[1], X_world[2], T(1.0)};

        T u_pred = T(0.0), v_pred = T(0.0), w_pred = T(0.0);

        for (int j = 0; j < 4; j++) {
            u_pred += T(P_[0 * 4 + j]) * X_homo[j];
            v_pred += T(P_[1 * 4 + j]) * X_homo[j];
            w_pred += T(P_[2 * 4 + j]) * X_homo[j];
        }

        u_pred /= w_pred;
        v_pred /= w_pred;

        // Residual in image space.
        residuals[0] = u_pred - T(u_obs_);
        residuals[1] = v_pred - T(v_obs_);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Mat& P,
                                       const array<double, 3>& X_obj,
                                       double u, double v) {
        return new ceres::AutoDiffCostFunction<PoseReprojCost, 2, 6>(
            new PoseReprojCost(P, X_obj, u, v));
    }

private:
    double P_[12];
    array<double, 3> X_obj_;
    double u_obs_, v_obs_;
};

// Four-point cubic interpolation used to predict the center frame from
// t-2, t-1, t+1, and t+2.
template <typename T>
void cubic_spline_predict_center(const T* p_m2, const T* p_m1,
                                  const T* p_p1, const T* p_p2,
                                  T* result, int dim) {
    // The sample times are fixed at -2, -1, +1, +2 relative to the center
    // frame t=0, so the cubic Lagrange weights at t=0 reduce to constants.
    // l0(0) = (0-(-1))*(0-1)*(0-2) / ((-2)-(-1))*((-2)-1)*((-2)-2) = 1*(-1)*(-2)/((-1)*(-3)*(-4)) = -2/12 = -1/6
    // l1(0) = (0-(-2))*(0-1)*(0-2) / ((-1)-(-2))*((-1)-1)*((-1)-2) = 2*(-1)*(-2)/(1*(-2)*(-3)) = 4/6 = 2/3
    // l2(0) = (0-(-2))*(0-(-1))*(0-2) / (1-(-2))*(1-(-1))*(1-2) = 2*1*(-2)/(3*2*(-1)) = -4/(-6) = 2/3
    // l3(0) = (0-(-2))*(0-(-1))*(0-1) / (2-(-2))*(2-(-1))*(2-1) = 2*1*(-1)/(4*3*1) = -2/12 = -1/6

    T w0 = T(-1.0/6.0);
    T w1 = T(2.0/3.0);
    T w2 = T(2.0/3.0);
    T w3 = T(-1.0/6.0);

    for (int i = 0; i < dim; i++) {
        result[i] = w0 * p_m2[i] + w1 * p_m1[i] + w2 * p_p1[i] + w3 * p_p2[i];
    }
}

// Motion-track grid constraint with a five-frame stencil.
struct MotionTrackGridCost {
    MotionTrackGridCost(double weight, double grid_step,
                        const array<double, 3>& grid_center_obj,
                        double body_scale)
        : weight_(weight), grid_step_(grid_step), grid_center_obj_(grid_center_obj),
          body_scale_(body_scale > 1e-12 ? body_scale : 1.0) {}

    template <typename T>
    bool operator()(const T* const pose_m2, const T* const pose_m1,
                    const T* const pose_0, const T* const pose_p1,
                    const T* const pose_p2, T* residuals) const {

        // Predict the center pose from its four neighbors without using pose_0.
        T pose_spline[6];
        cubic_spline_predict_center(pose_m2, pose_m1, pose_p1, pose_p2,
                                     pose_spline, 6);

        // Compare the actual pose and the predicted pose on a 3x3x3 grid.
        int idx = 0;
        for (int ix = -1; ix <= 1; ix++) {
            for (int iy = -1; iy <= 1; iy++) {
                for (int iz = -1; iz <= 1; iz++) {
                    // Grid point in the object frame.
                    T grid_pt[3] = {
                        T(grid_center_obj_[0] + ix * grid_step_),
                        T(grid_center_obj_[1] + iy * grid_step_),
                        T(grid_center_obj_[2] + iz * grid_step_)
                    };

                    // Transform with the actual center-frame pose.
                    T X_actual[3];
                    ceres::AngleAxisRotatePoint(pose_0, grid_pt, X_actual);
                    X_actual[0] += pose_0[3];
                    X_actual[1] += pose_0[4];
                    X_actual[2] += pose_0[5];

                    // Transform with the spline-predicted pose.
                    T X_pred[3];
                    ceres::AngleAxisRotatePoint(pose_spline, grid_pt, X_pred);
                    X_pred[0] += pose_spline[3];
                    X_pred[1] += pose_spline[4];
                    X_pred[2] += pose_spline[5];

                    // Residual in 3D space, normalized by body_scale_.
                    residuals[idx++] = T(weight_) * (X_actual[0] - X_pred[0]) / T(body_scale_);
                    residuals[idx++] = T(weight_) * (X_actual[1] - X_pred[1]) / T(body_scale_);
                    residuals[idx++] = T(weight_) * (X_actual[2] - X_pred[2]) / T(body_scale_);
                }
            }
        }

        return true;
    }

    static ceres::CostFunction* Create(double weight, double grid_step,
                                       const array<double, 3>& grid_center_obj,
                                       double body_scale) {
        return new ceres::AutoDiffCostFunction<MotionTrackGridCost, 81, 6, 6, 6, 6, 6>(
            new MotionTrackGridCost(weight, grid_step, grid_center_obj, body_scale));
    }

private:
    double weight_;
    double grid_step_;
    array<double, 3> grid_center_obj_;
    double body_scale_;
};

// ============================================================================
// 4) Per-frame pose initialization
// ============================================================================

void initialize_pose_per_frame(
    int frame_idx,
    const vector<Obs2>& obs_frame,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<cv::Mat>& Ps,
    double* pose_init){

    // Require enough distinct parts before solving the single-frame problem.
    unordered_map<string, int> part_count;
    for (const auto& ob : obs_frame) {
        if (model_obj.count(ob.part)) {
            part_count[ob.part]++;
        }
    }

    if (part_count.size() < 3) {
        // Keep the current seed if the frame is too sparse.
        return;
    }

    ceres::Problem prob;
    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

    for (const auto& ob : obs_frame) {
        if (!model_obj.count(ob.part)) continue;

        const auto& X_obj = model_obj.at(ob.part);

        ceres::CostFunction* cost = PoseReprojCost::Create(
            Ps[ob.cam], X_obj, ob.u, ob.v);

        prob.AddResidualBlock(cost, loss, pose_init);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 5; // Short local refinement for warm start.
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);
}

// ============================================================================
// 5) Global BA refinement
// ============================================================================

ceres::Solver::Summary global_bundle_adjustment(
    map<long long, vector<Obs2>>& obs_by_frame,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<cv::Mat>& Ps,
    vector<array<double, 6>>& poses,
    vector<long long>& frame_numbers,
    bool use_motion_track,
    double lambda,
    double grid_step,
    const array<double, 3>& motion_track_center_obj,
    double body_scale) {

    ceres::Problem prob;
    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
    static const vector<Obs2> empty_obs;

    // 4.1 Add all reprojection residuals.
    for (size_t i = 0; i < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        auto obs_it = obs_by_frame.find(fr);
        const auto& obs_frame = (obs_it != obs_by_frame.end()) ? obs_it->second : empty_obs;

        for (const auto& ob : obs_frame) {
            if (!model_obj.count(ob.part)) continue;

            const auto& X_obj = model_obj.at(ob.part);

            ceres::CostFunction* cost = PoseReprojCost::Create(
                Ps[ob.cam], X_obj, ob.u, ob.v);

            prob.AddResidualBlock(cost, loss, poses[i].data());
        }
    }

    // 4.2 Add motion-track constraints.
    if (use_motion_track && frame_numbers.size() >= 5) {
        double weight = sqrt(lambda);

        for (size_t i = 2; i < frame_numbers.size() - 2; i++) {
            ceres::CostFunction* cost =
                MotionTrackGridCost::Create(weight, grid_step, motion_track_center_obj, body_scale);

            prob.AddResidualBlock(cost, nullptr,
                poses[i-2].data(),
                poses[i-1].data(),
                poses[i].data(),
                poses[i+1].data(),
                poses[i+2].data());
        }

        cout << "Added " << (frame_numbers.size() - 4)
             << " motion-track constraints centered at object point ("
             << motion_track_center_obj[0] << ", "
             << motion_track_center_obj[1] << ", "
             << motion_track_center_obj[2] << ")" << endl;
    }

    // 4.3 Solve the joint problem.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // Sparse solver for the global problem.
    options.max_num_iterations = 200;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());

    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

    cout << summary.BriefReport() << endl;
    return summary;
}

// ============================================================================
// Helper functions for error evaluation
// ============================================================================

// Apply a pose transform to a 3D point.
void apply_pose_transform(const array<double, 6>& pose,
                          const array<double, 3>& X_obj,
                          array<double, 3>& X_world) {
    double X_rot[3];
    ceres::AngleAxisRotatePoint(pose.data(), X_obj.data(), X_rot);
    X_world[0] = X_rot[0] + pose[3];
    X_world[1] = X_rot[1] + pose[4];
    X_world[2] = X_rot[2] + pose[5];
}

// Project a 3D point with a 3x4 projection matrix.
bool project_point(const cv::Mat& P, const array<double, 3>& X_world,
                   double& u, double& v) {
    double X_homo[4] = {X_world[0], X_world[1], X_world[2], 1.0};

    double u_h = 0, v_h = 0, w = 0;
    for (int j = 0; j < 4; j++) {
        u_h += P.at<double>(0, j) * X_homo[j];
        v_h += P.at<double>(1, j) * X_homo[j];
        w += P.at<double>(2, j) * X_homo[j];
    }

    if (fabs(w) < 1e-10) return false;

    u = u_h / w;
    v = v_h / w;
    return true;
}

// Reprojection error summary for one observation.
struct ReprojError {
    double du, dv;
    double error_px;  // sqrt(du^2 + dv^2)
};

struct TrackErrorStats {
    double rmse;
    double mean_abs;
    double max_abs;
};

ReprojError compute_reproj_error(const cv::Mat& P,
                                  const array<double, 6>& pose,
                                  const array<double, 3>& X_obj,
                                  double u_obs, double v_obs) {
    ReprojError err;

    // Transform the template point into world coordinates.
    array<double, 3> X_world;
    apply_pose_transform(pose, X_obj, X_world);

    // Project the 3D point.
    double u_pred, v_pred;
    if (!project_point(P, X_world, u_pred, v_pred)) {
        err.du = 0;
        err.dv = 0;
        err.error_px = 0;
        return err;
    }

    // Residual in pixels.
    err.du = u_pred - u_obs;
    err.dv = v_pred - v_obs;
    err.error_px = sqrt(err.du * err.du + err.dv * err.dv);

    return err;
}

TrackErrorStats compute_track_error_for_frame(
    const array<double, 6>& pose_m2,
    const array<double, 6>& pose_m1,
    const array<double, 6>& pose_0,
    const array<double, 6>& pose_p1,
    const array<double, 6>& pose_p2,
    double grid_step,
    const array<double, 3>& grid_center_obj) {

    // Predict the center pose from neighboring frames.
    double pose_spline[6];
    cubic_spline_predict_center(pose_m2.data(), pose_m1.data(),
                                 pose_p1.data(), pose_p2.data(),
                                 pose_spline, 6);

    // Evaluate the 3D mismatch on all virtual grid points.
    vector<double> point_errors;

    for (int ix = -1; ix <= 1; ix++) {
        for (int iy = -1; iy <= 1; iy++) {
            for (int iz = -1; iz <= 1; iz++) {
                array<double, 3> grid_pt = {
                    grid_center_obj[0] + ix * grid_step,
                    grid_center_obj[1] + iy * grid_step,
                    grid_center_obj[2] + iz * grid_step
                };

                // Transform with the optimized center-frame pose.
                array<double, 3> X_actual;
                apply_pose_transform(pose_0, grid_pt, X_actual);

                // Transform with the predicted pose.
                array<double, 6> pose_pred;
                for (int k = 0; k < 6; k++) pose_pred[k] = pose_spline[k];
                array<double, 3> X_pred;
                apply_pose_transform(pose_pred, grid_pt, X_pred);

                // Euclidean distance between the two transformed points.
                double dx = X_actual[0] - X_pred[0];
                double dy = X_actual[1] - X_pred[1];
                double dz = X_actual[2] - X_pred[2];
                double dist = sqrt(dx*dx + dy*dy + dz*dz);

                point_errors.push_back(dist);
            }
        }
    }

    TrackErrorStats stats{0.0, 0.0, 0.0};
    if (point_errors.empty()) {
        return stats;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    double max_err = 0.0;
    for (double e : point_errors) {
        sum += e;
        sum_sq += e * e;
        if (e > max_err) max_err = e;
    }

    stats.rmse = sqrt(sum_sq / point_errors.size());
    stats.mean_abs = sum / point_errors.size();
    stats.max_abs = max_err;
    return stats;
}

// ============================================================================
// 6) Error reporting
// ============================================================================

struct ErrorStats {
    double rmse;
    double mean_abs;
    double max_abs;
    size_t count;
};

ErrorStats compute_error_stats(const vector<double>& errors) {
    ErrorStats s{0.0, 0.0, 0.0, errors.size()};
    if (errors.empty()) return s;
    double sum = 0.0, sum_sq = 0.0, max_err = 0.0;
    for (double e : errors) {
        sum    += e;
        sum_sq += e * e;
        if (e > max_err) max_err = e;
    }
    s.rmse     = sqrt(sum_sq / errors.size());
    s.mean_abs = sum / errors.size();
    s.max_abs  = max_err;
    return s;
}

// (1) Reprojection error statistics per frame.
void save_reproj_frame_errors(
    const string& path,
    const map<long long, vector<Obs2>>& obs_by_frame,
    const vector<long long>& frame_numbers,
    const vector<array<double, 6>>& poses,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<cv::Mat>& Ps,
    double fps) {

    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,num_obs,rmse_px,mean_abs_px,max_abs_px\n";

    for (size_t i = 0; i < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        double t = fr / fps;

        auto it = obs_by_frame.find(fr);
        if (it == obs_by_frame.end()) continue;

        const auto& obs_frame = it->second;
        const auto& pose = poses[i];

        vector<double> errors;

        for (const auto& ob : obs_frame) {
            // Skip observations for parts that are not in the template.
            auto part_it = model_obj.find(ob.part);
            if (part_it == model_obj.end()) continue;

            ReprojError err = compute_reproj_error(
                Ps[ob.cam], pose, part_it->second, ob.u, ob.v);

            errors.push_back(err.error_px);
        }

        if (errors.empty()) continue;

        auto stats = compute_error_stats(errors);
        ofs << fr << "," << t << "," << errors.size() << ","
            << stats.rmse << "," << stats.mean_abs << "," << stats.max_abs << "\n";
    }

    cout << "Saved reprojection errors (per frame) to " << path << endl;
}

// (2) Reprojection error statistics per frame and camera.
void save_reproj_frame_cam_errors(
    const string& path,
    const map<long long, vector<Obs2>>& obs_by_frame,
    const vector<long long>& frame_numbers,
    const vector<array<double, 6>>& poses,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<cv::Mat>& Ps,
    double fps) {

    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,cam,num_obs,rmse_px,mean_abs_px,max_abs_px\n";

    for (size_t i = 0; i < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        double t = fr / fps;

        auto it = obs_by_frame.find(fr);
        if (it == obs_by_frame.end()) continue;

        const auto& obs_frame = it->second;
        const auto& pose = poses[i];

        // Group errors by camera.
        map<int, vector<double>> cam_errors;

        for (const auto& ob : obs_frame) {
            auto part_it = model_obj.find(ob.part);
            if (part_it == model_obj.end()) continue;

            ReprojError err = compute_reproj_error(
                Ps[ob.cam], pose, part_it->second, ob.u, ob.v);

            cam_errors[ob.cam].push_back(err.error_px);
        }

        // Write one row per camera.
        for (const auto& kv : cam_errors) {
            int cam = kv.first;
            const auto& errors = kv.second;

            if (errors.empty()) continue;

            auto stats = compute_error_stats(errors);
            ofs << fr << "," << t << "," << cam << "," << errors.size() << ","
                << stats.rmse << "," << stats.mean_abs << "," << stats.max_abs << "\n";
        }
    }

    cout << "Saved reprojection errors (per frame-cam) to " << path << endl;
}

// (3) Reprojection error statistics per frame and part.
void save_reproj_frame_part_errors(
    const string& path,
    const map<long long, vector<Obs2>>& obs_by_frame,
    const vector<long long>& frame_numbers,
    const vector<array<double, 6>>& poses,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<cv::Mat>& Ps,
    double fps) {

    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,part,num_obs,rmse_px,mean_abs_px,max_abs_px\n";

    for (size_t i = 0; i < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        double t = fr / fps;

        auto it = obs_by_frame.find(fr);
        if (it == obs_by_frame.end()) continue;

        const auto& obs_frame = it->second;
        const auto& pose = poses[i];

        // Group errors by body part.
        map<string, vector<double>> part_errors;

        for (const auto& ob : obs_frame) {
            auto part_it = model_obj.find(ob.part);
            if (part_it == model_obj.end()) continue;

            ReprojError err = compute_reproj_error(
                Ps[ob.cam], pose, part_it->second, ob.u, ob.v);

            part_errors[ob.part].push_back(err.error_px);
        }

        // Write one row per body part.
        for (const auto& kv : part_errors) {
            const string& part = kv.first;
            const auto& errors = kv.second;

            if (errors.empty()) continue;

            auto stats = compute_error_stats(errors);
            ofs << fr << "," << t << "," << part << "," << errors.size() << ","
                << stats.rmse << "," << stats.mean_abs << "," << stats.max_abs << "\n";
        }
    }

    cout << "Saved reprojection errors (per frame-part) to " << path << endl;
}

// (4) Motion-track error statistics per frame.
void save_track_frame_errors(
    const string& path,
    const vector<long long>& frame_numbers,
    const vector<array<double, 6>>& poses,
    const array<double, 3>& grid_center_obj,
    double grid_step,
    double fps) {

    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,track_rmse_mm,track_mean_abs_mm,track_max_abs_mm\n";

    // Only middle frames have a full five-frame neighborhood.
    for (size_t i = 2; i + 2 < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        double t = fr / fps;

        TrackErrorStats stats = compute_track_error_for_frame(
            poses[i-2], poses[i-1], poses[i], poses[i+1], poses[i+2],
            grid_step, grid_center_obj);

        ofs << fr << "," << t << ","
            << stats.rmse << ","
            << stats.mean_abs << ","
            << stats.max_abs << "\n";
    }

    cout << "Saved motion-track errors (per frame) to " << path << endl;
}

// ============================================================================
// 7) Output writers
// ============================================================================

void save_poses(const string& path,
                const vector<long long>& frame_numbers,
                const vector<array<double, 6>>& poses,
                double fps,
                bool use_motion_track,
                const unordered_set<long long>& gap_frame_set) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,rx,ry,rz,tx,ty,tz,motion_track,restored\n";

    for (size_t i = 0; i < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        double t = fr / fps;
        int restored_flag = gap_frame_set.count(fr) ? 2 : 0;
        ofs << fr << "," << t << ","
            << poses[i][0] << "," << poses[i][1] << "," << poses[i][2] << ","
            << poses[i][3] << "," << poses[i][4] << "," << poses[i][5] << ","
            << (use_motion_track ? 1 : 0) << ","
            << restored_flag << "\n";
    }

    cout << "Saved poses to " << path << endl;
}

void save_restored_3d_points(
    const string& path,
    const map<long long, vector<Obs2>>& obs_by_frame,
    const vector<long long>& frame_numbers,
    const vector<array<double, 6>>& poses,
    const unordered_map<string, array<double, 3>>& model_obj,
    const vector<string>& part_order,
    double fps,
    int num_cams,
    const unordered_set<long long>& gap_frame_set) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }

    ofs << "frame,t,part,X,Y,Z,restored,num_obs\n";

    for (size_t i = 0; i < frame_numbers.size(); i++) {
        long long fr = frame_numbers[i];
        double t = frame_numbers[i] / fps;
        const auto& pose = poses[i];

        unordered_map<string, int> part_obs_count;
        auto obs_it = obs_by_frame.find(fr);
        if (obs_it != obs_by_frame.end()) {
            for (const auto& ob : obs_it->second) {
                if (!model_obj.count(ob.part)) {
                    continue;
                }
                part_obs_count[ob.part]++;
            }
        }

        for (const string& part : part_order) {
            auto model_it = model_obj.find(part);
            if (model_it == model_obj.end()) {
                continue;
            }

            const auto& X_obj = model_it->second;

            array<double, 3> X_world;
            apply_pose_transform(pose, X_obj, X_world);

            int num_obs = 0;
            auto count_it = part_obs_count.find(part);
            if (count_it != part_obs_count.end()) {
                num_obs = count_it->second;
            }

            int restored;
            if (gap_frame_set.count(fr)) {
                restored = 2;          // Fully missing frame; pose from spline interpolation.
            } else if (num_obs < num_cams) {
                restored = 1;          // Partial observations; pose supported by reprojection.
            } else {
                restored = 0;          // All cameras observed.
            }

            ofs << fr << "," << t << "," << part << ","
                << X_world[0] << "," << X_world[1] << "," << X_world[2] << ","
                << restored << "," << num_obs << "\n";
        }
    }

    cout << "Saved restored-aware 3D points to " << path << endl;
}

// Save the list of gap (fully-missing) frame numbers.
void save_gap_frames(const string& path,
                     const vector<long long>& gap_frames) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }
    ofs << "frame\n";
    for (long long fr : gap_frames) {
        ofs << fr << "\n";
    }
    cout << "Saved " << gap_frames.size() << " gap frames to " << path << endl;
}

void save_ba_summary(const string& path,
                     const ceres::Solver::Summary& summary,
                     bool use_motion_track) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }
    ofs << "field,value\n";
    ofs << "motion_track," << (use_motion_track ? 1 : 0) << "\n";
    ofs << "initial_cost," << summary.initial_cost << "\n";
    ofs << "final_cost," << summary.final_cost << "\n";
    ofs << "num_iterations," << summary.iterations.size() << "\n";
    ofs << "termination_type," << ceres::TerminationTypeToString(summary.termination_type) << "\n";
    ofs << "num_residual_blocks," << summary.num_residual_blocks << "\n";
    ofs << "num_parameter_blocks," << summary.num_parameter_blocks << "\n";
    cout << "Saved BA summary to " << path << endl;
}

void save_run_config(const string& path,
                     double fps,
                     bool use_warm_start,
                     bool use_motion_track,
                     double lambda,
                     double grid_step,
                     int frame_stride,
                     const string& mt_center_part,
                     double motion_track_body_scale,
                     size_t num_frames_observed,
                     size_t num_frames_gap,
                     size_t num_frames_total) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "ERROR: Cannot write " << path << endl;
        return;
    }
    ofs << "{\n";
    ofs << "  \"fps\": " << fps << ",\n";
    ofs << "  \"use_warm_start\": " << (use_warm_start ? "true" : "false") << ",\n";
    ofs << "  \"use_motion_track\": " << (use_motion_track ? "true" : "false") << ",\n";
    ofs << "  \"lambda\": " << lambda << ",\n";
    ofs << "  \"grid_step_mm\": " << grid_step << ",\n";
    ofs << "  \"frame_stride\": " << frame_stride << ",\n";
    ofs << "  \"mt_center_part\": \"" << mt_center_part << "\",\n";
    ofs << "  \"motion_track_body_scale_mm\": " << motion_track_body_scale << ",\n";
    ofs << "  \"num_frames_observed\": " << num_frames_observed << ",\n";
    ofs << "  \"num_frames_gap\": " << num_frames_gap << ",\n";
    ofs << "  \"num_frames_total\": " << num_frames_total << "\n";
    ofs << "}\n";
    cout << "Saved run config to " << path << endl;
}

// ============================================================================
// 8) Spline interpolation for gap-frame pose infill
// ============================================================================

// Natural cubic spline interpolation. x_vals must be strictly increasing.
// Returns interpolated values at each query_x; boundary values are clamped.
vector<double> cubic_spline_interpolate(
    const vector<double>& x_vals,
    const vector<double>& y_vals,
    const vector<double>& query_x) {

    int n = static_cast<int>(x_vals.size());
    vector<double> result(query_x.size(), 0.0);
    if (n < 2) {
        double val = y_vals.empty() ? 0.0 : y_vals[0];
        fill(result.begin(), result.end(), val);
        return result;
    }

    vector<double> h(n - 1);
    for (int i = 0; i < n - 1; i++) h[i] = x_vals[i + 1] - x_vals[i];

    vector<double> alpha(n, 0.0);
    for (int i = 1; i < n - 1; i++) {
        alpha[i] = (3.0 / h[i]) * (y_vals[i + 1] - y_vals[i])
                 - (3.0 / h[i - 1]) * (y_vals[i] - y_vals[i - 1]);
    }

    vector<double> l(n, 1.0), mu(n, 0.0), z(n, 0.0);
    for (int i = 1; i < n - 1; i++) {
        l[i]  = 2.0 * (x_vals[i + 1] - x_vals[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i]  = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    vector<double> c(n, 0.0), b(n, 0.0), d(n, 0.0);
    for (int j = n - 2; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y_vals[j + 1] - y_vals[j]) / h[j]
             - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    for (size_t qi = 0; qi < query_x.size(); qi++) {
        double qx = query_x[qi];
        if (qx <= x_vals.front()) { result[qi] = y_vals.front(); continue; }
        if (qx >= x_vals.back())  { result[qi] = y_vals.back();  continue; }

        int lo = 0, hi = n - 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (x_vals[mid] <= qx) lo = mid; else hi = mid - 1;
        }
        int j = lo;
        double dx = qx - x_vals[j];
        result[qi] = y_vals[j] + b[j] * dx + c[j] * dx * dx + d[j] * dx * dx * dx;
    }
    return result;
}

// Interpolate poses for gap frames using the known observed-frame poses.
void interpolate_gap_poses(
    const vector<long long>& known_frames,
    const vector<array<double, 6>>& known_poses,
    const vector<long long>& gap_frames,
    vector<array<double, 6>>& gap_poses) {

    gap_poses.clear();
    if (gap_frames.empty()) return;
    if (known_frames.empty()) {
        gap_poses.assign(gap_frames.size(), {0, 0, 0, 0, 0, 0});
        return;
    }

    vector<double> x_known(known_frames.size());
    for (size_t i = 0; i < known_frames.size(); i++)
        x_known[i] = static_cast<double>(known_frames[i]);

    vector<double> x_query(gap_frames.size());
    for (size_t i = 0; i < gap_frames.size(); i++)
        x_query[i] = static_cast<double>(gap_frames[i]);

    gap_poses.resize(gap_frames.size());
    for (int comp = 0; comp < 6; comp++) {
        vector<double> y_known(known_frames.size());
        for (size_t i = 0; i < known_frames.size(); i++)
            y_known[i] = known_poses[i][comp];

        vector<double> y_interp = cubic_spline_interpolate(x_known, y_known, x_query);
        for (size_t i = 0; i < gap_frames.size(); i++)
            gap_poses[i][comp] = y_interp[i];
    }
}

// ============================================================================
// Main entry point
// ============================================================================

int main(int argc, char** argv) {
    // Default inputs. Override them with command-line arguments if needed.
    string cams_path = "3cam/cams.txt";
    string front_csv = "3cam/obs_de_front.csv";
    string side_csv = "3cam/obs_de_side.csv";
    string top_csv = "3cam/obs_de_top.csv";
    string template_path = "results/mouse_model_obj.csv";
    string output_dir = "results";
    string mt_center_part = "";
    bool mt_center_part_specified = false;

    double fps = 30.0;
    bool use_warm_start = false;
    bool use_motion_track = false;   // Motion-track is off unless explicitly enabled.
    double lambda = 1.0;             // Motion-track weight.
    double grid_step = 10.0;         // Virtual grid spacing in mm.
    int frame_stride = 1;            // 1 = use every frame, 2 = use every other frame, etc.

    // Parse command-line arguments.
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--fps" && i+1 < argc) {
            fps = stod(argv[++i]);
        } else if (arg == "--warm-start") {
            use_warm_start = true;
        } else if (arg == "--motion-track") {
            use_motion_track = true;
        } else if (arg == "--no-motion-track") {
            use_motion_track = false;
        } else if (arg == "--lambda" && i+1 < argc) {
            lambda = stod(argv[++i]);
        } else if ((arg == "--grid-step" || arg == "--grid_step") && i+1 < argc) {
            grid_step = stod(argv[++i]);
        } else if (arg == "--stride" && i+1 < argc) {
            frame_stride = stoi(argv[++i]);
        } else if (arg == "--cams" && i+1 < argc) {
            cams_path = argv[++i];
        } else if (arg == "--front" && i+1 < argc) {
            front_csv = argv[++i];
        } else if (arg == "--side" && i+1 < argc) {
            side_csv = argv[++i];
        } else if (arg == "--top" && i+1 < argc) {
            top_csv = argv[++i];
        } else if (arg == "--template" && i+1 < argc) {
            template_path = argv[++i];
        } else if ((arg == "--mt-center" || arg == "--mt-center-part") && i+1 < argc) {
            mt_center_part = argv[++i];
            mt_center_part_specified = true;
        } else if (arg == "--outdir" && i+1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --cams <path>       Camera file path (default: 3cam/cams.txt)\n";
            cout << "  --front <path>      Front camera CSV (default: 3cam/obs_de_front.csv)\n";
            cout << "  --side <path>       Side camera CSV (default: 3cam/obs_de_side.csv)\n";
            cout << "  --top <path>        Top camera CSV (default: 3cam/obs_de_top.csv)\n";
            cout << "  --template <path>   Rigid template CSV (default: results/mouse_model_obj.csv)\n";
            cout << "  --fps <value>       Frame rate (default: 30.0)\n";
            cout << "  --warm-start        Enable per-frame pose warm start\n";
            cout << "  --motion-track      Enable motion-track constraint\n";
            cout << "  --no-motion-track   Disable motion-track constraint\n";
            cout << "  --lambda <value>    Motion-track weight (default: 1.0)\n";
            cout << "  --grid-step <value> Grid step in mm (default: 10.0)\n";
            cout << "  --grid_step <value> Alias for --grid-step\n";
            cout << "  --mt-center <part>  Use a template body part as the motion-track grid center\n";
            cout << "  --stride <value>    Frame stride, use every Nth frame (default: 1)\n";
            cout << "  --outdir <path>     Output directory for evaluation and debug files (default: results)\n";
            cout << "  --help, -h          Show this help\n";
            return 0;
        }
    }

    const string executable_dir = get_executable_dir(argv[0]);
    cams_path = resolve_input_path(cams_path, executable_dir);
    front_csv = resolve_input_path(front_csv, executable_dir);
    side_csv = resolve_input_path(side_csv, executable_dir);
    top_csv = resolve_input_path(top_csv, executable_dir);
    template_path = resolve_input_path(template_path, executable_dir);
    cout << "\n=== BA Motion Track ===\n" << endl;
    cout << "FPS: " << fps << endl;
    cout << "Warm start: " << (use_warm_start ? "ON" : "OFF") << endl;
    cout << "Motion-track constraint: " << (use_motion_track ? "ON" : "OFF") << endl;
    if (use_motion_track) {
        cout << "Lambda: " << lambda << ", Grid step: " << grid_step << endl;
    }
    cout << "Resolved cameras: " << cams_path << endl;
    cout << "Resolved front CSV: " << front_csv << endl;
    cout << "Resolved side CSV: " << side_csv << endl;
    cout << "Resolved top CSV: " << top_csv << endl;
    cout << "Resolved template CSV: " << template_path << endl;
    cout << "Output directory: " << output_dir << endl;
    cout << endl;

    // 1) Load input data.
    cout << "Step 1: Loading data..." << endl;
    vector<cv::Mat> Ps = load_Ps_txt_first3(cams_path);
    if (Ps.size() != 3) {
        cerr << "ERROR: Need exactly 3 cameras" << endl;
        return -1;
    }

    vector<Obs2> obs_front = load_obs_csv(front_csv, 0);
    vector<Obs2> obs_side = load_obs_csv(side_csv, 1);
    vector<Obs2> obs_top = load_obs_csv(top_csv, 2);

    vector<Obs2> obsAll;
    obsAll.insert(obsAll.end(), obs_front.begin(), obs_front.end());
    obsAll.insert(obsAll.end(), obs_side.begin(), obs_side.end());
    obsAll.insert(obsAll.end(), obs_top.begin(), obs_top.end());

    vector<string> part_order;  // Preserve the first-seen part order from the input CSVs.
    map<long long, vector<Obs2>> obs_by_frame = group_obs_by_frame(obsAll, fps, part_order);

    if (obs_by_frame.empty()) {
        cerr << "ERROR: No observations found" << endl;
        return -1;
    }

    // 2) Load the rigid template and estimate a translation seed.
    cout << "\nStep 2: Loading rigid template..." << endl;
    cout << "Initial translation seed will be fixed from the first valid frame with triangulatable 'head center'" << endl;

    unordered_map<string, array<double, 3>> model_obj = load_template(template_path);
    if (model_obj.empty()) {
        cerr << "ERROR: Failed to load template" << endl;
        return -1;
    }

    const array<double, 3> template_centroid_obj = compute_template_centroid_obj(model_obj);
    const auto head_center_it = model_obj.find("head center");
    if (head_center_it != model_obj.end() &&
        fabs(head_center_it->second[0]) < 1e-6 &&
        fabs(head_center_it->second[1]) < 1e-6 &&
        fabs(head_center_it->second[2]) < 1e-6) {
        cout << "Template check: head center is at object origin (head-centered template)." << endl;
    } else {
        cerr << "WARNING: Template check: head center is not at object origin." << endl;
    }

    cout << "Template centroid in object frame: ("
         << template_centroid_obj[0] << ", "
         << template_centroid_obj[1] << ", "
         << template_centroid_obj[2] << ")" << endl;

    double head_to_tail_base = -1.0;
    double tail_base_to_tail_root = -1.0;
    compute_template_body_scale_stats(model_obj, head_to_tail_base, tail_base_to_tail_root);
    if (head_to_tail_base > 0.0) {
        cout << "Template check: head center -> tail base distance = "
             << head_to_tail_base << endl;
    } else {
        cerr << "WARNING: Template check: head center -> tail base distance is unavailable." << endl;
    }
    if (tail_base_to_tail_root > 0.0) {
        cout << "Template check: tail base -> tail root distance = "
             << tail_base_to_tail_root << endl;
    } else {
        cerr << "WARNING: Template check: tail base -> tail root distance is unavailable." << endl;
    }

    double motion_track_body_scale = head_to_tail_base;
    if (motion_track_body_scale <= 0.0) {
        cerr << "ERROR: Cannot derive a valid body scale from template (head center -> tail base)" << endl;
        return -1;
    }
    cout << "Motion-track body scale for residual normalization: "
         << motion_track_body_scale << endl;

    map<long long, array<double, 3>> seed_map;
    if (!estimate_initial_translation_from_sequence(obs_by_frame, model_obj, Ps, seed_map)) {
        return -1;
    }

    array<double, 3> motion_track_center_obj = template_centroid_obj;
    if (mt_center_part_specified) {
        auto mt_center_it = model_obj.find(mt_center_part);
        if (mt_center_it == model_obj.end()) {
            cerr << "ERROR: Motion-track center body part '" << mt_center_part
                 << "' does not exist in the template" << endl;
            return -1;
        }
        motion_track_center_obj = mt_center_it->second;
        cout << "Motion-track grid center set from body part '" << mt_center_part << "': ("
             << motion_track_center_obj[0] << ", "
             << motion_track_center_obj[1] << ", "
             << motion_track_center_obj[2] << ")" << endl;
    } else {
        cout << "Motion-track grid center set from template centroid: ("
             << motion_track_center_obj[0] << ", "
             << motion_track_center_obj[1] << ", "
             << motion_track_center_obj[2] << ")" << endl;
    }

    cout << "Motion-track grid step: " << grid_step << endl;
    cout << "Motion-track grid center: ("
         << motion_track_center_obj[0] << ", "
         << motion_track_center_obj[1] << ", "
         << motion_track_center_obj[2] << ")" << endl;

    if (grid_step <= 0.0) {
        cerr << "ERROR: Motion-track grid_step must be positive" << endl;
        return -1;
    }
    if (head_to_tail_base > 0.0) {
        if (grid_step < 0.05 * head_to_tail_base) {
            cerr << "WARNING: Motion-track grid_step is very small relative to body length; rotation sensitivity may be weak." << endl;
        }
        if (grid_step > 1.0 * head_to_tail_base) {
            cerr << "WARNING: Motion-track grid_step is larger than the main body length; the MT grid may be too large." << endl;
        }
    }

    const string results_dir = output_dir;
    const string diag_dir = join_path(results_dir, "diagnostics");
    if (!ensure_directory(results_dir)) {
        return -1;
    }
    if (!ensure_directory(diag_dir)) {
        return -1;
    }

    vector<string> generated_files;

    // 3) Prepare the initial poses for global BA.
    cout << "\nStep 3: Preparing initial poses..." << endl;

    const vector<Obs2> empty_obs;   // Fallback for frames with no observations.

    // 3.1 Infer frame step and find gap frames.
    long long frame_step = infer_frame_step(obs_by_frame);
    vector<long long> gap_frames = find_gap_frames(obs_by_frame, frame_step);
    if (!gap_frames.empty()) {
        cout << "Detected " << gap_frames.size() << " gap frames (step="
             << frame_step << ")" << endl;
    }
    unordered_set<long long> gap_frame_set(gap_frames.begin(), gap_frames.end());

    // 3.2 Build observed-frame list (with stride).
    vector<long long> frame_numbers;
    vector<array<double, 6>> poses;

    int frame_idx = 0;
    for (const auto& kv : obs_by_frame) {
        if (frame_idx % frame_stride == 0) {
            frame_numbers.push_back(kv.first);
            auto t_seed = lookup_translation_seed(kv.first, seed_map);
            poses.push_back({0.0, 0.0, 0.0,
                             t_seed[0], t_seed[1], t_seed[2]});
        }
        frame_idx++;
    }

    if (frame_stride > 1) {
        cout << "Using frame stride " << frame_stride
             << " (processing " << frame_numbers.size()
             << " out of " << obs_by_frame.size() << " frames)" << endl;
    }

    {
        const string suffix = use_motion_track ? "_with_motion" : "_no_motion";
        const string init_pose_path = join_path(diag_dir, "init_pose" + suffix + ".csv");
        save_poses(init_pose_path, frame_numbers, poses, fps, use_motion_track, gap_frame_set);
        generated_files.push_back(init_pose_path);
    }

    if (use_warm_start) {
        cout << "\nStep 3.5: Running warm start..." << endl;
        for (size_t i = 0; i < frame_numbers.size(); i++) {
            long long fr = frame_numbers[i];

            // Use the previous optimized frame as the next initialization seed.
            if (i > 0) {
                poses[i] = poses[i-1];
            }

            auto ws_it = obs_by_frame.find(fr);
            const auto& ws_obs = (ws_it != obs_by_frame.end()) ? ws_it->second : empty_obs;
            initialize_pose_per_frame(i, ws_obs, model_obj, Ps, poses[i].data());

            if ((i+1) % 10 == 0 || i+1 == frame_numbers.size()) {
                cout << "Warm-start initialized " << (i+1) << "/" << frame_numbers.size() << " frames" << endl;
            }
        }

        cout << "Completed warm start for all frames" << endl;

        {
            const string suffix = use_motion_track ? "_with_motion" : "_no_motion";
            const string warm_start_pose_path = join_path(diag_dir, "warm_start_pose" + suffix + ".csv");
            save_poses(warm_start_pose_path, frame_numbers, poses, fps, use_motion_track, gap_frame_set);
            generated_files.push_back(warm_start_pose_path);
        }
    } else {
        cout << "Warm start disabled; global BA will start from the direct pose seed" << endl;
    }

    // 3.4 Interpolate gap-frame poses from observed-frame poses (path B).
    {
        vector<array<double, 6>> gap_poses;
        interpolate_gap_poses(frame_numbers, poses, gap_frames, gap_poses);

        for (size_t i = 0; i < gap_frames.size(); i++) {
            frame_numbers.push_back(gap_frames[i]);
            poses.push_back(gap_poses[i]);
        }

        // Sort merged list by frame number.
        vector<size_t> idx(frame_numbers.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(),
             [&](size_t a, size_t b){ return frame_numbers[a] < frame_numbers[b]; });
        vector<long long> fn_sorted(frame_numbers.size());
        vector<array<double, 6>> poses_sorted(poses.size());
        for (size_t i = 0; i < idx.size(); i++) {
            fn_sorted[i]    = frame_numbers[idx[i]];
            poses_sorted[i] = poses[idx[i]];
        }
        frame_numbers = std::move(fn_sorted);
        poses         = std::move(poses_sorted);

        if (!gap_frames.empty()) {
            cout << "Merged " << gap_frames.size()
                 << " spline-interpolated gap frames; total frames: "
                 << frame_numbers.size() << endl;
        }
    }

    auto poses_before_ba = poses; // Sanity check that global BA actually updates the trajectory.

    // Output pre-BA reprojection diagnostics for before/after comparison.
    {
        const string suffix = use_motion_track ? "_with_motion" : "_no_motion";
        const string pre_reproj_frame_path      = join_path(diag_dir, "pre_ba_reproj_frame" + suffix + ".csv");
        const string pre_reproj_frame_cam_path  = join_path(diag_dir, "pre_ba_reproj_frame_cam" + suffix + ".csv");
        const string pre_reproj_frame_part_path = join_path(diag_dir, "pre_ba_reproj_frame_part" + suffix + ".csv");

        save_reproj_frame_errors(pre_reproj_frame_path, obs_by_frame, frame_numbers,
                                 poses, model_obj, Ps, fps);
        save_reproj_frame_cam_errors(pre_reproj_frame_cam_path, obs_by_frame, frame_numbers,
                                     poses, model_obj, Ps, fps);
        save_reproj_frame_part_errors(pre_reproj_frame_part_path, obs_by_frame, frame_numbers,
                                      poses, model_obj, Ps, fps);

        generated_files.push_back(pre_reproj_frame_path);
        generated_files.push_back(pre_reproj_frame_cam_path);
        generated_files.push_back(pre_reproj_frame_part_path);
    }

    // 4) Run the global BA refinement.
    cout << "\nStep 4: Global bundle adjustment..." << endl;

    ceres::Solver::Summary ba_summary =
        global_bundle_adjustment(obs_by_frame, model_obj, Ps, poses, frame_numbers,
                                 use_motion_track, lambda, grid_step, motion_track_center_obj,
                                 motion_track_body_scale);

    const string ba_summary_path = join_path(
        diag_dir,
        use_motion_track ? "ba_summary_with_motion.csv" : "ba_summary_no_motion.csv");
    save_ba_summary(ba_summary_path, ba_summary, use_motion_track);
    generated_files.push_back(ba_summary_path);


    // Report the largest pose change introduced by global BA.
    double max_abs = 0.0;
    for (size_t i = 0; i < poses.size(); i++) {
        for (int k = 0; k < 6; k++) {
            max_abs = std::max(max_abs, std::abs(poses[i][k] - poses_before_ba[i][k]));
        }
    }
    std::cout << "[check] max |pose_after - pose_before| = " << max_abs << std::endl;

    // 5) Save final evaluation outputs.
    cout << "\nStep 5: Saving evaluation outputs..." << endl;

    {
        const string suffix = use_motion_track ? "_with_motion" : "_no_motion";
        const string config_path = join_path(results_dir, "run_config" + suffix + ".json");
        save_run_config(
            config_path,
            fps,
            use_warm_start,
            use_motion_track,
            lambda,
            grid_step,
            frame_stride,
            mt_center_part,
            motion_track_body_scale,
            obs_by_frame.size(),
            gap_frames.size(),
            frame_numbers.size());
        generated_files.push_back(config_path);
    }

    const string pose_output_path = join_path(
        results_dir,
        use_motion_track ? "pose_with_motion.csv" : "pose_no_motion.csv");
    save_poses(pose_output_path, frame_numbers, poses, fps, use_motion_track, gap_frame_set);
    generated_files.push_back(pose_output_path);

    const string restored_output_path = join_path(
        results_dir,
        use_motion_track ? "xyz_restored_with_motion.csv" : "xyz_restored_no_motion.csv");
    save_restored_3d_points(restored_output_path, obs_by_frame, frame_numbers, poses,
                            model_obj, part_order, fps, static_cast<int>(Ps.size()),
                            gap_frame_set);
    generated_files.push_back(restored_output_path);

    const string gap_frames_path = join_path(
        results_dir,
        use_motion_track ? "gap_frames_with_motion.csv" : "gap_frames_no_motion.csv");
    save_gap_frames(gap_frames_path, gap_frames);
    generated_files.push_back(gap_frames_path);

    {
        cout << "\nStep 6: Saving diagnostics..." << endl;
        const string suffix = use_motion_track ? "_with_motion" : "_no_motion";

        const string reproj_frame_path      = join_path(diag_dir, "reproj_frame"      + suffix + ".csv");
        const string reproj_frame_cam_path  = join_path(diag_dir, "reproj_frame_cam"  + suffix + ".csv");
        const string reproj_frame_part_path = join_path(diag_dir, "reproj_frame_part" + suffix + ".csv");

        save_reproj_frame_errors(reproj_frame_path, obs_by_frame, frame_numbers,
                                 poses, model_obj, Ps, fps);
        save_reproj_frame_cam_errors(reproj_frame_cam_path, obs_by_frame, frame_numbers,
                                     poses, model_obj, Ps, fps);
        save_reproj_frame_part_errors(reproj_frame_part_path, obs_by_frame, frame_numbers,
                                      poses, model_obj, Ps, fps);

        generated_files.push_back(reproj_frame_path);
        generated_files.push_back(reproj_frame_cam_path);
        generated_files.push_back(reproj_frame_part_path);

        if (use_motion_track) {
            const string track_frame_path = join_path(diag_dir, "track_frame" + suffix + ".csv");
            save_track_frame_errors(track_frame_path, frame_numbers, poses,
                                    motion_track_center_obj, grid_step, fps);
            generated_files.push_back(track_frame_path);
        }
    }

    cout << "\n=== All done! ===" << endl;
    cout << "\nGenerated files:" << endl;
    for (const string& path : generated_files) {
        cout << "  - " << path << endl;
    }
    
    return 0;
}
