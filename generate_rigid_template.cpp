#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <array>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <sys/stat.h>
#include <limits.h>
#include <unistd.h>
#include <set>

#include <opencv2/core.hpp>

using namespace std;

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR ""
#endif

struct Obs2 {
    int cam;
    string part;
    double u, v;
    double t;
    long long frame;
};

struct PartSampleStats {
    size_t raw_candidates = 0;
    size_t rejected_invalid = 0;
    size_t rejected_reproj = 0;
    size_t rejected_angle = 0;
    size_t accepted = 0;
};

struct SampleCollectionResult {
    unordered_map<string, vector<array<double, 3>>> part_samples;
    unordered_map<string, PartSampleStats> stats;
};

string remove_bom(const string& str);
double compute_median(vector<double> values);

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

string replace_extension(const string& path, const string& new_ext) {
    size_t slash_pos = path.find_last_of("/\\");
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == string::npos || (slash_pos != string::npos && dot_pos < slash_pos)) {
        return path + new_ext;
    }
    return path.substr(0, dot_pos) + new_ext;
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

        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

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
    }

    return Ps;
}

string remove_bom(const string& str) {
    if (str.size() >= 3 &&
        (unsigned char)str[0] == 0xEF &&
        (unsigned char)str[1] == 0xBB &&
        (unsigned char)str[2] == 0xBF) {
        return str.substr(3);
    }
    return str;
}

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
    int frame_col = -1;
    int t_col = -1;
    int part_col = -1;
    int x_col = -1;
    int y_col = -1;

    for (size_t i = 0; i < header_fields.size(); i++) {
        string name = normalize_header_name(header_fields[i]);
        if (name == "frame") {
            frame_col = static_cast<int>(i);
        } else if (name == "t" || name == "time" || name == "timestamp") {
            t_col = static_cast<int>(i);
        } else if (name == "part" || name == "bodypart") {
            part_col = static_cast<int>(i);
        } else if (name == "x" || name == "u") {
            x_col = static_cast<int>(i);
        } else if (name == "y" || name == "v") {
            y_col = static_cast<int>(i);
        }
    }

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
            continue;
        }
    }

    cout << "Loaded " << obs.size() << " observations from cam " << cam_id << endl;
    return obs;
}

map<long long, vector<Obs2>> group_obs_by_frame(const vector<Obs2>& obsAll,
                                                double fps,
                                                vector<string>& part_order) {
    map<long long, vector<Obs2>> obs_by_frame;
    unordered_map<string, bool> part_seen;

    for (const auto& ob : obsAll) {
        long long frame = (ob.frame >= 0) ? ob.frame : (long long)round(ob.t * fps);
        obs_by_frame[frame].push_back(ob);

        if (!part_seen[ob.part]) {
            part_seen[ob.part] = true;
            part_order.push_back(ob.part);
        }
    }

    cout << "Grouped into " << obs_by_frame.size() << " frames" << endl;
    cout << "Part order (" << part_order.size() << " parts): ";
    for (size_t i = 0; i < part_order.size(); i++) {
        cout << part_order[i];
        if (i + 1 < part_order.size()) cout << ", ";
    }
    cout << endl;

    return obs_by_frame;
}

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

        for (int j = 0; j < 4; j++) {
            A.at<double>(2 * i, j) = u * P.at<double>(2, j) - P.at<double>(0, j);
            A.at<double>(2 * i + 1, j) = v * P.at<double>(2, j) - P.at<double>(1, j);
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

cv::Point2d project_point(const cv::Point3d& X, const cv::Mat& P) {
    cv::Mat X_homo(4, 1, CV_64F);
    X_homo.at<double>(0, 0) = X.x;
    X_homo.at<double>(1, 0) = X.y;
    X_homo.at<double>(2, 0) = X.z;
    X_homo.at<double>(3, 0) = 1.0;

    cv::Mat x = P * X_homo;
    const double w = x.at<double>(2, 0);
    if (!std::isfinite(w) || fabs(w) < 1e-12) {
        return cv::Point2d(numeric_limits<double>::quiet_NaN(),
                           numeric_limits<double>::quiet_NaN());
    }

    return cv::Point2d(x.at<double>(0, 0) / w, x.at<double>(1, 0) / w);
}

double reproj_error_px(const cv::Point3d& X,
                       const vector<pair<int, cv::Point2d>>& views,
                       const vector<cv::Mat>& Ps) {
    if (views.empty()) {
        return numeric_limits<double>::infinity();
    }

    double error_sum = 0.0;
    for (const auto& view : views) {
        const cv::Point2d projected = project_point(X, Ps[view.first]);
        if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
            return numeric_limits<double>::infinity();
        }

        const double du = projected.x - view.second.x;
        const double dv = projected.y - view.second.y;
        error_sum += sqrt(du * du + dv * dv);
    }

    return error_sum / static_cast<double>(views.size());
}

cv::Point3d camera_center_from_P(const cv::Mat& P) {
    cv::Mat w, u, vt;
    cv::SVD::compute(P, w, u, vt);

    cv::Mat C_homo = vt.row(vt.rows - 1).t();
    const double w_h = C_homo.at<double>(3, 0);
    if (!std::isfinite(w_h) || fabs(w_h) < 1e-12) {
        return cv::Point3d(numeric_limits<double>::quiet_NaN(),
                           numeric_limits<double>::quiet_NaN(),
                           numeric_limits<double>::quiet_NaN());
    }

    return cv::Point3d(C_homo.at<double>(0, 0) / w_h,
                       C_homo.at<double>(1, 0) / w_h,
                       C_homo.at<double>(2, 0) / w_h);
}

double angle_between_rays_deg(const cv::Point3d& X,
                              const cv::Point3d& C1,
                              const cv::Point3d& C2) {
    const cv::Point3d r1 = X - C1;
    const cv::Point3d r2 = X - C2;

    const double n1 = sqrt(r1.dot(r1));
    const double n2 = sqrt(r2.dot(r2));
    if (!std::isfinite(n1) || !std::isfinite(n2) || n1 < 1e-12 || n2 < 1e-12) {
        return 0.0;
    }

    double cos_angle = r1.dot(r2) / (n1 * n2);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    return acos(cos_angle) * 180.0 / M_PI;
}

double max_triangulation_angle_deg(const cv::Point3d& X,
                                   const vector<pair<int, cv::Point2d>>& views,
                                   const vector<cv::Point3d>& camera_centers) {
    if (views.size() < 2) {
        return 0.0;
    }

    double max_angle = 0.0;
    for (size_t i = 0; i < views.size(); i++) {
        for (size_t j = i + 1; j < views.size(); j++) {
            const cv::Point3d& C1 = camera_centers[views[i].first];
            const cv::Point3d& C2 = camera_centers[views[j].first];
            const double angle = angle_between_rays_deg(X, C1, C2);
            max_angle = std::max(max_angle, angle);
        }
    }

    return max_angle;
}

array<double, 3> compute_coordinate_median_point(const vector<array<double, 3>>& samples) {
    vector<double> xs;
    vector<double> ys;
    vector<double> zs;
    xs.reserve(samples.size());
    ys.reserve(samples.size());
    zs.reserve(samples.size());

    for (const auto& sample : samples) {
        xs.push_back(sample[0]);
        ys.push_back(sample[1]);
        zs.push_back(sample[2]);
    }

    return {compute_median(xs), compute_median(ys), compute_median(zs)};
}

double euclidean_distance_3d(const array<double, 3>& a, const array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

array<double, 3> vec_sub(const array<double, 3>& a, const array<double, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

array<double, 3> vec_add(const array<double, 3>& a, const array<double, 3>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

array<double, 3> vec_scale(const array<double, 3>& v, double s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

double vec_dot(const array<double, 3>& a, const array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

array<double, 3> vec_cross(const array<double, 3>& a, const array<double, 3>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

double vec_norm(const array<double, 3>& v) {
    return sqrt(vec_dot(v, v));
}

bool normalize_vec(array<double, 3>& v, double eps = 1e-8) {
    const double n = vec_norm(v);
    if (!std::isfinite(n) || n < eps) {
        return false;
    }
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
    return true;
}

array<double, 3> point3d_to_array(const cv::Point3d& p) {
    return {p.x, p.y, p.z};
}

bool get_left_right_reference(
    const unordered_map<string, array<double, 3>>& frame_points_world,
    array<double, 3>& left_right_vec,
    string& source_name) {

    auto left_ear = frame_points_world.find("left ear");
    auto right_ear = frame_points_world.find("right ear");
    if (left_ear != frame_points_world.end() && right_ear != frame_points_world.end()) {
        left_right_vec = vec_sub(left_ear->second, right_ear->second);
        source_name = "ears";
        return true;
    }

    auto left_front_paw = frame_points_world.find("left front paw");
    auto right_front_paw = frame_points_world.find("right front paw");
    if (left_front_paw != frame_points_world.end() && right_front_paw != frame_points_world.end()) {
        left_right_vec = vec_sub(left_front_paw->second, right_front_paw->second);
        source_name = "front paws";
        return true;
    }

    auto left_hind_paw = frame_points_world.find("left hind paw");
    auto right_hind_paw = frame_points_world.find("right hind paw");
    if (left_hind_paw != frame_points_world.end() && right_hind_paw != frame_points_world.end()) {
        left_right_vec = vec_sub(left_hind_paw->second, right_hind_paw->second);
        source_name = "hind paws";
        return true;
    }

    return false;
}

bool build_frame_alignment_axes(
    const unordered_map<string, array<double, 3>>& frame_points_world,
    const array<double, 3>& left_right_vec,
    array<double, 3>& origin,
    array<double, 3>& x_axis,
    array<double, 3>& y_axis,
    array<double, 3>& z_axis) {

    auto head_center_it = frame_points_world.find("head center");
    auto tail_base_it = frame_points_world.find("tail base");
    if (head_center_it == frame_points_world.end() || tail_base_it == frame_points_world.end()) {
        return false;
    }

    origin = head_center_it->second;
    y_axis = vec_sub(tail_base_it->second, head_center_it->second);
    if (!normalize_vec(y_axis)) {
        return false;
    }

    const array<double, 3> x_parallel = vec_scale(y_axis, vec_dot(left_right_vec, y_axis));
    x_axis = vec_sub(left_right_vec, x_parallel);
    if (!normalize_vec(x_axis)) {
        return false;
    }

    z_axis = vec_cross(x_axis, y_axis);
    if (!normalize_vec(z_axis)) {
        return false;
    }

    // re-orthogonalize x
    x_axis = vec_cross(y_axis, z_axis);
    if (!normalize_vec(x_axis)) return false;

    return true;
}

array<double, 3> transform_world_to_local(
    const array<double, 3>& X_world,
    const array<double, 3>& origin,
    const array<double, 3>& x_axis,
    const array<double, 3>& y_axis,
    const array<double, 3>& z_axis) {

    const array<double, 3> d = vec_sub(X_world, origin);
    return {
        vec_dot(d, x_axis),
        vec_dot(d, y_axis),
        vec_dot(d, z_axis)
    };
}

void save_template(const string& path,
                   const unordered_map<string, array<double, 3>>& model,
                   const vector<string>& part_order) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "WARNING: Cannot write " << path << endl;
        return;
    }

    ofs << "part,X,Y,Z\n";
    for (const string& part : part_order) {
        auto it = model.find(part);
        if (it == model.end()) {
            continue;
        }

        ofs << part << ","
            << it->second[0] << ","
            << it->second[1] << ","
            << it->second[2] << "\n";
    }

    cout << "Saved template to " << path << endl;
}

void save_template_point_cloud_ply(const string& path,
                                   const unordered_map<string, array<double, 3>>& model,
                                   const vector<string>& part_order) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "WARNING: Cannot write point cloud " << path << endl;
        return;
    }

    size_t vertex_count = 0;
    for (const string& part : part_order) {
        if (model.find(part) != model.end()) {
            vertex_count++;
        }
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "comment Generated from final rigid template output\n";
    ofs << "element vertex " << vertex_count << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for (const string& part : part_order) {
        auto it = model.find(part);
        if (it == model.end()) {
            continue;
        }

        ofs << it->second[0] << " "
            << it->second[1] << " "
            << it->second[2] << "\n";
    }

    cout << "Saved template point cloud to " << path << endl;
}

vector<pair<string, string>> build_default_mouse_skeleton_edges(
    const unordered_map<string, array<double, 3>>& model) {

    const vector<pair<string, string>> candidate_edges = {
        {"nose", "head center"},
        {"left ear", "head center"},
        {"right ear", "head center"},
        {"nose", "left ear"},
        {"nose", "right ear"},
        {"head center", "tail base"},
        {"tail root", "tail base"},
        {"head center", "left front paw"},
        {"head center", "right front paw"},
        {"tail base", "left hind paw"},
        {"tail base", "right hind paw"}
    };

    vector<pair<string, string>> edges;
    for (const auto& edge : candidate_edges) {
        if (model.find(edge.first) != model.end() && model.find(edge.second) != model.end()) {
            edges.push_back(edge);
        }
    }
    return edges;
}

vector<array<double, 3>> interpolate_edge_points(const array<double, 3>& a,
                                                 const array<double, 3>& b,
                                                 int num_points) {
    num_points = std::max(num_points, 2);

    vector<array<double, 3>> points;
    points.reserve(static_cast<size_t>(num_points));
    for (int i = 0; i < num_points; i++) {
        const double alpha = static_cast<double>(i) / static_cast<double>(num_points - 1);
        points.push_back({
            (1.0 - alpha) * a[0] + alpha * b[0],
            (1.0 - alpha) * a[1] + alpha * b[1],
            (1.0 - alpha) * a[2] + alpha * b[2]
        });
    }
    return points;
}

void save_template_skeleton_html(const string& path,
                                 const unordered_map<string, array<double, 3>>& model,
                                 const vector<string>& part_order) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "WARNING: Cannot write skeleton html " << path << endl;
        return;
    }

    vector<string> parts;
    vector<double> xs;
    vector<double> ys;
    vector<double> zs;
    vector<string> node_colors;
    for (const string& part : part_order) {
        auto it = model.find(part);
        if (it == model.end()) {
            continue;
        }
        parts.push_back(part);
        xs.push_back(it->second[0]);
        ys.push_back(it->second[1]);
        zs.push_back(it->second[2]);

        if (part == "nose" || part == "tail root") {
            node_colors.push_back("white");
        } else if (part == "head center" || part == "tail base") {
            node_colors.push_back("gold");
        } else if (part == "left ear" || part == "left front paw" || part == "left hind paw") {
            node_colors.push_back("red");
        } else if (part == "right ear" || part == "right front paw" || part == "right hind paw") {
            node_colors.push_back("green");
        } else {
            node_colors.push_back("gold");
        }
    }

    const vector<pair<string, string>> edges = build_default_mouse_skeleton_edges(model);
    vector<array<double, 3>> dotted_edge_points;
    for (const auto& edge : edges) {
        const auto& a = model.at(edge.first);
        const auto& b = model.at(edge.second);
        const vector<array<double, 3>> edge_points = interpolate_edge_points(a, b, 40);
        dotted_edge_points.insert(dotted_edge_points.end(), edge_points.begin(), edge_points.end());
    }

    ofs << "<!DOCTYPE html>\n";
    ofs << "<html lang=\"en\">\n";
    ofs << "<head>\n";
    ofs << "  <meta charset=\"utf-8\">\n";
    ofs << "  <title>Mouse Template Skeleton</title>\n";
    ofs << "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n";
    ofs << "</head>\n";
    ofs << "<body>\n";
    ofs << "  <div id=\"plot\" style=\"width: 100%; height: 95vh;\"></div>\n";
    ofs << "  <script>\n";

    ofs << "    const nodeTrace = {\n";
    ofs << "      type: 'scatter3d',\n";
    ofs << "      mode: 'markers',\n";
    ofs << "      x: [";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << xs[i];
    }
    ofs << "],\n";

    ofs << "      y: [";
    for (size_t i = 0; i < ys.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << ys[i];
    }
    ofs << "],\n";

    ofs << "      z: [";
    for (size_t i = 0; i < zs.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << zs[i];
    }
    ofs << "],\n";

    ofs << "      text: [";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << "'" << parts[i] << "'";
    }
    ofs << "],\n";

    ofs << "      hovertemplate: '<b>%{text}</b><br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>',\n";
    ofs << "      showlegend: false,\n";
    ofs << "      marker: {\n";
    ofs << "        size: 6,\n";
    ofs << "        color: [";
    for (size_t i = 0; i < node_colors.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << "'" << node_colors[i] << "'";
    }
    ofs << "],\n";
    ofs << "        line: { color: 'rgb(80,80,80)', width: 1 }\n";
    ofs << "      },\n";
    ofs << "      name: 'body parts'\n";
    ofs << "    };\n";

    ofs << "    const dottedEdgeTrace = {\n";
    ofs << "      type: 'scatter3d',\n";
    ofs << "      mode: 'markers',\n";
    ofs << "      x: [";
    for (size_t i = 0; i < dotted_edge_points.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << dotted_edge_points[i][0];
    }
    ofs << "],\n";
    ofs << "      y: [";
    for (size_t i = 0; i < dotted_edge_points.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << dotted_edge_points[i][1];
    }
    ofs << "],\n";
    ofs << "      z: [";
    for (size_t i = 0; i < dotted_edge_points.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << dotted_edge_points[i][2];
    }
    ofs << "],\n";
    ofs << "      hoverinfo: 'skip',\n";
    ofs << "      showlegend: false,\n";
    ofs << "      marker: {\n";
    ofs << "        size: 2.5,\n";
    ofs << "        color: 'rgb(140,140,140)',\n";
    ofs << "        opacity: 0.9\n";
    ofs << "      },\n";
    ofs << "      name: 'skeleton'\n";
    ofs << "    };\n";

    ofs << "    const data = [dottedEdgeTrace, nodeTrace];\n";
    ofs << "    const layout = {\n";
    ofs << "      title: 'Mouse Template Skeleton',\n";
    ofs << "      scene: {\n";
        ofs << "        xaxis: { title: 'X' },\n";
        ofs << "        yaxis: { title: 'Y' },\n";
        ofs << "        zaxis: { title: 'Z' },\n";
        ofs << "        aspectmode: 'data',\n";
        ofs << "        dragmode: 'turntable'\n";
    ofs << "      }\n";
    ofs << "    };\n";
    ofs << "    Plotly.newPlot('plot', data, layout);\n";
    ofs << "  </script>\n";
    ofs << "</body>\n";
    ofs << "</html>\n";

    cout << "Saved interactive skeleton html to " << path << endl;
}

double compute_median(vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

vector<array<double, 3>> filter_spatial_outliers(
    const vector<array<double, 3>>& samples,
    double outlier_thresh_mult) {

    if (samples.size() < 4) {
        return samples;
    }

    const array<double, 3> provisional_center = compute_coordinate_median_point(samples);
    vector<double> distances;
    distances.reserve(samples.size());
    for (const auto& sample : samples) {
        distances.push_back(euclidean_distance_3d(sample, provisional_center));
    }

    const double median_dist = compute_median(distances);
    if (!std::isfinite(median_dist) || median_dist < 1e-9) {
        return samples;
    }

    const double dist_thresh = outlier_thresh_mult * median_dist;
    vector<array<double, 3>> filtered;
    filtered.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        if (distances[i] <= dist_thresh) {
            filtered.push_back(samples[i]);
        }
    }

    if (filtered.empty()) {
        return samples;
    }
    return filtered;
}

SampleCollectionResult collect_part_samples(
    const map<long long, vector<Obs2>>& obs_by_frame,
    const vector<cv::Mat>& Ps,
    const vector<string>& part_order,
    double reproj_thresh_px,
    double angle_thresh_deg,
    int min_parts_for_frame_align) {

    SampleCollectionResult result;
    vector<cv::Point3d> camera_centers;
    camera_centers.reserve(Ps.size());
    for (const auto& P : Ps) {
        camera_centers.push_back(camera_center_from_P(P));
    }

    size_t total_frames = obs_by_frame.size();
    size_t frames_skipped_low_support = 0;
    size_t frames_skipped_missing_head_center = 0;
    size_t frames_skipped_missing_tail_root = 0;
    size_t frames_skipped_missing_lr_reference = 0;
    size_t frames_skipped_bad_alignment = 0;
    size_t frames_accepted = 0;
    size_t accepted_parts_accumulator = 0;
    size_t lr_reference_ears = 0;
    size_t lr_reference_front_paws = 0;
    size_t lr_reference_hind_paws = 0;

    for (const auto& frame_entry : obs_by_frame) {
        unordered_map<string, vector<pair<int, cv::Point2d>>> part2views;
        for (const auto& ob : frame_entry.second) {
            part2views[ob.part].push_back({ob.cam, cv::Point2d(ob.u, ob.v)});
        }

        unordered_map<string, array<double, 3>> frame_points_world;

        for (const auto& kv : part2views) {
            if (kv.second.size() < 2) {
                continue;
            }

            PartSampleStats& stats = result.stats[kv.first];
            stats.raw_candidates++;

            cv::Point3d X = triangulate_DLT(kv.second, Ps);
            if (!std::isfinite(X.x) || !std::isfinite(X.y) || !std::isfinite(X.z)) {
                stats.rejected_invalid++;
                continue;
            }

            const double mean_reproj_error = reproj_error_px(X, kv.second, Ps);
            if (!std::isfinite(mean_reproj_error) || mean_reproj_error > reproj_thresh_px) {
                stats.rejected_reproj++;
                continue;
            }

            const double max_angle_deg = max_triangulation_angle_deg(X, kv.second, camera_centers);
            if (!std::isfinite(max_angle_deg) || max_angle_deg < angle_thresh_deg) {
                stats.rejected_angle++;
                continue;
            }

            frame_points_world[kv.first] = point3d_to_array(X);
        }

        if (static_cast<int>(frame_points_world.size()) < min_parts_for_frame_align) {
            frames_skipped_low_support++;
            continue;
        }

        if (frame_points_world.find("head center") == frame_points_world.end()) {
            frames_skipped_missing_head_center++;
            continue;
        }

        if (frame_points_world.find("tail base") == frame_points_world.end()) {
            frames_skipped_missing_tail_root++;
            continue;
        }

        array<double, 3> left_right_vec = {0.0, 0.0, 0.0};
        string lr_reference_source;
        if (!get_left_right_reference(frame_points_world, left_right_vec, lr_reference_source)) {
            frames_skipped_missing_lr_reference++;
            continue;
        }

        array<double, 3> origin = {0.0, 0.0, 0.0};
        array<double, 3> x_axis = {0.0, 0.0, 0.0};
        array<double, 3> y_axis = {0.0, 0.0, 0.0};
        array<double, 3> z_axis = {0.0, 0.0, 0.0};
        if (!build_frame_alignment_axes(frame_points_world, left_right_vec,
                                        origin, x_axis, y_axis, z_axis)) {
            frames_skipped_bad_alignment++;
            continue;
        }

        if (lr_reference_source == "ears") {
            lr_reference_ears++;
        } else if (lr_reference_source == "front paws") {
            lr_reference_front_paws++;
        } else if (lr_reference_source == "hind paws") {
            lr_reference_hind_paws++;
        }

        for (const auto& kv : frame_points_world) {
            const array<double, 3> X_local =
                transform_world_to_local(kv.second, origin, x_axis, y_axis, z_axis);
            result.part_samples[kv.first].push_back(X_local);
            result.stats[kv.first].accepted++;
        }

        frames_accepted++;
        accepted_parts_accumulator += frame_points_world.size();
    }

    cout << "Frame alignment stats:"
         << " total_frames=" << total_frames
         << ", skipped_low_support=" << frames_skipped_low_support
         << ", skipped_missing_head_center=" << frames_skipped_missing_head_center
         << ", skipped_missing_tail_root=" << frames_skipped_missing_tail_root
         << ", skipped_missing_lr_reference=" << frames_skipped_missing_lr_reference
         << ", skipped_bad_alignment=" << frames_skipped_bad_alignment
         << ", accepted_frames=" << frames_accepted;
    if (frames_accepted > 0) {
        const double mean_parts =
            static_cast<double>(accepted_parts_accumulator) / static_cast<double>(frames_accepted);
        cout << ", mean_accepted_parts=" << mean_parts;
    }
    cout << endl;

    cout << "Left-right reference usage:"
         << " ears=" << lr_reference_ears
         << ", front_paws=" << lr_reference_front_paws
         << ", hind_paws=" << lr_reference_hind_paws << endl;

    for (const string& part : part_order) {
        const PartSampleStats stats = result.stats[part];
        cout << "Part " << part
             << ": raw=" << stats.raw_candidates
             << ", rejected_invalid=" << stats.rejected_invalid
             << ", rejected_reproj=" << stats.rejected_reproj
             << ", rejected_angle=" << stats.rejected_angle
             << ", accepted=" << stats.accepted << endl;
    }

    return result;
}

unordered_map<string, array<double, 3>> aggregate_model_local(
    const unordered_map<string, vector<array<double, 3>>>& part_samples,
    const vector<string>& part_order,
    double outlier_thresh_mult) {

    unordered_map<string, array<double, 3>> model_local;

    for (const string& part : part_order) {
        auto it = part_samples.find(part);
        if (it == part_samples.end() || it->second.empty()) {
            cerr << "WARNING: No valid body-aligned local 3D samples collected for part " << part << endl;
            continue;
        }

        const vector<array<double, 3>> filtered_samples =
            filter_spatial_outliers(it->second, outlier_thresh_mult);
        if (filtered_samples.empty()) {
            cerr << "WARNING: All samples were filtered out for part " << part << endl;
            continue;
        }

        model_local[part] = compute_coordinate_median_point(filtered_samples);

        cout << "Aggregated part " << part
             << " from " << it->second.size()
             << " body-aligned local samples, kept " << filtered_samples.size()
             << ": ("
             << model_local[part][0] << ", "
             << model_local[part][1] << ", "
             << model_local[part][2] << ")" << endl;
    }

    return model_local;
}

int main(int argc, char** argv) {
    string cams_path = "3cam/cams.txt";
    string front_csv = "3cam/obs_de_front.csv";
    string side_csv = "3cam/obs_de_side.csv";
    string top_csv = "3cam/obs_de_top.csv";
    string output_path = "results/mouse_model_obj.csv";
    double fps = 30.0;
    double reproj_thresh_px = 3.0;
    double angle_thresh_deg = 5.0;
    double outlier_thresh_mult = 3.0;
    int min_parts_for_frame_align = 4;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--cams" && i + 1 < argc) {
            cams_path = argv[++i];
        } else if (arg == "--front" && i + 1 < argc) {
            front_csv = argv[++i];
        } else if (arg == "--side" && i + 1 < argc) {
            side_csv = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_csv = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = stod(argv[++i]);
        } else if (arg == "--reproj-thresh" && i + 1 < argc) {
            reproj_thresh_px = stod(argv[++i]);
        } else if (arg == "--angle-thresh" && i + 1 < argc) {
            angle_thresh_deg = stod(argv[++i]);
        } else if (arg == "--outlier-mult" && i + 1 < argc) {
            outlier_thresh_mult = stod(argv[++i]);
        } else if (arg == "--min-parts-frame" && i + 1 < argc) {
            min_parts_for_frame_align = stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --cams <path>       Camera file path (default: 3cam/cams.txt)\n";
            cout << "  --front <path>      Front camera CSV (default: 3cam/obs_de_front.csv)\n";
            cout << "  --side <path>       Side camera CSV (default: 3cam/obs_de_side.csv)\n";
            cout << "  --top <path>        Top camera CSV (default: 3cam/obs_de_top.csv)\n";
            cout << "  --output <path>     Output template CSV (default: results/mouse_model_obj.csv)\n";
            cout << "  --fps <value>       Frame rate (default: 30.0)\n";
            cout << "  --reproj-thresh <value>   Mean reprojection threshold in px (default: 3.0)\n";
            cout << "  --angle-thresh <value>    Minimum triangulation angle in deg (default: 5.0)\n";
            cout << "  --outlier-mult <value>    Spatial outlier distance multiplier (default: 3.0)\n";
            cout << "  --min-parts-frame <value> Minimum accepted parts per frame for body-frame alignment (default: 4)\n";
            cout << "  --help, -h          Show this help\n";
            return 0;
        }
    }

    const string executable_dir = get_executable_dir(argv[0]);
    cams_path = resolve_input_path(cams_path, executable_dir);
    front_csv = resolve_input_path(front_csv, executable_dir);
    side_csv = resolve_input_path(side_csv, executable_dir);
    top_csv = resolve_input_path(top_csv, executable_dir);
    cout << "\n=== Generate Rigid Template ===\n" << endl;
    cout << "FPS: " << fps << endl;
    cout << "Resolved cameras: " << cams_path << endl;
    cout << "Resolved front CSV: " << front_csv << endl;
    cout << "Resolved side CSV: " << side_csv << endl;
    cout << "Resolved top CSV: " << top_csv << endl;
    cout << "Output template CSV: " << output_path << endl;
    cout << "Output skeleton PLY: " << replace_extension(output_path, ".ply") << endl;
    cout << "Output skeleton HTML: " << replace_extension(output_path, ".html") << endl;
    cout << "Reprojection threshold (px): " << reproj_thresh_px << endl;
    cout << "Triangulation angle threshold (deg): " << angle_thresh_deg << endl;
    cout << "Spatial outlier threshold multiplier: " << outlier_thresh_mult << endl;
    cout << "Minimum accepted parts per frame: " << min_parts_for_frame_align << endl;
    cout << endl;

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

    vector<string> part_order;
    map<long long, vector<Obs2>> obs_by_frame = group_obs_by_frame(obsAll, fps, part_order);
    if (obs_by_frame.empty()) {
        cerr << "ERROR: No observations found" << endl;
        return -1;
    }

    cout << "\nStep 2: Collecting head-centered, body-axis aligned local samples across frames..." << endl;
    SampleCollectionResult collection =
        collect_part_samples(obs_by_frame, Ps, part_order,
                             reproj_thresh_px, angle_thresh_deg,
                             min_parts_for_frame_align);

    cout << "\nStep 3: Aggregating per-part body-aligned local samples with coordinate-wise medians..." << endl;
    unordered_map<string, array<double, 3>> model_local =
        aggregate_model_local(collection.part_samples, part_order, outlier_thresh_mult);
    if (model_local.empty()) {
        cerr << "ERROR: Failed to build a body-aligned local template" << endl;
        return -1;
    }

    cout << "\nStep 4: Final template cleanup..." << endl;
    unordered_map<string, array<double, 3>> model_obj = model_local;
    cout << "This template is head-centered and body-axis aligned" << endl;
    cout << "No final centroid recenter applied" << endl;

    cout << "\nStep 5: Saving template..." << endl;
    save_template(output_path, model_obj, part_order);
    save_template_point_cloud_ply(replace_extension(output_path, ".ply"), model_obj, part_order);
    save_template_skeleton_html(replace_extension(output_path, ".html"), model_obj, part_order);

    cout << "\n=== All done! ===" << endl;
    cout << "Generated template with " << model_obj.size() << " parts" << endl;
    return 0;
}
