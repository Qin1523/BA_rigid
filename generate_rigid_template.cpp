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

void save_template(const string& path,
                   const unordered_map<string, array<double, 3>>& model) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "WARNING: Cannot write " << path << endl;
        return;
    }

    ofs << "part,X,Y,Z\n";
    for (const auto& kv : model) {
        ofs << kv.first << ","
            << kv.second[0] << ","
            << kv.second[1] << ","
            << kv.second[2] << "\n";
    }

    cout << "Saved template to " << path << endl;
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

unordered_map<string, vector<array<double, 3>>> collect_part_samples(
    const map<long long, vector<Obs2>>& obs_by_frame,
    const vector<cv::Mat>& Ps) {

    unordered_map<string, vector<array<double, 3>>> part_samples;

    for (const auto& frame_entry : obs_by_frame) {
        unordered_map<string, vector<pair<int, cv::Point2d>>> part2views;
        for (const auto& ob : frame_entry.second) {
            part2views[ob.part].push_back({ob.cam, cv::Point2d(ob.u, ob.v)});
        }

        for (const auto& kv : part2views) {
            if (kv.second.size() < 2) {
                continue;
            }

            cv::Point3d X = triangulate_DLT(kv.second, Ps);
            if (!std::isfinite(X.x) || !std::isfinite(X.y) || !std::isfinite(X.z)) {
                continue;
            }

            part_samples[kv.first].push_back({X.x, X.y, X.z});
        }
    }

    return part_samples;
}

unordered_map<string, array<double, 3>> aggregate_model_world(
    const unordered_map<string, vector<array<double, 3>>>& part_samples,
    const vector<string>& part_order) {

    unordered_map<string, array<double, 3>> model_world;

    for (const string& part : part_order) {
        auto it = part_samples.find(part);
        if (it == part_samples.end() || it->second.empty()) {
            cerr << "WARNING: No valid 3D samples collected for part " << part << endl;
            continue;
        }

        vector<double> xs;
        vector<double> ys;
        vector<double> zs;
        xs.reserve(it->second.size());
        ys.reserve(it->second.size());
        zs.reserve(it->second.size());

        for (const auto& sample : it->second) {
            xs.push_back(sample[0]);
            ys.push_back(sample[1]);
            zs.push_back(sample[2]);
        }

        model_world[part] = {
            compute_median(xs),
            compute_median(ys),
            compute_median(zs)
        };

        cout << "Aggregated part " << part << " from " << it->second.size()
             << " samples: ("
             << model_world[part][0] << ", "
             << model_world[part][1] << ", "
             << model_world[part][2] << ")" << endl;
    }

    return model_world;
}

unordered_map<string, array<double, 3>> center_model(
    const unordered_map<string, array<double, 3>>& model_world,
    array<double, 3>& centroid_world) {

    centroid_world = {0.0, 0.0, 0.0};
    if (model_world.empty()) {
        return {};
    }

    for (const auto& kv : model_world) {
        centroid_world[0] += kv.second[0];
        centroid_world[1] += kv.second[1];
        centroid_world[2] += kv.second[2];
    }

    const double inv_count = 1.0 / static_cast<double>(model_world.size());
    centroid_world[0] *= inv_count;
    centroid_world[1] *= inv_count;
    centroid_world[2] *= inv_count;

    cout << "Template centroid in world coordinates: ("
         << centroid_world[0] << ", "
         << centroid_world[1] << ", "
         << centroid_world[2] << ")" << endl;

    unordered_map<string, array<double, 3>> model_obj;
    for (const auto& kv : model_world) {
        model_obj[kv.first] = {
            kv.second[0] - centroid_world[0],
            kv.second[1] - centroid_world[1],
            kv.second[2] - centroid_world[2]
        };
    }

    return model_obj;
}

int main(int argc, char** argv) {
    string cams_path = "3cam_rigid/cams.txt";
    string front_csv = "3cam_rigid/obs_de_front.csv";
    string side_csv = "3cam_rigid/obs_de_side.csv";
    string top_csv = "3cam_rigid/obs_de_top.csv";
    string output_path = "mouse_model_obj.csv";
    double fps = 30.0;

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
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --cams <path>       Camera file path (default: 3cam_rigid/cams.txt)\n";
            cout << "  --front <path>      Front camera CSV (default: 3cam_rigid/obs_de_front.csv)\n";
            cout << "  --side <path>       Side camera CSV (default: 3cam_rigid/obs_de_side.csv)\n";
            cout << "  --top <path>        Top camera CSV (default: 3cam_rigid/obs_de_top.csv)\n";
            cout << "  --output <path>     Output template CSV (default: mouse_model_obj.csv)\n";
            cout << "  --fps <value>       Frame rate (default: 30.0)\n";
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

    cout << "\nStep 2: Collecting triangulated samples across frames..." << endl;
    unordered_map<string, vector<array<double, 3>>> part_samples =
        collect_part_samples(obs_by_frame, Ps);
    for (const string& part : part_order) {
        size_t sample_count = 0;
        auto it = part_samples.find(part);
        if (it != part_samples.end()) {
            sample_count = it->second.size();
        }
        cout << "Part " << part << ": " << sample_count << " triangulated samples" << endl;
    }

    cout << "\nStep 3: Aggregating per-part samples with coordinate-wise medians..." << endl;
    unordered_map<string, array<double, 3>> model_world =
        aggregate_model_world(part_samples, part_order);
    if (model_world.empty()) {
        cerr << "ERROR: Failed to build a world-space template" << endl;
        return -1;
    }

    cout << "\nStep 4: Centering the template at its centroid..." << endl;
    array<double, 3> centroid_world = {0.0, 0.0, 0.0};
    unordered_map<string, array<double, 3>> model_obj =
        center_model(model_world, centroid_world);

    cout << "\nStep 5: Saving template..." << endl;
    save_template(output_path, model_obj);

    cout << "\n=== All done! ===" << endl;
    cout << "Generated template with " << model_obj.size() << " parts" << endl;
    return 0;
}
