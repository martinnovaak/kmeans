#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <limits>
#include <sstream>

using Point = std::pair<double, double>;

const double THRESHOLD = 0.0001; // Threshold to consider centers as "almost the same"

// Function to calculate Euclidean distance between two points
double euclidean_distance(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// Function to compare old and new centers
bool are_centers_almost_the_same(const std::vector<Point>& old_centers,
    const std::vector<Point>& new_centers, double threshold) {
    for (size_t i = 0; i < old_centers.size(); ++i) {
        if (euclidean_distance(old_centers[i].first, old_centers[i].second,
            new_centers[i].first, new_centers[i].second) >= threshold) {
            return false;
        }
    }
    return true;
}

// Function to load points from file
void load_points(std::vector<Point>& points, const std::string& filename) {
    std::ifstream input_file(filename);
    if (!input_file) {
        std::cerr << "Error opening file!" << std::endl;
        exit(1);
    }

    double x, y;
    while (input_file >> x >> y) {
        points.emplace_back(x, y);
    }
    input_file.close();
}

// Function to load initial centroids from file
void load_initial_centroids(std::vector<Point>& centroids, int number_of_clusters, const std::string& filename) {
    std::ifstream centers_file(filename);
    if (!centers_file) {
        std::cerr << "Error opening centers file!" << std::endl;
        exit(1);
    }

    for (int i = 0; i < number_of_clusters; ++i) {
        double x, y;
        centers_file >> x >> y;
        centroids[i] = { x, y };
    }
    centers_file.close();
}

// Function to calculate clusters
void calculate_clusters(std::vector<int>& clusters, const std::vector<Point>& points,
    const std::vector<Point>& centroids, int number_of_clusters) {
    for (size_t i = 0; i < clusters.size(); ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int assigned_cluster = -1;
        for (int j = 0; j < number_of_clusters; ++j) {
            double distance = euclidean_distance(points[i].first, points[i].second,
                centroids[j].first, centroids[j].second);
            if (distance < min_distance) {
                min_distance = distance;
                assigned_cluster = j;
            }
        }
        clusters[i] = assigned_cluster;
    }
}

// Function to compute new centroids
void compute_new_centroids(int K, const std::vector<Point>& points, const std::vector<int>& assignments,
    std::vector<Point>& centroids) {
    std::vector<Point> sums(K, { 0.0, 0.0 });
    std::vector<int> counts(K, 0);

    for (size_t i = 0; i < points.size(); ++i) {
        int cluster_id = assignments[i];
        sums[cluster_id].first += points[i].first;
        sums[cluster_id].second += points[i].second;
        counts[cluster_id]++;
    }

    for (int i = 0; i < K; ++i) {
        if (counts[i] != 0) {
            centroids[i].first = sums[i].first / counts[i];
            centroids[i].second = sums[i].second / counts[i];
        }
    }
}

// Function to print results
void print_results(const std::vector<int>& assignments, const std::vector<Point>& centroids) {
    std::ofstream assignments_file("assignments.txt");
    for (const auto& cluster : assignments) {
        assignments_file << cluster << std::endl;
    }

    std::ofstream centers_file("final_centers.txt");
    for (const auto& center : centroids) {
        centers_file << center.first << ", " << center.second << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int K = 4;
    if (argc == 2) {
        /*
        std::cerr << "Usage: " << argv[0] << " <number_of_clusters>" << std::endl;
        return 1;
        */
        K = std::stoi(argv[1]);
    }

    //int K = std::stoi(argv[1]); // Number of clusters (centroids)
    if (K <= 0) {
        std::cerr << "Number of clusters must be a positive integer." << std::endl;
        return 1;
    }

    std::vector<Point> points;
    std::vector<Point> centroids(K);

    load_points(points, "data.txt");
    load_initial_centroids(centroids, K, "cluster_centers.txt");

    std::vector<int> assignments(points.size());

    bool centers_almost_same = false;
    int iteration = 0;

    while (true) {
        // Assign points to clusters
        calculate_clusters(assignments, points, centroids, K);

        // Compute new centroids
        std::vector<Point> new_centroids(K, { 0.0, 0.0 });
        compute_new_centroids(K, points, assignments, new_centroids);

        // Check for convergence
        centers_almost_same = are_centers_almost_the_same(centroids, new_centroids, THRESHOLD);
        centroids = new_centroids;

        iteration++;
        if (centers_almost_same) {
            break;
        }
    }

    std::cout << "Total number of iterations: " << iteration << std::endl;

    // print_results(assignments, centroids);

    system("python draw.py");

    return 0;
}
