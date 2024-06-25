#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include <numeric>
#include <limits>
#include <sstream>

using Point = std::pair<double, double>;

const double THRESHOLD = 0.00001; // Threshold to consider centers as "almost the same"

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

void load_points(std::vector<Point> & points, std::string filename) {

    std::ifstream input_file(filename);
    if (!input_file) {
        std::cerr << "Error opening file!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double x, y;
    while (input_file >> x >> y) {
        points.emplace_back(x, y);
    }
    input_file.close();
}

void load_initial_centroids(std::vector<Point>& centroids, int number_of_clusters, std::string filename) {
    std::ifstream centers_file(filename);
    if (!centers_file) {
        std::cerr << "Error opening centers_file!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string header;
    std::getline(centers_file, header); // Read and discard header line

    for (int i = 0; i < number_of_clusters; ++i) {
        centers_file >> centroids[i].first >> centroids[i].second;
    }

    centers_file.close();
}

void calculate_clusters(std::vector<int> & clusters, const std::vector<Point> & local_points, const std::vector<Point> & old_centroids, const int number_of_clusters) {
    // Assign points to clusters
    for (size_t i = 0; i < clusters.size(); ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int assigned_cluster = -1;
        for (int j = 0; j < number_of_clusters; ++j) {
            double distance = euclidean_distance(local_points[i].first, local_points[i].second, old_centroids[j].first, old_centroids[j].second);
            if (distance < min_distance) {
                min_distance = distance;
                assigned_cluster = j;
            }
        }
        clusters[i] = assigned_cluster;
    }
}

// Function to initialize MPI and parse arguments
void initialize_mpi(int argc, char* argv[], int& rank, int& size, int& K) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <number_of_clusters>" << std::endl;
        }
        MPI_Finalize();
        exit(1);
    }

    K = atoi(argv[1]); // Number of clusters (centroids)
    if (K <= 0) {
        if (rank == 0) {
            std::cerr << "Number of clusters must be a positive integer." << std::endl;
        }
        MPI_Finalize();
        exit(1);
    }
}

// Function to scatter points across processes
void scatter_points(int rank, int size, std::vector<Point>& points, std::vector<Point>& local_points, 
    std::vector<int>& counts, std::vector<int>& displacements
) {
    int total_points = points.size();
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of points per process and the displacement
    int points_per_process = total_points / size;
    int remainder = total_points % size;

    counts.resize(size);
    displacements.resize(size);
    for (int i = 0; i < size; ++i) {
        counts[i] = (i < remainder) ? points_per_process + 1 : points_per_process;
        counts[i] *= 2; // Multiply by 2 since each point has two values (x and y)
        displacements[i] = (i == 0) ? 0 : displacements[i - 1] + counts[i - 1];
    }

    // Flatten the points data
    std::vector<double> flat_points(total_points * 2);
    if (rank == 0) {
        for (size_t i = 0; i < points.size(); ++i) {
            flat_points[2 * i] = points[i].first;
            flat_points[2 * i + 1] = points[i].second;
        }
    }

    // Allocate space for local points
    std::vector<double> local_flat_points(counts[rank]);
    MPI_Scatterv(flat_points.data(), counts.data(), displacements.data(), MPI_DOUBLE,
        local_flat_points.data(), local_flat_points.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Convert local_flat_points back to vector<Point>
    local_points.resize(local_flat_points.size() / 2);
    for (size_t i = 0; i < local_points.size(); ++i) {
        local_points[i] = { local_flat_points[2 * i], local_flat_points[2 * i + 1] };
    }
}

// Function to compute new centroids
void compute_new_centroids(int K, const std::vector<Point>& local_points, const std::vector<int>& local_assignments,
    std::vector<Point>& local_sums, std::vector<int>& local_counts
) {
    for (size_t i = 0; i < local_points.size(); ++i) {
        int cluster_id = local_assignments[i];
        local_sums[cluster_id].first += local_points[i].first;
        local_sums[cluster_id].second += local_points[i].second;
        local_counts[cluster_id]++;
    }
}

// Function to reduce centroids to global values
void reduce_centroids(int K, const std::vector<Point>& local_sums, const std::vector<int>& local_counts,
    std::vector<Point>& global_sums, std::vector<int>& global_counts) {
    MPI_Reduce(local_sums.data(), global_sums.data(), 2 * K, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_counts.data(), global_counts.data(), K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

// Function to print results
void print_results(int total_points, const std::vector<int>& global_assignments, const std::vector<Point>& final_centers) {
    std::ofstream assignments_file("assignments.txt");
    for (int i = 0; i < total_points; ++i) {
        assignments_file << global_assignments[i] << std::endl;
    }

    std::ofstream centers_file("final_centers.txt");
    for (const auto& center : final_centers) {
        centers_file << center.first << ", " << center.second << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int rank, size, K;
    initialize_mpi(argc, argv, rank, size, K);

    std::vector<Point> points;
    std::vector<Point> old_centroids(K);
    // Read data from file on rank 0
    if (rank == 0) {
        load_points(points, "data.txt");
        load_initial_centroids(old_centroids, K, "cluster_centers.txt");
    }

    // Broadcast total number of points to all processes
    int total_points = points.size();
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of points per process and the displacement
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<Point> local_points;
    scatter_points(rank, size, points, local_points, counts, displacements);

    // Broadcast initial cluster centers to all processes
    MPI_Bcast(old_centroids.data(), K * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure the execution time
    double start_time = MPI_Wtime();

    bool centers_almost_same = false;
    int iteration = 0;
    std::vector<int> global_assignments(total_points); // Allocate space for global assignments
    std::vector<Point> final_centers(K); // Allocate space for final centers

    while (true) {
        // Assign points to clusters
        std::vector<int> local_assignments(local_points.size());
        calculate_clusters(local_assignments, local_points, old_centroids, K);

        // Calculate local sums and counts for new cluster centers
        std::vector<Point> local_sums(K, { 0.0, 0.0 });
        std::vector<int> local_counts(K, 0);
        compute_new_centroids(K, local_points, local_assignments, local_sums, local_counts);

        // Reduce local sums and counts to calculate global sums and counts
        std::vector<Point> global_sums(K, { 0.0, 0.0 });
        std::vector<int> global_counts(K, 0);
        reduce_centroids(K, local_sums, local_counts, global_sums, global_counts);

        // Calculate new cluster centers and compare on rank 0
        if (rank == 0) {
            iteration++;
            std::vector<Point> new_centers(K, { 0.0, 0.0 });
            for (int i = 0; i < K; ++i) {
                if (global_counts[i] != 0) {
                    new_centers[i].first = global_sums[i].first / global_counts[i];
                    new_centers[i].second = global_sums[i].second / global_counts[i];
                }
            }

            // Compare old and new centers
            centers_almost_same = are_centers_almost_the_same(old_centroids, new_centers, THRESHOLD);
            old_centroids = new_centers;

            final_centers = new_centers; // Store the final centers
        }

        // Broadcast new cluster centers to all processes
        MPI_Bcast(old_centroids.data(), K * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Broadcast the termination flag
        MPI_Bcast(&centers_almost_same, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // Gather all assignments to rank 0 at the end of the loop
        MPI_Gatherv(local_assignments.data(), local_assignments.size(), MPI_INT,
            global_assignments.data(), counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

        if (centers_almost_same) {
            break;
        }
    }

    // Measure the end time and print the elapsed time
    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Total execution time: " << end_time - start_time << " seconds" << std::endl;
        std::cout << "Number of iterations: " << iteration << std::endl;

        // Print results
        print_results(points.size(), global_assignments, final_centers);

        system("python draw.py");
    }

    MPI_Finalize();
    return 0;
}
