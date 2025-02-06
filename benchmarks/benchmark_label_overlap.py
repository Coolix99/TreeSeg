import numpy as np
import time
from tree_seg.metrices.overlap import label_overlap

def generate_random_labels(size, num_labels=10):
    """Generate a random label matrix of a given size and number of labels."""
    return np.random.randint(0, num_labels, size, dtype=np.uint32)

def benchmark_label_overlap(method):
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000), (5000, 5000)]
    num_labels = 10  # Number of unique labels for the random images

    # Warm-up to compile with Numba
    print("Warming up...")
    x = generate_random_labels((100, 100), num_labels)
    y = generate_random_labels((100, 100), num_labels)
    label_overlap(x, y,method=method)  # Run once to allow Numba compilation

    print("Benchmarking label_overlap with increasing data sizes")
    print("Size\t\tTotal Pixels\tTime (s)")

    for size in sizes:
        x = generate_random_labels(size, num_labels)
        y = generate_random_labels(size, num_labels)
        
        # Time the label_overlap function
        start_time = time.time()
        overlap_matrix = label_overlap(x, y,method=method)
        end_time = time.time()

        total_pixels = size[0] * size[1]
        print(f"{size}\t{total_pixels}\t\t{end_time - start_time:.4f} seconds")

def benchmark_label_overlap_label_count(method):
    fixed_size = (5000, 5000)  # Fix the image size
    label_counts = [10, 50, 100, 200, 500, 1000]  # Vary the number of labels

    # Warm-up to compile with Numba
    print("Warming up...")
    x = generate_random_labels(100, 10)
    y = generate_random_labels(100, 10)
    label_overlap(x, y,method=method)  # Run once to allow Numba compilation

    print("Benchmarking label_overlap with increasing label counts")
    print("Labels\t\tTime (s)")

    for num_labels in label_counts:
        x = generate_random_labels(fixed_size, num_labels)
        y = generate_random_labels(fixed_size, num_labels)

        # Time the label_overlap function
        start_time = time.time()
        overlap_matrix = label_overlap(x, y,method=method)
        end_time = time.time()

        print(f"{num_labels}\t\t{end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    method='numba'
    benchmark_label_overlap(method)
    benchmark_label_overlap_label_count(method)
