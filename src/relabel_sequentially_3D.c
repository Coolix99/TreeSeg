#include <stdio.h>
#include <stdlib.h>

void relabel_sequentially_3D(int *labels, int x_dim, int y_dim, int z_dim, int max_label) {
    int *label_mapping = (int *)calloc(max_label + 1, sizeof(int));
    int current_label = 1;

    // First pass: Build the mapping for unique labels
    for (int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {
            for (int z = 0; z < z_dim; z++) {
                int index = x * y_dim * z_dim + y * z_dim + z;
                int label = labels[index];
                if (label > 0 && label_mapping[label] == 0) {
                    label_mapping[label] = current_label++;
                }
            }
        }
    }

    // Second pass: Apply the mapping to relabel the array
    for (int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {
            for (int z = 0; z < z_dim; z++) {
                int index = x * y_dim * z_dim + y * z_dim + z;
                int label = labels[index];
                if (label > 0) {
                    labels[index] = label_mapping[label];
                }
            }
        }
    }

    free(label_mapping);
}
