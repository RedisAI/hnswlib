import hnswlib
import numpy as np
import time

dim = 128
num_elements = 1000
M = 6
ef = 10
k = 10
nun_queries = 10

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# Split the data to batches of ten:
data_batches = [data[10*i:10*(i+1)] for i in range(num_elements//10)]
print("data_batch_size: ", len(data_batches[0]))
print("total_data_batches: ", len(data_batches))

# Declaring index
hnsw_index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
bf_index = hnswlib.BFIndex(space='l2', dim=dim)

# Initing both hnsw and brute force indices
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# hnsw construction params:
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

hnsw_index.init_index(max_elements=num_elements, ef_construction=ef, M=M)
bf_index.init_index(max_elements=num_elements)

# Controlling the recall for hnsw by setting ef:
# higher ef leads to better accuracy, but slower search
hnsw_index.set_ef(ef)

# Set number of threads used during batch search/construction in hnsw
# By default using all available cores
hnsw_index.set_num_threads(1)

# Keep the deleted labels so we won't remove a deleted item
deleted_labels = set()

print("Adding batches of %d elements" % (len(data_batches[0])))

hnsw_insert_time = 0
hnsw_delete_time = 0

for i in range(len(data_batches)):
    ids = np.array([10*i + j for j in range(10)])
    hnsw_start = time.time()
    hnsw_index.add_items(data_batches[i], ids)
    hnsw_end = time.time()
    hnsw_insert_time += (hnsw_end-hnsw_start)
    bf_index.add_items(data_batches[i], ids)

    label_to_delete = np.random.randint(10*(i+1))
    while label_to_delete in deleted_labels:
        label_to_delete = np.random.randint(10*(i+1))

    hnsw_start = time.time()
    hnsw_index.delete_vector(label_to_delete)
    hnsw_end = time.time()
    hnsw_delete_time += (hnsw_end-hnsw_start)

    bf_index.delete_vector(label_to_delete)

    deleted_labels.add(label_to_delete)

print("Indices built")
hnsw_index.check_integrity()
print("HNSW insert time: ", hnsw_insert_time)
print("HNSW delete time: ", hnsw_delete_time)

# Generating query data
query_data = np.float32(np.random.random((10, dim)))

# Query the elements and measure recall:
search_time_start = time.time()
labels_hnsw, distances_hnsw = hnsw_index.knn_query(query_data, k)
search_time_end = time.time()
print("search time is: ", search_time_end-search_time_start)
labels_bf, distances_bf = bf_index.knn_query(query_data, k)

# Measure recall

correct = 0
for i in range(nun_queries):
    for label in labels_hnsw[i]:
        for correct_label in labels_bf[i]:
            if label == correct_label:
                correct += 1
                break

print("recall is :", float(correct)/(k*nun_queries))
