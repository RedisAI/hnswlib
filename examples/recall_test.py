import hnswlib
import numpy as np
from RLTest import Env
import time


# Get Redis connection
module_path = "/home/alon/Repositories/redis_hnsw/target/release/libredis_hnsw.so"
env = Env(module=module_path)
con = env.getConnection()

dim = 1280
num_elements = 1000000
M = 64
ef = 500
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
con.execute_command('HNSW.NEW', 'rust_hnsw_index', 'DIM', dim, 'M', M, 'EFCON', ef)

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
rust_insert_time = 0
hnsw_delete_time = 0
rust_delete_time = 0

for i in range(len(data_batches)):
    ids = np.array([10*i + j for j in range(10)])
    hnsw_start = time.time()
    hnsw_index.add_items(data_batches[i], ids)
    hnsw_end = time.time()
    hnsw_insert_time += (hnsw_end-hnsw_start)
    bf_index.add_items(data_batches[i], ids)

    # Insert one by one to the rust index
    for j, vector in enumerate(data_batches[i]):
        node_id = 10*i + j
        vec_values_as_list = [str(entry) for entry in vector]
        rust_start = time.time()
        # con.execute_command('HNSW.NODE.ADD', 'rust_hnsw_index', f'{node_id}', 'DATA', dim, *vec_values_as_list)
        rust_end = time.time()
        rust_insert_time += (rust_end-rust_start)

    label_to_delete = np.random.randint(10*(i+1))
    while label_to_delete in deleted_labels:
        label_to_delete = np.random.randint(10*(i+1))

    hnsw_start = time.time()
    hnsw_index.delete_vector(label_to_delete)
    hnsw_end = time.time()
    hnsw_delete_time += (hnsw_end-hnsw_start)

    bf_index.delete_vector(label_to_delete)

    rust_start = time.time()
    # con.execute_command('HNSW.NODE.DEL', 'rust_hnsw_index', f'{label_to_delete}')
    rust_end = time.time()
    rust_delete_time += (rust_end-rust_start)

    deleted_labels.add(label_to_delete)

print("Indices built")
hnsw_index.check_integrity()
print("HNSW insert time: ", hnsw_insert_time)
print("HNSW delete time: ", hnsw_delete_time)
print("rust insert time: ", rust_insert_time)
print("rust delete time: ", rust_delete_time)

# Generating query data
query_data = np.float32(np.random.random((10, dim)))

# Query the elements and measure recall:
search_time_start = time.time()
labels_hnsw, distances_hnsw = hnsw_index.knn_query(query_data, k)
search_time_end = time.time()
print("search time is: ", search_time_end-search_time_start)
labels_bf, distances_bf = bf_index.knn_query(query_data, k)

# Query one by one for rust index
'''labels_rust = []
for vector in query_data:
    vec_values_as_list = [str(entry) for entry in vector]
    rust_res = con.execute_command('HNSW.SEARCH', 'rust_hnsw_index', 'K', k, 'QUERY', dim, *vec_values_as_list)
    labels_rust.append([int(rust_res[i+1][3]) for i in range(k)])'''

# Measure recall

correct = 0
for i in range(nun_queries):
    for label in labels_hnsw[i]:
        for correct_label in labels_bf[i]:
            if label == correct_label:
                correct += 1
                break

print("recall is :", float(correct)/(k*nun_queries))

'''correct_rust = 0
for i in range(nun_queries):
    for label in labels_rust[i]:
        for correct_label in labels_bf[i]:
            if label == correct_label:
                correct_rust += 1
                break

print("rust recall is :", float(correct_rust)/(k*nun_queries))'''
env.stop()
