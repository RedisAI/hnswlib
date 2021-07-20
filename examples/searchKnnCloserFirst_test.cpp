// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <sys/resource.h>

namespace
{

using idx_t = hnswlib::labeltype;

    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        } else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= end)) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        } catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                             * This will work even when current is the largest value that
                             * size_t can fit, because fetch_add returns the previous value
                             * before the increment (what will result in overflow
                             * and produce 0 instead of current + 1).
                             */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }


    }

void test(int d, long n, int k, int M, int ef_c, int ef) {
    idx_t nq = 10;
   
    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }
      

    hnswlib::L2Space space(d);
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&space, n);
    hnswlib::AlgorithmInterface<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef_c, ef);


    long long insert_duration = 0;
    long long remove_duration = 0;
    long long total_duration = 0;
    auto valid_labels = std::vector<size_t>();
    int next_stat_print = 10;

//    for (size_t i = 0; i < n; ++i) {
//
//        alg_brute->addPoint(data.data() + d * i, i);
//
//        auto start = std::chrono::high_resolution_clock::now();
//        alg_hnsw->addPoint(data.data() + d * i, i);
//        auto elapsed = std::chrono::high_resolution_clock::now() - start;
//        insert_duration += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//        if (i%next_stat_print==9 ) {
//            std::cerr << "after " << i << " avg insert takes: " <<
//            insert_duration / (i+1) << std::endl;
//            struct rusage self_ru{};
//            getrusage(RUSAGE_SELF, &self_ru);
//            std::cerr << "memory usage is : " << self_ru.ru_maxrss << std::endl;
//        }
//
//        valid_labels.push_back(i);
//
//        if (i%10 == 9) {
//            auto label_index = (size_t)(distrib(rng) * valid_labels.size());
//            auto label = valid_labels[label_index];
//
//            if(!alg_brute->removePoint(label)) {
//                throw std::runtime_error("Trying to remove an element that doesn't exist");
//            }
//
//            start = std::chrono::high_resolution_clock::now();
//            alg_hnsw->removePoint(label);
//            elapsed = std::chrono::high_resolution_clock::now() - start;
//            remove_duration += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
//            if (i%next_stat_print == 9) {
//                std::cerr << "after " << i << " steps, avg delete takes: " <<
//                          remove_duration / ((i/10) + 1) << std::endl;
//                next_stat_print *= 10;
//            }
//            valid_labels[label_index] = valid_labels[valid_labels.size()-1];
//            valid_labels.pop_back();
//        }
//    }

    for (size_t i = 0; i < n; i+=10) {
        auto start = std::chrono::high_resolution_clock::now();
        ParallelFor(i, i+10, 8, [&](size_t id, size_t threadId) {
            alg_hnsw->addPoint(data.data() + d * id, id);
        });
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        insert_duration += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

        for (size_t j=i; j<(i+10); j++) {
            valid_labels.push_back(j);
        }

        if (i == next_stat_print) {
            std::cerr << "after " << i << " avg insert takes: " <<
            insert_duration / (i+1) << std::endl;
            struct rusage self_ru{};
            getrusage(RUSAGE_SELF, &self_ru);
            std::cerr << "memory usage is : " << self_ru.ru_maxrss << std::endl;
        }
        ParallelFor(i, i+10, 8, [&](size_t id, size_t threadId) {
            alg_brute->addPoint(data.data() + d * id, id);
        });

        // after 10 insert, delete one
        auto label_index = (size_t)(distrib(rng) * valid_labels.size());
        auto label = valid_labels[label_index];

        if(!alg_brute->removePoint(label)) {
            throw std::runtime_error("Trying to remove an element that doesn't exist");
        }

        start = std::chrono::high_resolution_clock::now();
        alg_hnsw->removePoint(label);
        elapsed = std::chrono::high_resolution_clock::now() - start;
        remove_duration += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (i == next_stat_print) {
            std::cerr << "after " << i << " steps, avg delete takes: " << remove_duration / ((i/10) + 1) << std::endl;
            next_stat_print *= 10;
        }
        valid_labels[label_index] = valid_labels[valid_labels.size()-1];
        valid_labels.pop_back();
    }


    total_duration = insert_duration + remove_duration;
    std::cerr << "total insert time in microseconds: " << insert_duration << std::endl;
    std::cerr << "total remove time in microseconds: " << remove_duration << std::endl;
    std::cerr << "total build time in microseconds: " << total_duration << std::endl;

    // test searchKnnCloserFirst of BruteforceSearch
    long long search_time =0;
    size_t total_correct = 0;
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_brute->searchKnn(p, k);
        auto bf_res = alg_brute->searchKnnCloserFirst(p, k);
        assert(gd.size() == bf_res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == bf_res[--t]);
            gd.pop();
        }
        size_t correct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto hnsw_res = alg_hnsw->searchKnnCloserFirst(p, k);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        search_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        for (auto res : hnsw_res) {
            for (auto expect_res : bf_res) {
                if (res.second == expect_res.second) {
                    correct++;
                    break;
                }
            }
        }
        //std::cout << "correct: " << correct << " out of " << k << std::endl;
        total_correct+=correct;
    }
    std::cerr << "total search time in microseconds: " << search_time << std::endl;
    std::cerr << "recall is: " << float(total_correct)/(k*nq) << std::endl;
    std::cout << float(total_correct)/(k*nq) << std::endl;

    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            gd.pop();
        }
    }
    delete alg_brute;
    delete alg_hnsw;
}

} // namespace

int main(int argc, char** argv) {
    //std::cout << "Testing ..." << std::endl;
    int d;
    long n;
    int k;
    int M;
    int ef_c;
    int ef;

    int i=1;
    while (i<argc) {
        if(strcasecmp(argv[i], "d") == 0) {
            d = std::stoi(argv[++i]);
        } else if(strcasecmp(argv[i], "n") == 0) {
            n = std::stol(argv[++i]);
        } else if(strcasecmp(argv[i], "k") == 0) {
            k = std::stoi(argv[++i]);
        } else if(strcasecmp(argv[i], "M") == 0) {
            M = std::stoi(argv[++i]);
        } else if(strcasecmp(argv[i], "ef_c") == 0) {
            ef_c = std::stoi(argv[++i]);
        } else if(strcasecmp(argv[i], "ef") == 0) {
            ef = std::stoi(argv[++i]);
        }
        i++;
    }

    std::cerr << "d=" << d << ", n=" << n << ", k="<< k << ", M=" << M << ", ef_c=" << ef_c << ", ef=" << ef << std::endl;
    test(d,n,k,M,ef_c,ef);
    //std::cout << "Test ok" << std::endl;

    return 0;
}
