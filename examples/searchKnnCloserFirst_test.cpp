// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>

namespace
{

using idx_t = hnswlib::labeltype;

void test() {
    int d = 128;
    idx_t n = 10000;
    idx_t nq = 10;
    size_t M = 20;
    size_t ef_construction = 200;
    size_t ef = 200;
    size_t k = 10;
   
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
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&space, 2 * n);
    hnswlib::AlgorithmInterface<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * n, M, ef_construction, ef);

    for (size_t i = 0; i < n; ++i) {
        alg_brute->addPoint(data.data() + d * i, i);
        alg_hnsw->addPoint(data.data() + d * i, i);
        if (i%10 == 9) {
            int label = distrib(rng) * i;
            alg_brute->removePoint(label);
            alg_hnsw->removePoint(label);
        }
    }
    alg_hnsw->checkIntegrity();
    std::cout << "finish building index" << std::endl;

    // test searchKnnCloserFirst of BruteforceSearch
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
        auto hnsw_res = alg_hnsw->searchKnnCloserFirst(p, k);
        for (auto res : hnsw_res) {
            for (auto expect_res : bf_res) {
                if (res.second == expect_res.second) {
                    correct++;
                    break;
                }
            }
        }
        std::cout << "correct: " << correct << " out of " << k << std::endl;
        total_correct+=correct;
    }
    std::cout << "recall is: " << float(total_correct)/(k*nq) << std::endl;

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

int main() {
    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}
