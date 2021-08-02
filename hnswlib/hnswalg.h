#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <sys/resource.h>
#include <unordered_set>
#include <list>
#include <set>

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    template<typename dist_t>
    using CandidatesQueue = std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst<dist_t>>;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:

        // Index build parameters
        size_t max_elements_;
        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        // Index search parameter
        size_t ef_;

        // Index meta-data (based on the data dimensionality and index parameters)
        size_t data_size_;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t size_links_level0_;
        size_t label_offset_;
        size_t offsetData_, offsetLevel0_;
        size_t incoming_links_offset0;
        size_t incoming_links_offset;
        double mult_;

        // Index level generator of the top level for a new element
        std::default_random_engine level_generator_;

        // Index state
        size_t cur_element_count;
        size_t max_id;
        int maxlevel_;

        // Index data structures
        tableint enterpoint_node_;
        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;
        std::set<tableint> available_ids;
        std::unordered_map<labeltype, tableint> label_lookup_;
        VisitedListPool *visited_list_pool_;

        // used for synchronization only when parallel indexing / searching is enabled.
        std::mutex global;
        std::mutex cur_element_count_guard_;
        std::vector<std::mutex> link_list_locks_;

        // callback for computing distance between two points in the underline space.
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t ef = 10, size_t random_seed = 100) :
                link_list_locks_(max_elements), element_levels_(max_elements) {

            max_elements_ = max_elements;
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = ef;

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();


            cur_element_count = 0;
            visited_list_pool_ = new VisitedListPool(1, (int)max_elements);

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            mult_ = 1 / log(1.0 * M_);
            level_generator_.seed(random_seed);

            // data_level0_memory will look like this:
            // -----4------ | -----4*M0----------- | ----8------------------| ------32------- | ----8---- |
            // <links_len>  | <link_1> <link_2>... | <incoming_links_set> |   <data>        |  <label>
            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint)
              + sizeof(void*);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            incoming_links_offset0 = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");


            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");

            // The i-th entry in linkLists array points to max_level[i] (continuous)
            // chunks of memory, each one will look like this:
            // -----4------ | -----4*M-------------- | ----8------------------|
            // <links_len>  | <link_1> <link_2> ...  | <incoming_links_set>
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint)
              + sizeof(void *);
            incoming_links_offset = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        }

        ~HierarchicalNSW() {

            for (tableint id = 0; id < max_id; id++) {
                if (available_ids.find(id) != available_ids.end()) {
                    continue;
                }
                for (size_t level = 0; level <= element_levels_[id]; level++) {
                    delete getIncomingEdgesPtr(id, level);
                }

                if (element_levels_[id] > 0)
                    free(linkLists_[id]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        void setEf(size_t ef) {
            ef_ = ef;
        }

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabelPtr(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }

        std::set<tableint> *getIncomingEdgesPtr(tableint internal_id, int level) {
            if (level == 0) {
                return reinterpret_cast<std::set<tableint>*>(*(void**) (data_level0_memory_ +
                                internal_id * size_data_per_element_ +
                                incoming_links_offset0));
            }
            return reinterpret_cast<std::set<tableint>*>(*(void**) (linkLists_[internal_id] +
            (level - 1) * size_links_per_element_ + incoming_links_offset));
        }

        void setIncomingEdgesPtr(tableint internal_id, int level, void *set_ptr) {
            if (level == 0) {
                memcpy(data_level0_memory_ + internal_id * size_data_per_element_ + incoming_links_offset0,
                       &set_ptr, sizeof(void*));
            } else {
                memcpy(linkLists_[internal_id] + (level - 1) * size_links_per_element_ + incoming_links_offset,
                       &set_ptr, sizeof(void*));
            }
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        CandidatesQueue<dist_t> searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            CandidatesQueue<dist_t> top_candidates;
            CandidatesQueue<dist_t> candidateSet;

            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);

            dist_t lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);

            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;
#ifdef ENABLE_PARALLELIZATION
                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);
#endif
                linklistsizeint *data = get_linklist_at_level(curNodeNum, layer);
                size_t size = getListCount(data);
                auto *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool collect_metrics=false>
        CandidatesQueue<dist_t> searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            CandidatesQueue<dist_t> top_candidates;
            CandidatesQueue<dist_t> candidate_set;

            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            dist_t lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);

                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=(long)size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (visited_array[candidate_id] != visited_array_tag) {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
#endif

                            top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        void getNeighborsByHeuristic2(CandidatesQueue<dist_t> &top_candidates, const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                // the distance is saved negatively to have the queue ordered such that first is closer (higher).
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> current_pair = queue_closest.top();
                dist_t dist_to_query = -current_pair.first;
                queue_closest.pop();
                bool good = true;

                // a candidate is "good" to become a neighbour, unless we find
                // another item that was already selected to the neighbours set which is closer
                // to both q and the candidate than the distance between the candidate and q.
                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(current_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(current_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        tableint mutuallyConnectNewElement(tableint cur_c, CandidatesQueue<dist_t> &top_candidates, int level) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();
            {
                linklistsizeint *ll_cur = get_linklist_at_level(cur_c, level);
                if (*ll_cur) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                auto *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx])
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");
                    data[idx] = selectedNeighbors[idx];
                }
                auto *incoming_edges = new std::set<tableint>();
                setIncomingEdgesPtr(cur_c, level, (void *)incoming_edges);
            }

            // go over the selected neighbours - selected[idx] is the neighbour id
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
#ifdef ENABLE_PARALLELIZATION
                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);
#endif
                linklistsizeint *ll_other = get_linklist_at_level(selectedNeighbors[idx], level);
                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                // data is the array of neighbours - for the current neighbour (selected[idx])
                auto *data = (tableint *) (ll_other + 1);

                // If the selected neighbor can add another link (hasn't reached the max) - add it.
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // try finding "weak" elements to replace it with the new one with the heuristic:
                    CandidatesQueue<dist_t> candidates;
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]),
                                                        getDataByInternalId(selectedNeighbors[idx]), dist_func_param_), data[j]);
                    }

                    auto orig_candidates = candidates;
                    // candidates will store the newly selected neighbours (for the current neighbour).
                    // it must be a subset of the old neighbours set + the new node (cur_c)
                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    // check the diff in the link list, save the neighbours
                    // that were chosen to be removed, and update the new neighbours
                    tableint removed_links[sz_link_list_other+1];
                    size_t removed_idx=0;
                    size_t link_idx=0;

                    while (orig_candidates.size() > 0) {
                        if(orig_candidates.top().second !=
                           candidates.top().second) {
                            removed_links[removed_idx++] = orig_candidates.top().second;
                            orig_candidates.pop();
                        } else {
                            data[link_idx++] = candidates.top().second;
                            candidates.pop();
                            orig_candidates.pop();
                        }
                    }
                    setListCount(ll_other, link_idx);
                    if (removed_idx+link_idx != sz_link_list_other+1) {
                        throw std::runtime_error("Error in repairing links");
                    }

                    // remove the current neighbor from the incoming list of nodes for the
                    // neighbours that were chosen to remove (if edge wasn't bidirectional)
                    std::set<tableint>* neighbour_incoming_edges = getIncomingEdgesPtr(selectedNeighbors[idx], level);
                    for (size_t i=0; i<removed_idx; i++) {
                        tableint node_id = removed_links[i];
                        std::set<tableint>* node_incoming_edges = getIncomingEdgesPtr(node_id, level);
                        // if we removed cur_c (the node just inserted), then it points to the current neighbour, but not vise versa.
                        if (node_id == cur_c) {
                            neighbour_incoming_edges->insert(cur_c);
                            continue;
                        }

                        // if the node id (the neighbour's neighbour to be removed)
                        // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
                        // we should remove the current neighbor from the node's incoming edges.
                        // otherwise, the edge turned from bidirectional to
                        // uni-directional, so we insert it to the neighbour's
                        // incoming edges set.
                        if (node_incoming_edges->find(selectedNeighbors[idx]) != node_incoming_edges->end()) {
                            node_incoming_edges->erase(selectedNeighbors[idx]);
                        } else {
                            neighbour_incoming_edges->insert(node_id);
                        }
                    }
                }
            }
            return next_closest_entry_point;
        }

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, (int)new_max_elements);

            element_levels_.resize(new_max_elements);
            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            if (label_lookup_.find(label) == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            label_c = label_lookup_[label];

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            auto* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        unsigned short int getListCount(const linklistsizeint *ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr)) = *((unsigned short int *)&size);
        }

        void repairConnectionsForDeletion(tableint element_internal_id, tableint neighbour_id,
          tableint *neighbours_list, tableint *neighbour_neighbours_list, int level) {

            // put the deleted element's neighbours in the candidates.
            CandidatesQueue<dist_t> candidates;
            std::set<tableint> candidates_set;
            unsigned short neighbours_count = getListCount(neighbours_list);
            tableint *neighbours = (tableint *)(neighbours_list + 1);
            for (size_t j = 0; j < neighbours_count; j++) {
                // Don't put the neighbor itself in his own candidates
                if (neighbours[j] == neighbour_id) {
                    continue;
                }
                candidates.emplace(
                  fstdistfunc_(getDataByInternalId(neighbours[j]), getDataByInternalId(neighbour_id),
                    dist_func_param_), neighbours[j]);
                candidates_set.insert(neighbours[j]);
            }

            // add the deleted element's neighbour's original neighbors in the candidates.
            std::set<tableint> neighbour_orig_neighbours_set;
            unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);
            tableint *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                neighbour_orig_neighbours_set.insert(neighbour_neighbours[j]);
                // Don't add the removed element to the candidates, nor nodes that are already in the candidates set.
                if (candidates_set.find(neighbour_neighbours[j]) != candidates_set.end() ||
                  neighbour_neighbours[j] == element_internal_id) {
                    continue;
                }
                candidates.emplace(fstdistfunc_(getDataByInternalId(neighbour_id),
                  getDataByInternalId(neighbour_neighbours[j]), dist_func_param_),
                  neighbour_neighbours[j]);
            }

            auto orig_candidates = candidates;
            size_t Mcurmax = level ? maxM_ : maxM0_;

            // candidates will store the newly selected neighbours for neighbour_id.
            // it must be a subset of the old neighbours set + the element to removed neighbours set.
            getNeighborsByHeuristic2(candidates, Mcurmax);

            // check the diff in the link list, save the neighbours that were chosen to be removed, and update the new neighbours
            tableint removed_links[neighbour_neighbours_count];
            size_t removed_idx=0;
            size_t link_idx=0;

            while (orig_candidates.size() > 0) {
                if (orig_candidates.top().second != candidates.top().second) {
                    if (neighbour_orig_neighbours_set.find(orig_candidates.top().second)
                    != neighbour_orig_neighbours_set.end()) {
                        removed_links[removed_idx++] = orig_candidates.top().second;
                    }
                    orig_candidates.pop();
                } else {
                    neighbour_neighbours[link_idx++] = candidates.top().second;
                    candidates.pop();
                    orig_candidates.pop();
                }
            }
            setListCount(neighbour_neighbours_list, link_idx);

            // remove neighbour id from the incoming list of nodes for his
            // neighbours that were chosen to remove
            std::set<tableint>* neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);

            for (size_t i=0; i<removed_idx; i++) {
                tableint node_id = removed_links[i];
                std::set<tableint>* node_incoming_edges = getIncomingEdgesPtr(node_id, level);

                // if the node id (the neighbour's neighbour to be removed)
                // wasn't pointing to the neighbour (edge was one directional),
                // we should remove it from the node's incoming edges.
                // otherwise, edge turned from bidirectional to one directional,
                // and it should be saved in the neighbor's incoming edges.
                if(node_incoming_edges->find(neighbour_id) != node_incoming_edges->end()) {
                    node_incoming_edges->erase(neighbour_id);
                } else {
                    neighbour_incoming_edges->insert(node_id);
                }
            }

            // updates for the new edges created
            for (size_t i=0; i<link_idx; i++) {
                tableint node_id = neighbour_neighbours[i];
                if (neighbour_orig_neighbours_set.find(node_id) == neighbour_orig_neighbours_set.end()) {
                    std::set<tableint>* node_incoming_edges = getIncomingEdgesPtr(node_id, level);
                    // if the node has an edge to the neighbour as well, remove it
                    // from the incoming nodes of the neighbour
                    // otherwise, need to update the edge as incoming.
                    linklistsizeint *node_links_list = get_linklist_at_level(node_id, level);
                    unsigned short node_links_size = getListCount(node_links_list);
                    auto *node_links = (tableint *)(node_links_list +1);
                    bool bidirectional_edge = false;
                    for (size_t j=0; j<node_links_size; j++) {
                        if (node_links[j] == neighbour_id) {
                            neighbour_incoming_edges->erase(node_id);
                            bidirectional_edge = true;
                            break;
                        }
                    }
                    if (!bidirectional_edge) {
                        node_incoming_edges->insert(neighbour_id);
                    }
                }
            }
        }

        bool removePoint(const labeltype label) {
            // check that the label actually exists in the graph, and update the number of elements.
            tableint element_internal_id;
            if (label_lookup_.find(label) == label_lookup_.end()) {
                return false;
            }
            // add the element id to the available ids for future reuse.
            element_internal_id = label_lookup_[label];
            cur_element_count--;
            label_lookup_.erase(label);
            available_ids.insert(element_internal_id);

            // go over levels from the top and repair connections
            int element_top_level = element_levels_[element_internal_id];
            for (int level = element_top_level; level >= 0; level--) {
                linklistsizeint *neighbours_list = get_linklist_at_level(element_internal_id, level);
                unsigned short neighbours_count = getListCount(neighbours_list);
                auto *neighbours = (tableint *)(neighbours_list + 1);

                // go over the neighbours that also points back to the removed point and make a local repair.
                for (size_t i = 0; i < neighbours_count; i++) {
                    tableint neighbour_id = neighbours[i];
                    linklistsizeint *neighbour_neighbours_list = get_linklist_at_level(neighbour_id, level);
                    unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);

                    auto *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
                    bool bidirectional_edge = false;
                    for (size_t j = 0; j< neighbour_neighbours_count; j++) {
                        // if the edge is bidirectional, do repair for this neighbor
                        if (neighbour_neighbours[j] == element_internal_id) {
                            bidirectional_edge = true;
                            repairConnectionsForDeletion(element_internal_id, neighbour_id, neighbours_list,
                                                         neighbour_neighbours_list, level);
                            break;
                        }
                    }

                    // if this edge is uni-directional, we should remove the element from the neighbor's incoming edges.
                    if (!bidirectional_edge) {
                        std::set<tableint>* neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
                        neighbour_incoming_edges->erase(element_internal_id);
                    }
                }

                // next, go over the rest of incoming edges (the ones that are not bidirectional) and make repairs.
                std::set<tableint>* incoming_edges = getIncomingEdgesPtr(element_internal_id, level);
                for (auto incoming_edge: *incoming_edges) {
                    linklistsizeint *incoming_node_neighbours_list = get_linklist_at_level(incoming_edge, level);
                    unsigned short incoming_node_neighbours_count = getListCount(incoming_node_neighbours_list);
                    auto *incoming_node_neighbours = (tableint *)(incoming_node_neighbours_list + 1);
                    repairConnectionsForDeletion(element_internal_id, incoming_edge, neighbours_list,
                                                 incoming_node_neighbours_list, level);
                }
                delete incoming_edges;
            }

            // replace the entry point with another one, if we are deleting the current entry point.
            if (element_internal_id==enterpoint_node_) {
                assert(element_top_level == maxlevel_);
                linklistsizeint *top_level_list = get_linklist_at_level(element_internal_id, maxlevel_);
                unsigned short list_len = getListCount(top_level_list);
                while (list_len == 0) {
                    maxlevel_--;
                    if (maxlevel_ < 0) {
                        enterpoint_node_ = -1;
                        break;
                    }
                    top_level_list = get_linklist_at_level(element_internal_id, maxlevel_);
                    list_len = getListCount(top_level_list);
                }
                // set the (arbitrary) first neighbor as the entry point (if there is some element in the index).
                if (enterpoint_node_ >= 0) {
                    enterpoint_node_= ((tableint *)(top_level_list+1))[0];
                }
            }

            if (element_levels_[element_internal_id] > 0) {
                free(linkLists_[element_internal_id]);
            }
            memset(data_level0_memory_ + element_internal_id * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            return true;
        }

        void addPoint(const void *data_point, labeltype label) {

            tableint cur_c = 0;

            // Checking if an element with the given label already exists. if so, return an error.
            if (label_lookup_.find(label) != label_lookup_.end()) {
                throw std::runtime_error("The given label already exit in the index. Consider using updatePoint instead");
            }
            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }
            {
#ifdef ENABLE_PARALLELIZATION
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
#endif
                if (available_ids.empty()) {
                    cur_c = cur_element_count;
                    max_id = cur_element_count;
                } else {
                    cur_c = *available_ids.begin();
                    available_ids.erase(available_ids.begin());
                }
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }
#ifdef ENABLE_PARALLELIZATION
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
#endif
            // choose randomly the maximum level in which the new element will be in the index.
            int element_max_level = getRandomLevel(mult_);
            element_levels_[cur_c] = element_max_level;

#ifdef ENABLE_PARALLELIZATION
            std::unique_lock <std::mutex> entry_point_lock(global);
#endif
            int maxlevelcopy = maxlevel_;

#ifdef ENABLE_PARALLELIZATION
            if (element_max_level <= maxlevelcopy)
                entry_point_lock.unlock();
#endif
            tableint currObj = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabelPtr(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (element_max_level > 0) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * element_max_level + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * element_max_level + 1);
            }

            // this condition only means that we are not inserting the first element.
            if (enterpoint_node_ != -1) {

                if (element_max_level < maxlevelcopy) {
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > element_max_level; level--) {

                        // this is done for the levels which are above the max level
                        // to which we are going to insert the new element. We do
                        // a greedy search in the graph starting from the entry point
                        // at each level, and move on with the closest element we can find.
                        // When there is no improvement to do, we take a step down.
                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
#ifdef ENABLE_PARALLELIZATION
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
#endif
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            auto *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("candidate error: candidate id must be within the range [0,max_elements]");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                for (int level = std::min(element_max_level, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    CandidatesQueue<dist_t> top_candidates = searchBaseLayer(currObj, data_point, level);
                    currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
                }
            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = element_max_level;
            }

            // updating the maximum level (holding a global lock)
            if (element_max_level > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = element_max_level;
                // create the incoming edges set for the new levels.
                for (size_t level_idx = maxlevelcopy+1; level_idx <= element_max_level; level_idx++) {
                    auto* incoming_edges = new std::set<tableint>();
                    setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
                }
            }
        }

        std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0) return result;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    auto *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            CandidatesQueue<dist_t> top_candidates;
            top_candidates=searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k));

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        }

        void checkIntegrity(){
            struct rusage self_ru{};
            getrusage(RUSAGE_SELF, &self_ru);
            std::cerr << "memory usage is : " << self_ru.ru_maxrss << std::endl;
            int connections_checked=0;
            int double_connections=0;
            std::vector <int > inbound_connections_num(max_id,0);
            for(int i = 0;i <= max_id; i++){
                if (available_ids.find(i) != available_ids.end()) {
                    continue;
                }
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    auto *data = (tableint *) (ll_cur + 1);
                    std::set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] >= 0);
                        assert(data[j] <= cur_element_count);
                        assert (data[j] != i);
                        //inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                        // alon: check if this is bi directional
                        linklistsizeint *ll_other = get_linklist_at_level(data[j],l);
                        int size_other = getListCount(ll_other);
                        auto *data_other = (tableint *) (ll_other + 1);
                        for (int r=0; r<size_other; r++) {
                            if (data_other[r] == (tableint)i) {
                                double_connections++;
                                break;
                            }
                        }
                    }
                    assert(s.size() == size);
                }
            }
            std::cout << "connections: " << connections_checked;
            std::cout << " double connections: " << double_connections << std::endl;
            std::cout << "integrity ok\n";
        }
    };

}
