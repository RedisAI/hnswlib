#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <set>

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {

        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t ef = 10, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            has_deletions_=false;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = ef;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);


            // memory looks like this:
            // -----4------ | -----4*M0----------- | ----8*M0------------|-----8-----------------| ---32---- | ----8-- |
            // <links_len>  | <link_1> <link_2>... | <dist_1> <dist_2>...|<incoming_links_array> |   <data>  |  <label>
            // alon: added a dynamic array for the incoming edges which are not outgoing as well.
            size_links_level0_ = maxM0_ * (sizeof(tableint) + sizeof(dist_t)) + sizeof(linklistsizeint)
              + sizeof(void*);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            incoming_links_offset0 = maxM0_ * (sizeof(tableint) + sizeof(dist_t))  + sizeof(linklistsizeint);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");

            // The i-th entry in linkLists array points to max_level[i] (continuous)
            // chunks of memory, each one looks like this:
            // -----4------ | -----4*M-------------- | ------8*M----------- | ----8------------------|
            // <links_len>  | <link_1> <link_2> ...  | <dist_1> <dist_2>... | <incoming_links_array>
            // alon: added a dynamic array for the incoming edges which are not outgoing as well.
            size_links_per_element_ = maxM_ * (sizeof(tableint)+sizeof(dist_t)) + sizeof(linklistsizeint)
              + sizeof(void *);
            incoming_links_offset = maxM_ * (sizeof(tableint)+sizeof(dist_t)) + sizeof(linklistsizeint);

            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };


        ~HierarchicalNSW() {


            for (tableint id = 0; id < max_id; id++) {
                if (available_ids.find(id) != available_ids.end()) {
                    continue;
                }
                for (size_t level = 0; level <= element_levels_[id]; level++) {
                    delete reinterpret_cast<std::set<tableint>*>(*(void**)getIncomingEdgesPtr(id, level));
                }

                if (element_levels_[id] > 0)
                    free(linkLists_[id]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t max_id;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;


        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;
        size_t incoming_links_offset0;
        size_t incoming_links_offset;

        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;
        std::set<tableint> available_ids;

        size_t data_size_;

        bool has_deletions_;


        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
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

        void *getIncomingEdgesPtr(tableint internal_id, int level) {
            if (level == 0) {
                return (void*) (data_level0_memory_ +
                                internal_id * size_data_per_element_ +
                                incoming_links_offset0);
            }
            return (void*) (linkLists_[internal_id] +
            (level - 1) * size_links_per_element_ + incoming_links_offset);
        }
        void setIncomingEdgesPtr(tableint internal_id, int level, void *set_ptr) {
            memcpy(getIncomingEdgesPtr(internal_id, level), &set_ptr, sizeof(void*));
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
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

                        if (!isMarkedDeleted(candidate_id))
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

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

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
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

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

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
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

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                // alon: the distance is saved negatively, so that the queue is ordered
                // so that first is closer (higher).
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                // alon: a candidate is "good" to become a neighbour, unless we find that
                // another item that was already selected to the neighbours set is closer
                // to q than the candidate.
                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
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

        dist_t *get_linkDistList(tableint internal_id, int level) const {
            return (dist_t *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_
                               + maxM_ * sizeof(tableint) + sizeof(linklistsizeint));
        };

        dist_t *get_linkDistList0(tableint internal_id) const {
            return (dist_t *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_
                               + maxM0_ * sizeof(tableint) + sizeof(linklistsizeint));
        };

        dist_t *get_linkDistList_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linkDistList0(internal_id) : get_linkDistList(internal_id, level);
        };

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            std::vector<dist_t> dist_to_selected_neighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                dist_to_selected_neighbors.push_back(top_candidates.top().first);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();
            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);
                dist_t *link_dist_cur = get_linkDistList_at_level(cur_c, level);
                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                    link_dist_cur[idx] = dist_to_selected_neighbors[idx];
                }
                auto *incoming_edges = new std::set<tableint>();
                setIncomingEdgesPtr(cur_c, level, incoming_edges);
            }
            //alon: go over the selected neighbours - selected[idx] is the neighbour id
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                //alon: data is the array of neighbours - for the current neighbour (idx)
                tableint *data = (tableint *) (ll_other + 1);
                dist_t *neighbor_dist = get_linkDistList_at_level(selectedNeighbors[idx], level);

                bool is_cur_c_present = false;
                //alon: it is possible that the "new" node was already the neighbour of idx,
                // only if it is not an actual new node, but an update.
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        neighbor_dist[sz_link_list_other] = dist_to_selected_neighbors[idx];
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        // alon: we already have it  - optimize
                        /*dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);*/
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(dist_to_selected_neighbors[idx], cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            // alon: this has been optimized.
                            candidates.emplace(neighbor_dist[j], data[j]);
                        }
                        auto orig_candidates = candidates;

                        // alon: candidates will store the newly selected neighbours for idx.
                        // it must be a subset of the old neighbours set + the new node (cur_c)
                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        // alon: check the diff in the link list, save the neighbours
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
                                data[link_idx] = candidates.top().second;
                                neighbor_dist[link_idx++] = candidates.top().first;
                                candidates.pop();
                                orig_candidates.pop();
                            }
                        }
                        setListCount(ll_other, link_idx);
                        if (removed_idx+link_idx != sz_link_list_other+1) {
                            throw std::runtime_error("Error in repairing links");
                        }
                        //alon: remove idx from the incoming list of nodes for the
                        // neighbours that were chosen to remove

                        std::set<tableint>* neighbour_incoming_edges =
                          reinterpret_cast<std::set<tableint>*>(*(void**)getIncomingEdgesPtr(selectedNeighbors[idx], level));

                        for (size_t i=0; i<removed_idx; i++) {
                            tableint node_id = removed_links[i];
                            std::set<tableint>* node_incoming_edges =
                              reinterpret_cast<std::set<tableint>*>(*(void**) getIncomingEdgesPtr(
                                node_id, level));
                            // alon: if we removed cur_c (the node just inserted),
                            // then it points to neighbour but not vise versa.
                            if(node_id == cur_c) {
                                neighbour_incoming_edges->insert(cur_c);
                                continue;
                            }

                            // alon: if the node id (the neighbour's neighbour to be removed)
                            // wasn't pointing to the neighbour (i.e., the edge was one directional),
                            // we should remove from the node's incoming edges.
                            // otherwise, the edge turned from bidirectional to
                            // one directional, so we insert it to the neighbour's
                            // incoming edges set.
                            if (node_incoming_edges->find(selectedNeighbors[idx])
                            != node_incoming_edges->end()) {
                                node_incoming_edges->erase(selectedNeighbors[idx]);
                            } else {
                                neighbour_incoming_edges->insert(node_id);
                            }
                        }

                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }




        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
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

            if (has_deletions_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);



            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) malloc(new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            memcpy(data_level0_memory_new, data_level0_memory_,cur_element_count * size_data_per_element_);
            free(data_level0_memory_);
            data_level0_memory_=data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) malloc(sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            memcpy(linkLists_new, linkLists_,cur_element_count * sizeof(void *));
            free(linkLists_);
            linkLists_=linkLists_new;

            max_elements_=new_max_elements;

        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {


            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements=max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);


            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);




            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);


            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);


            visited_list_pool_ = new VisitedListPool(1, max_elements);


            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            has_deletions_=false;

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    has_deletions_=true;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
//        static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            has_deletions_=true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        void repairConnectionsForDeletion(tableint element_internal_id, tableint neighbour_id,
          tableint *neighbours_list, tableint *neighbour_neighbours_list, int level){


            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;

            // alon: put the neighbor's neighbours in the candidates.
            std::set<tableint> neighbour_orig_neighbours_set;
            unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);
            tableint *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
            dist_t *neighbour_neighbours_dist = get_linkDistList_at_level(neighbour_id, level);
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                if (neighbour_neighbours[j] == element_internal_id) {
                    continue;
                }
                neighbour_orig_neighbours_set.insert(neighbour_neighbours[j]);
                candidates.emplace(neighbour_neighbours_dist[j], neighbour_neighbours[j]);
            }

            // alon: put the deleted element's neighbours in the candidates.
            unsigned short neighbours_count = getListCount(neighbours_list);
            tableint *neighbours = (tableint *)(neighbours_list + 1);
            for (size_t j = 0; j < neighbours_count; j++) {
                if (neighbours[j] == neighbour_id ||
                neighbour_orig_neighbours_set.find(neighbours[j]) != neighbour_orig_neighbours_set.end()) {
                    continue;
                }
                candidates.emplace(
                  fstdistfunc_(getDataByInternalId(neighbours[j]), getDataByInternalId(neighbour_id),
                    dist_func_param_), neighbours[j]);
            }

            int total_candidates = candidates.size();
            auto orig_candidates = candidates;
            size_t Mcurmax = level ? maxM_ : maxM0_;

            // alon: candidates will store the newly selected neighbours for neighbour_id.
            // it must be a subset of the old neighbours set + the element to removed neighbours set.
            getNeighborsByHeuristic2(candidates, Mcurmax);

            // alon: check the diff in the link list, save the neighbours
            // that were chosen to be removed, and update the new neighbours
            tableint removed_links[neighbour_neighbours_count];
            size_t removed_idx=0;
            size_t link_idx=0;

            while (orig_candidates.size() > 0) {
                if(orig_candidates.top().second !=
                   candidates.top().second) {
                    if (neighbour_orig_neighbours_set.find(orig_candidates.top().second)
                    != neighbour_orig_neighbours_set.end()) {
                        removed_links[removed_idx++] = orig_candidates.top().second;
                    }
                    orig_candidates.pop();
                } else {
                    neighbour_neighbours[link_idx] = candidates.top().second;
                    neighbour_neighbours_dist[link_idx++] = candidates.top().first;
                    candidates.pop();
                    orig_candidates.pop();
                }
            }
            setListCount(neighbour_neighbours_list, link_idx);

            //alon: remove neighbour id from the incoming list of nodes for his
            // neighbours that were chosen to remove
            std::set<tableint>* neighbour_incoming_edges =
              reinterpret_cast<std::set<tableint>*>(*(void**)getIncomingEdgesPtr(neighbour_id, level));

            for (size_t i=0; i<removed_idx; i++) {
                tableint node_id = removed_links[i];
                std::set<tableint>* node_incoming_edges =
                  reinterpret_cast<std::set<tableint>*>(*(void**) getIncomingEdgesPtr(
                    node_id, level));

                // alon: if the node id (the neighbour's neighbour to be removed)
                // wasn't pointing to the neighbour (edge was one directional),
                // we should remove it from the nodes's incoming edges.
                // otherwise, edge turned from bidirectional to one directional,
                // and it should be saved in the neighbours id incoming edges.
                if(node_incoming_edges->find(neighbour_id) !=
                   node_incoming_edges->end()) {
                    node_incoming_edges->erase(neighbour_id);
                } else {
                    neighbour_incoming_edges->insert(node_id);
                }
            }
            //alon: need updates for the new edges created
            for (size_t i=0; i<link_idx; i++) {
                tableint node_id = neighbour_neighbours[i];
                if (neighbour_orig_neighbours_set.find(node_id) == neighbour_orig_neighbours_set.end()) {
                    std::set<tableint>* node_incoming_edges =
                      reinterpret_cast<std::set<tableint>*>(*(void**) getIncomingEdgesPtr(
                        node_id, level));
                    //alon: if the node has an edge to the neighbour as well, remove it
                    //from the incoming nodes of the neighbour
                    //otherwise, need to update the edge as incoming.
                    linklistsizeint *node_links_list = get_linklist_at_level(node_id, level);
                    unsigned short node_links_size = getListCount(node_links_list);
                    tableint *node_links = (tableint *)(node_links_list +1);
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
            // alon: check that the label actually exist in the graph,
            // and update the number of elements.
            tableint element_internal_id;
            {
                std::unique_lock<std::mutex> templock_curr(
                  cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search == label_lookup_.end()) {
                    return false;
                }
                element_internal_id = label_lookup_[label];
                // alon: add the element id to the available ids for future reuse.
                cur_element_count--;
                label_lookup_.erase(label);
                available_ids.insert(element_internal_id);
            }

            // alon: go over levels from top and repair connections
            int element_top_level = element_levels_[element_internal_id];
            for (int level = element_top_level; level >= 0; level--) {
                linklistsizeint *neighbours_list = get_linklist_at_level(element_internal_id, level);
                unsigned short neighbours_count = getListCount(neighbours_list);
                tableint *neighbours = (tableint *)(neighbours_list + 1);

                // alon: go over the neighbours that also points back to the removed point
                // and make a local repair.
                for (size_t i = 0; i < neighbours_count; i++) {
                    tableint neighbour_id = neighbours[i];
                    linklistsizeint *neighbour_neighbours_list = get_linklist_at_level(neighbour_id, level);
                    unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);
                    tableint *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
                    bool bidirectional_edge = false;
                    for (size_t j = 0; j< neighbour_neighbours_count; j++) {
                        // alon: if the edge is bidirectional, do repair
                        if (neighbour_neighbours[j] == element_internal_id) {
                            bidirectional_edge = true;
                            repairConnectionsForDeletion(element_internal_id,
                              neighbour_id, neighbours_list, neighbour_neighbours_list, level);
                            break;
                        }
                    }
                    if (!bidirectional_edge) {
                        std::set<tableint>* neighbour_incoming_edges =
                          reinterpret_cast<std::set<tableint>*>(*(void**)getIncomingEdgesPtr(neighbour_id, level));
                        neighbour_incoming_edges->erase(element_internal_id);
                    }
                }
                std::set<tableint>* incoming_edges =
                  reinterpret_cast<std::set<tableint>*>(*(void**)getIncomingEdgesPtr(element_internal_id, level));
                for (auto incoming_edge: *incoming_edges) {
                    linklistsizeint *incoming_node_neighbours_list = get_linklist_at_level(incoming_edge, level);
                    unsigned short incoming_node_neighbours_count = getListCount(incoming_node_neighbours_list);
                    tableint *incoming_node_neighbours = (tableint *)(incoming_node_neighbours_list + 1);
                    repairConnectionsForDeletion(element_internal_id, incoming_edge,
                      neighbours_list, incoming_node_neighbours_list, level);
                }
                delete incoming_edges;
            }
            if (element_internal_id==enterpoint_node_) {
                linklistsizeint *top_level_list = get_linklist_at_level(element_internal_id, element_top_level);
                unsigned short list_len = getListCount(top_level_list);
                if (list_len > 0) {
                    enterpoint_node_= ((tableint *)(top_level_list+1))[0];
                } else {
                    top_level_list = get_linklist_at_level(element_internal_id, element_levels_[element_internal_id]-1);
                    list_len = getListCount(top_level_list);
                    if(list_len>0) {
                        enterpoint_node_= ((tableint *)(top_level_list+1))[0];
                        maxlevel_ = element_top_level-1;
                    } else {
                        throw std::runtime_error("error when deleting entry point");
                    }
                }
            }
            if (element_levels_[element_internal_id] > 0) {
                free(linkLists_[element_internal_id]);
            }
            memset(data_level0_memory_ + element_internal_id * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            return true;
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
//                    if (neigh == internalId)
//                        continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                if (available_ids.size() == 0) {
                    cur_c = cur_element_count;
                    max_id = cur_element_count;
                } else {
                    cur_c = *available_ids.begin();
                    available_ids.erase(available_ids.begin());
                }

                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = level;
            // alon: At first, level equals -1, so we use the random level we got to begin with.
            // (not clear why this condition exist)
            if (level == -1)
                curlevel = getRandomLevel(mult_);

            // alon: Update the top level in graph that contains the element.
            element_levels_[cur_c] = curlevel;

            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            // alon: this condition only means that we are not inserting the first element.
            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {

                        // alon: this is done for the levels which are above the max level
                        // to which we are going to insert the new element. We do
                        // a greedy search in the graph starting from the entry point
                        // at each level, and move on with the closest element we can find.
                        // When there is no improvment to do, we take a step down.
                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
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

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
                //alon: create the incoming edges set for the new levels.
                for(size_t level_idx = maxlevelcopy+1; level_idx <= curlevel; level_idx++) {
                    auto* incoming_edges = new std::set<tableint>();
                    setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
                }
            }
            return cur_c;
        };

        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype >> result;
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

                    tableint *datal = (tableint *) (data + 1);
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

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            if (has_deletions_) {
                top_candidates=searchBaseLayerST<true,true>(
                        currObj, query_data, std::max(ef_, k));
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(
                        currObj, query_data, std::max(ef_, k));
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        void checkIntegrity(){
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
                    tableint *data = (tableint *) (ll_cur + 1);
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
                        tableint *data_other = (tableint *) (ll_other + 1);
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
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i <= max_id; i++){
                    if (available_ids.find(i) != available_ids.end()) {
                        continue;
                    }
                    //assert(inbound_connections_num[i] > 0);
                    //min1=std::min(inbound_connections_num[i],min1);
                    //max1=std::max(inbound_connections_num[i],max1);
                }
                //std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

    };

}
