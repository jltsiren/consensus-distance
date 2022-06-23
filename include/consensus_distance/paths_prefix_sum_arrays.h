//
// Created by GI-Loaner-05 on 6/15/22.
//
// TODO: implement a method that take the path_handle and return directly its prefix sum array.
// TODO: implement a method "build_prexif_sum_arrays" (we can discuss about the name)
//  to get the constructor nice and clean (remember memory leaks).
// TODO: implement anothe to_stirng that takes as a parameter a single path_handle, if you want to see just one

#ifndef CONSENSUS_DISTANCE_PREFIX_SUM_ARRAY_H
#define CONSENSUS_DISTANCE_PREFIX_SUM_ARRAY_H

// Standard
#include <iostream>
#include <iterator>
#include <map>
#include <vector>

// GBWT
#include "gbwt/gbwt.h"

// GBWTGraph
#include "gbwtgraph/gbwtgraph.h"
#include "gbwtgraph/gfa.h"
#include "gbwtgraph/gbz.h"

// HandleGraph
#include <handlegraph/handle_graph.hpp>
#include <handlegraph/mutable_handle_graph.hpp>
#include <handlegraph/mutable_path_deletable_handle_graph.hpp>

using namespace gbwtgraph;

/**
 * This class represent the paths' prefix sum arrays.
 */
class PathsPrefixSumArrays{

private:
    // Every vector has as a key its path_handle_t, each entry in the vector is stored with the related handle_t
    std::map<path_handle_t , std::vector<std::pair<handle_t , int>>*> *prefix_sum_arrays;

    std::map<gbwt::size_type , sdsl::sd_vector<>*> *psa;


    /**
     * Get the all the path handles in the graph, which is a reference to a path (opaque 64-bit identifier).
     * @param g the graph on which we are getting the path_handles.
     * @return a pointer to vector of path_handle_t type.
     */
    std::vector<path_handle_t>* get_graph_path_handles(GBWTGraph &g);
    /**
     * Get all the paths from a GBWTGraph.
     * @param gbwtGraph
     * @return a map of vectors (path_handle_t as key). Each vector, in each entru has the length of the node and the
     * relative handle_t.
     */
    std::map<path_handle_t , std::vector<std::pair<handle_t , int>>*>* get_paths(GBWTGraph gbwtGraph);


    /**
     * Compute distance between two node position, knowing that the pos_node_1 is less than pos_node_2.
     * @param pos_node_1
     * @param pos_node_2
     * @param sdb_sel is the select operation.
     * @return the distance betweem the two nodes.
     */
    size_t compute_node_distance(size_t pos_node_1, size_t pos_node_2, sdsl::sd_vector<>::select_1_type &sdb_sel);


public:
    /**
     * Default constructor
     */
    PathsPrefixSumArrays();

    /**
     * Constructor
     * @param gbwtGraph
     */
    PathsPrefixSumArrays(GBWTGraph gbwtGraph);

    /**
     * Get the prefix sum arrays.
     * @return prefix sum arrays in a map
     */
    const std::map<path_handle_t , std::vector<std::pair<handle_t , int>>*>* get_prefix_sum_arrays() const;

    /**
     * Destructor
     */
    ~PathsPrefixSumArrays();

    /**
     * Get a string with all the prefix sum arrays.
     * @return a string containing the prefix sum arrays.
     */
    std::string toString();

    std::string print_prefsum_of_path(path_handle_t path_handle);

    std::vector<std::pair<handle_t , int>> get_prefsum_of_path(path_handle_t path_handle);

    /**
     * Given the path_id and two position node compute the distance between the nodes inside the path.
     * @param pos_node_1
     * @param pos_node_2
     * @param path_id
     * @return the distance.
     */
    size_t get_distance_in_a_path(size_t pos_node_1, size_t pos_node_2, size_t path_id);


};

#endif //CONSENSUS_DISTANCE_PREFIX_SUM_ARRAY_H