/**
 * Authors:
 *  - Andrea Mariotti
 *  - Davide Piovani
 */

#ifndef CONSENSUS_DISTANCE_PREFIX_SUM_ARRAY_H
#define CONSENSUS_DISTANCE_PREFIX_SUM_ARRAY_H



// Standard
#include <iostream>
#include <iterator>
#include <map>
#include <vector>

// GBWT
#include <gbwt/gbwt.h>
#include <gbwt/fast_locate.h>


// GBWTGraph
#include <gbwtgraph/gbwtgraph.h>
#include <gbwtgraph/gfa.h>
#include <gbwtgraph/gbz.h>

// HandleGraph
#include <handlegraph/handle_graph.hpp>
#include <handlegraph/mutable_handle_graph.hpp>
#include <handlegraph/mutable_path_deletable_handle_graph.hpp>

//TEST
#include <gtest/gtest.h>
/**
 * This class represent the paths' prefix sum arrays.
 */

namespace pathsprefixsumarrays{

// Exceptions
class OutOfBoundsPositionInPathException : std::out_of_range {

public:
    //TODO: test that has to be called
    OutOfBoundsPositionInPathException(const std::string &arg) : out_of_range(arg) {}

    const char * what() {
        return std::out_of_range::what();
    }
};

class PathNotInGraphException : std::invalid_argument {
public:

    PathNotInGraphException(const std::string &arg) : std::invalid_argument(arg) {}

    const char * what() {
        return std::invalid_argument::what();
    }
};


class NotExistentDistanceException : std::invalid_argument {
public:

    NotExistentDistanceException(const std::string &arg) : std::invalid_argument(arg) {}

    const char * what() {
        return std::invalid_argument::what();
    }
};


// Class
class PathsPrefixSumArrays {

private:
    friend class PrefixSumArraysTest;
    FRIEND_TEST(PrefixSumArraysTest, get_distance_between_positions_in_path);
    FRIEND_TEST(PrefixSumArraysTest, get_all_nodes_distances);
    FRIEND_TEST(PrefixSumArraysTest, get_positions_of_a_node_in_path);
    FRIEND_TEST(PrefixSumArraysTest, get_all_node_distance_in_a_path_vector);


   // std::map<gbwt::size_type, sdsl::sd_vector<> *>* psa; // Prefix sum arrays (seq_id, prefix sum array)

    std::shared_ptr<gbwt::FastLocate> fast_locate; // It is needed to perform select operation on sd_vector

    std::vector<std::shared_ptr<sdsl::sd_vector<>>> prefix_sum_arrays; // new data structure


    /**
     * Compute distance between two node position in a path, it's an auxiliar method used by the public
     * get_distance_between_positions_in_path.
     * @param pos_node_1
     * @param pos_node_2
     * @param sdb_sel is the select operation.
     * @return the distance between the two positions.
     */
    size_t get_distance_between_positions_in_path_aux(size_t pos_node_1, size_t pos_node_2,
                                                      sdsl::sd_vector<>::select_1_type &sdb_sel) const;

    /**
     * Get all the positions of a node in a path.
     * @param path_id it's the sequence id (not GBWTGraph representation).
     * @param node
     * @param ones the number of ones inside the sd_vector prefix sum array representation. It is needed to compute the
     * positions.
     * @return a non ordered vector of the positions of a node in the path.
     */
    std::shared_ptr<std::vector<size_t>> get_positions_of_a_node_in_path(size_t path_id, gbwt::node_type node, size_t &ones) const;

    /**
     * Get all distances between two nodes in a path. Each nodes can occur several time in a path in different positions.
     *
     * This function calls get_distance_between_positions_in_path which raises NotExistentDistanceException if the two
     * positions in input are the same. This function shouldn't call that function with two equal positions.
     *
     * @param node_1_positions positions of node_1 inside the path (sequence) forward or reverse.
     * @param node_2_positions positions of node_1 inside the path (sequence) forward or reverse.
     * @param path_id
     * @throws OutOfBoundsPositionInPathException when in the parameters there is at least one positions that doesn't
     * exist in the path_id (in reverse or in forward direction).
     * @return a vector of all the distances between the two nodes in a path. The returned vector is empty if one of the
     * two vectors in input is empty or if the path_id doesn't exist. If the two vectors are equals it returns all the
     * distances between the not equal positions.
     */
    std::shared_ptr<std::vector<size_t>> get_all_nodes_distances_in_path( std::shared_ptr<std::vector<size_t>> node_1_positions,
                                                          std::shared_ptr<std::vector<size_t>> node_2_positions,
                                                          size_t path_id) const;

public:

    /**
     * Default constructor.
     */
    PathsPrefixSumArrays();

    /**
     * Constructor.
     * @param gbwtGraph
     */
    PathsPrefixSumArrays(gbwtgraph::GBWTGraph &gbwtGraph);


    /**
     * Destructor.
     */
    ~PathsPrefixSumArrays();

    /**
     * This method is used to clear all the memory of the object.
     */
    void clear();

    /**
     * Get a string with all the prefix sum arrays as sd_vectors representation (0,1).
     * @return a string containing the prefix sum arrays.
     */
    std::string toString_sd_vectors() const;


    /**
     * Get a string with all the prefix sum arrays as arrays of integers.
     * @return a string representing the prefix sum arrays.
     */
    std::string toString() const;


    /**
     * Get fast locate used in Test.
     * @return fast_locate.
     */
    std::shared_ptr<const gbwt::FastLocate> get_fast_locate() const;


    /**
     * Get prefix sum arrays.
     * @return a map in which for each path we have the prefix sum array.
     */
    const std::map<gbwt::size_type, std::shared_ptr<const sdsl::sd_vector<>>>* get_prefix_sum_arrays_map() const;


    /**
     * Given the path_id and two position node inside that path, compute the distance between the positions inside the
     * path.
     * @param pos_node_1
     * @param pos_node_2
     * @param path_id
     * @throws NotExistentDistanceException if the two positions in input are equal. If that is the case, is raised before
     * all the other exceptions.
     * @throws PathNotInGraphException if the path_id doesn't exist in the graph (in reverse or in forward direction).
     * @throws OutOfBoundsPositionInPathException if at least one of the two positions is not in the path.
     * @return the distance. If the two positions are consecutive returns 0.
     */
    size_t get_distance_between_positions_in_path(size_t pos_node_1, size_t pos_node_2, size_t path_id) const;


    /**
     * Get all the distances between two nodes in a path, also takes into account multiple occurences of the same node
     * in a looping path.
     * @param node_1 id of the node.
     * @param node_2 id of the node.
     * @param path_id id of the path (sequence).
     * @throws PathNotInGraphException if the path_id doesn't exist in the graph (in reverse or in forward direction).
     * @return a vector of size_t distances.
     */
    std::shared_ptr<std::vector<size_t>> get_all_nodes_distances_in_path(gbwt::node_type node_1, gbwt::node_type node_2, size_t path_id) const;


    /**
     * Get all node distance between two nodes.
     *
     * This function calls get_all_nodes_distances_in_path which can raise PathNotInGraphException if the path_id in input
     * doesn't exist in the graph. It shouldn't happen but it something to keep in mind.
     *
     * @param node_1
     * @param node_2
     * @return a vector with all the distance between two nodes. If the two nodes haven't any path in common can return
     * an empty vector.
     */
    std::shared_ptr<std::vector<size_t>> get_all_nodes_distances(gbwt::node_type node_1, gbwt::node_type node_2) const;


    /**
     * Get all node positions in every path that visits the node. both forward and reverse.
     * @param node
     * @return a map where the key is the path id (sequence id) and the value is a pointer to a vector of positions in
     * that path.
     */
    std::shared_ptr<std::map<size_t,std::shared_ptr<std::vector<size_t>>>> get_all_node_positions(gbwt::node_type node) const;


    /**
     * Get the prefix sum array related to the path.
     * @param path_id
     * @return prefix sum array as sdsl::sd_vector type.
     */
    std::shared_ptr<const sdsl::sd_vector<>> get_prefix_sum_array_of_path(size_t path_id) const;
};
}// end of namespace pathsprefixsumarrays
#endif //CONSENSUS_DISTANCE_PREFIX_SUM_ARRAY_H