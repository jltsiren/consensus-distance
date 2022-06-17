#include <iostream>
#include <iterator>
#include <map>
#include <vector>
#include "gbwtgraph/gbwtgraph.h"
#include "gbwtgraph/gfa.h"
#include "gbwt/gbwt.h"
#include "include/consensus_distance/paths_prefix_sum_arrays.h"
using namespace gbwtgraph;


std::vector<path_handle_t>* get_graph_path_handles(GBWTGraph &g){
    std::vector<path_handle_t> *path_handles = new std::vector<path_handle_t>();
    g.for_each_path_handle([&](const path_handle_t path_handle) {
        (*path_handles).push_back(path_handle);
    }); // end of lambda expression)
    return path_handles;
}


std::map<path_handle_t , std::vector<std::pair<handle_t , int>>*>* get_prefix_sum_array(GBWTGraph gbwtGraph){
    std::map<path_handle_t , std::vector<std::pair<handle_t , int>>*> *paths_steps_length = new std::map<path_handle_t , std::vector<std::pair<handle_t , int>>*>();

    //VA ELIMINATA LA MEMORIA QUA dopo che lo usi
    auto path_handles = get_graph_path_handles(gbwtGraph);

    //OCHO ALLA MEMORIA QUA
    for (auto path_handle : (*path_handles)) {
        (*paths_steps_length).insert({path_handle, new std::vector<std::pair<handle_t , int>>()});

        gbwtGraph.for_each_step_in_path(path_handle,[&](const step_handle_t step_handle) {
            auto handle = gbwtGraph.get_handle_of_step(step_handle);

            (*paths_steps_length)[path_handle]->push_back(std::pair<handle_t , int>(handle, gbwtGraph.get_length(handle)));
        }); // end of lambda expression));
    }



    return paths_steps_length;
}




int main() {
    std::string s= "c";
    std::cout << "Hello, World!" << sizeof(s) << std::endl;



    //auto gfa_parse = gfa_to_gbwt("/home/andrea/vg/test/tiny/tiny.gfa");
    auto gfa_parse = gfa_to_gbwt("/Users/gi-loaner-05/tesi/vg/test/tiny/tiny.gfa");

    const gbwt::GBWT& index = *(gfa_parse.first);
    GBWTGraph graph(*(gfa_parse.first), *(gfa_parse.second));
    PathsPrefixSumArrays *a = new PathsPrefixSumArrays(graph);
    std::cout << a->toString();
    std::cout << a->get_prefsum_of_path(graph.get_path_handle("x")).at(0).second;

    /*
    auto c =get_prefix_sum_array(graph);



    // Create a map iterator and point to beginning of map
    auto iterator = (*c).begin();

    // Iterate over the map using Iterator till end.
    while (iterator != (*c).end())
    {
        // Accessing KEY from element pointed by it.
        std::string path_name = graph.get_path_name(iterator->first);
        // Accessing VALUE from element pointed by it.
        std::vector<std::pair<handle_t ,int>> nodes = *(iterator->second);
        std::cout << path_name << " :: ";
        for(int i=0; i< nodes.size(); ++i){
            std::cout<< " (id: " << graph.get_id(nodes[i].first) << ", " << nodes[i].second << ") ";
        }


        std::cout << std::endl;
        // Increment the Iterator to point to next entry
        iterator++;
    }
*/

    return 0;
}