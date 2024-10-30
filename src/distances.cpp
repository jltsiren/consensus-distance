#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <getopt.h>
#include <unistd.h>

#include <omp.h>

#include <consensus_distance/paths_prefix_sum_arrays.h>
#include <gbwtgraph/gbz.h>
#include <gbwtgraph/index.h>
#include <handlegraph/algorithms/dijkstra.hpp>

//------------------------------------------------------------------------------

struct Config {
    constexpr static size_t KMER_LENGTH = gbwtgraph::Key64::KMER_LENGTH;
    constexpr static size_t WINDOW_LENGTH = gbwtgraph::Key64::WINDOW_LENGTH;
    constexpr static size_t READ_LENGTH = 150;
    constexpr static size_t DISTANCE = 1000;
    constexpr static size_t QUERIES = 1000;
    constexpr static size_t MAX_ATTEMPTS = 30;

    std::string base_graph;
    std::string sampled_graph;
    std::string sample_fasta;

    size_t read_length = READ_LENGTH;
    size_t kmer_length = KMER_LENGTH;
    size_t window_length = WINDOW_LENGTH;

    size_t distance = DISTANCE;
    size_t queries = QUERIES;

    int threads;

    // TODO: verbosity? random seed? attempts?

    Config(int argc, char** argv);

    static void usage(int exit_code);
    void print(std::ostream& out) const;

    // Minimizer window length.
    size_t window_bp() const { return this->kmer_length + this->window_length - 1; }

    static int max_threads() { return omp_get_max_threads(); }
};

std::vector<std::string> read_fasta(const std::string& filename);

struct Sampler {
    // offset -> sequence id for the start of each sequence included in the sampler.
    std::map<size_t, size_t> positions;
    size_t size;

    std::mt19937_64 rng;

    // We sample starting positions for windows of the given length.
    Sampler(const std::vector<std::string>& sequences, size_t window_length, std::uint64_t seed = 0x1234567890ABCDEFULL);

    // Returns (sequence id, sequence offset).
    std::pair<size_t, size_t> sample();
};

struct Position {
    gbwt::node_type node;
    size_t node_offset;
    size_t sequence_offset;

    bool empty() const { return this->node == gbwt::ENDMARKER; }
};

std::ostream& operator<<(std::ostream& out, const Position& position);

// Returns an uniquely mapping graph position for a minimizer in sequence[offset, offset + read_length).
// If no such position exists, returns an empty Position.
Position find_position(const gbwtgraph::GBZ& graph, const gbwtgraph::MinimizerIndex<gbwtgraph::Key64, gbwtgraph::Position>& index, const std::string& sequence, size_t offset, size_t read_length);

// Returns the minimum distance between two positions in the graph, or Distances::NOT_REACHED if no path exists.
size_t graph_distance(const gbwtgraph::GBZ& graph, const Position& first, const Position& second);

// Returns the minimum and mean distance over all haplotypes between two positions in the graph.
// If no path exists, returns Distances::NOT_REACHED and -1.0.
std::pair<size_t, double> haplotype_distance(const gbwtgraph::GBZ& graph, const pathsprefixsumarrays::PathsPrefixSumArrays& index, const Position& first, const Position& second);

struct Distances {
    constexpr static size_t NOT_REACHED = std::numeric_limits<size_t>::max();

    std::vector<size_t> truth;

    std::vector<size_t> base_graph_min;
    std::vector<size_t> base_haplotype_min;
    std::vector<double> base_haplotype_mean;

    std::vector<size_t> sampled_graph_min;
    std::vector<size_t> sampled_haplotype_min;
    std::vector<double> sampled_haplotype_mean;

    void full_report(std::ostream& out) const;
    void summary_report(std::ostream& out) const;
};

//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::cerr << "Graph / haplotype distance experiment" << std::endl;
    std::cerr << std::endl;

    // Parse the command line arguments.
    Config config(argc, argv);
    config.print(std::cerr);
    std::cerr << std::endl;

    double start = gbwt::readTimer();

    // Load the sampled graph and build a minimizer index and a distance structure for it.
    std::cerr << "Loading sampled graph " << config.sampled_graph << std::endl;
    gbwtgraph::GBZ sampled_graph;
    sdsl::simple_sds::load_from(sampled_graph, config.sampled_graph);
    std::cerr << "Building a minimizer index for the sampled graph" << std::endl;
    gbwtgraph::MinimizerIndex<gbwtgraph::Key64, gbwtgraph::Position> minimizer_index(config.kmer_length, config.window_length);
    gbwtgraph::index_haplotypes(sampled_graph.graph, minimizer_index);
    std::cerr << "Building a distance structure for the sampled graph" << std::endl;
    pathsprefixsumarrays::PathsPrefixSumArrays sampled_distances(sampled_graph.graph);

    // Load the base graph and build a distance structure for it.
    std::cerr << "Loading base graph " << config.base_graph << std::endl;
    gbwtgraph::GBZ base_graph;
    sdsl::simple_sds::load_from(base_graph, config.base_graph);
    std::cerr << "Building a distance structure for the base graph" << std::endl;
    pathsprefixsumarrays::PathsPrefixSumArrays base_distances(base_graph.graph);

    // Read the sample and build data structures for sampling positions.
    std::cerr << "Reading sample " << config.sample_fasta << std::endl;
    std::vector<std::string> sequences = read_fasta(config.sample_fasta);
    size_t total_length = 0;
    for(const std::string& seq : sequences) { total_length += seq.length(); }
    std::cerr << "The sample consists of " << sequences.size() << " sequences of total length " << total_length << " bp" << std::endl;
    Sampler sampler(sequences, config.distance + config.read_length);
    std::cerr << std::endl;

    // Queries. We may have to abort if kmer length is too short to find unique minimizers.
    std::cerr << "Running " << config.queries << " queries" << std::endl;
    Distances distances;
    for(size_t query = 0; query < config.queries; query++) {
        size_t attempt = 0;
        for(; attempt < Config::MAX_ATTEMPTS; attempt++) {
            // NOTE: We try again if we don't find uniquely mapping minimizers or we don't get all the distances.
            std::pair<size_t, size_t> sample = sampler.sample();
            Position first = find_position(sampled_graph, minimizer_index, sequences[sample.first], sample.second, config.read_length);
            if(first.empty()) { continue; }
            Position second = find_position(sampled_graph, minimizer_index, sequences[sample.first], sample.second + config.distance, config.read_length);
            if(second.empty()) { continue; }
            size_t truth = second.sequence_offset - first.sequence_offset;

            // Base graph.
            size_t base_min = graph_distance(base_graph, first, second);
            if(base_min == Distances::NOT_REACHED) { continue; }
            auto base_haplotype = haplotype_distance(base_graph, base_distances, first, second);
            if(base_haplotype.first == Distances::NOT_REACHED) { continue; }

            // Sampled graph.
            size_t sampled_min = graph_distance(sampled_graph, first, second);
            if(sampled_min == Distances::NOT_REACHED) { continue; }
            auto sampled_haplotype = haplotype_distance(sampled_graph, sampled_distances, first, second);
            if(sampled_haplotype.first == Distances::NOT_REACHED) { continue; }

            // Filter out the outliers. They are likely incorrect mappings, graph issues, or bugs.
            size_t threshold = 10 * truth;
            if(base_min > threshold || sampled_min > threshold) { continue; }
            if(base_haplotype.first > threshold || sampled_haplotype.first > threshold) { continue; }

            // We have all the distances.
            distances.truth.push_back(truth);
            distances.base_graph_min.push_back(base_min);
            distances.base_haplotype_min.push_back(base_haplotype.first);
            distances.base_haplotype_mean.push_back(base_haplotype.second);
            distances.sampled_graph_min.push_back(sampled_min);
            distances.sampled_haplotype_min.push_back(sampled_haplotype.first);
            distances.sampled_haplotype_mean.push_back(sampled_haplotype.second);
            break;
        }
        if(attempt == Config::MAX_ATTEMPTS) {
            std::cerr << "Error: Could not find all distances for query " << query << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Output. Report full results to stdout and summary to stderr.
    distances.full_report(std::cout);
    distances.summary_report(std::cerr);
    std::cerr << std::endl;

    double seconds = gbwt::readTimer() - start;
    std::cerr << "Used " << seconds << " seconds, " << gbwt::inGigabytes(gbwt::memoryUsage()) << " GiB" << std::endl;
    std::cerr << std::endl;

    return 0;
}

//------------------------------------------------------------------------------

Config::Config(int argc, char** argv) {
    if(argc == 1) { Config::usage(EXIT_SUCCESS); }
    this->threads = Config::max_threads();

    // Data for getopt_long().
    int c = 0, option_index = 0;
    constexpr option OPTIONS[] = {
        { "read-length", required_argument, nullptr, 'r' },        
        { "kmer-length", required_argument, nullptr, 'k' },
        { "window-length", required_argument, nullptr, 'w' },
        { "distance", required_argument, nullptr, 'd' },
        { "queries", required_argument, nullptr, 'q' },
        { "threads", required_argument, nullptr, 't' },
        { "help", no_argument, nullptr, 'h' },
        { nullptr, 0, nullptr, 0 }
    };

    // Parse options.
    while((c = getopt_long(argc, argv, "r:k:w:d:q:t:h", OPTIONS, &option_index)) != -1) {
        switch (c) {
            case 'r':
                this->read_length = std::stoul(optarg);
                break;
            case 'k':
                this->kmer_length = std::stoul(optarg);
                break;
            case 'w':
                this->window_length = std::stoul(optarg);
                break;
            case 'd':
                this->distance = std::stoul(optarg);
                break;
            case 'q':
                this->queries = std::stoul(optarg);
                break;

            case 't':
                this->threads = std::stoi(optarg);
                break;
            case 'h':
                Config::usage(EXIT_SUCCESS);
                break;
            default:
                Config::usage(EXIT_FAILURE);
        }
    }

    // Determine file names.
    if(optind + 3 != argc) { Config::usage(EXIT_FAILURE); }
    this->base_graph = argv[optind];
    this->sampled_graph = argv[optind + 1];
    this->sample_fasta = argv[optind + 2];

    // Sanity checks.
    if(this->kmer_length == 0 || this->kmer_length > gbwtgraph::Key64::KMER_MAX_LENGTH) {
        std::cerr << "Error: K-mer length (" << this->kmer_length << ") must be between 1 and " << gbwtgraph::Key64::KMER_MAX_LENGTH << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->window_length == 0) {
        std::cerr << "Error: Window length cannot be 0" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->read_length < this->window_bp()) {
        std::cerr << "Error: Read length (" << this->read_length << ") must be at least as long as a minimizer window (" << this->window_bp() << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->distance < this->read_length) {
        std::cerr << "Error: Distance (" << this->distance << ") between reads must be at least as long as a read (" << this->read_length << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->threads < 1 || this->threads > Config::max_threads()) {
        std::cerr << "Error: Number of threads (" << this->threads << ") must be between 1 and " << Config::max_threads() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Apply the settings.
    omp_set_num_threads(this->threads);
    gbwt::Verbosity::set(gbwt::Verbosity::SILENT);
}

void Config::usage(int exit_code) {
    std::cerr << "Usage: distances [options] base_graph.gbz sampled_graph.gbz sample.fa" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Parameters:" << std::endl;
    std::cerr << "  -r, --read-length N    Read length (default: " << READ_LENGTH << ")" << std::endl;
    std::cerr << "  -k, --kmer-length N    Minimizer k-mer length (default: " << KMER_LENGTH << ")" << std::endl;
    std::cerr << "  -w, --window-length N  Minimizer window length (default: " << WINDOW_LENGTH << ")" << std::endl;
    std::cerr << "  -d, --distance N       Approximate distance in the sample (default: " << DISTANCE << ")" << std::endl;
    std::cerr << "  -q, --queries N        Number of queries (default: " << QUERIES << ")" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Other options:" << std::endl;
    std::cerr << "  -t, --threads          Number of threads" << std::endl;
    std::cerr << "  -h, --help             Print this help message" << std::endl;
    std::cerr << std::endl;
    std::exit(exit_code);
}

void Config::print(std::ostream& out) const {
    out << "Base graph:     " << this->base_graph << std::endl;
    out << "Sampled graph:  " << this->sampled_graph << std::endl;
    out << "Sample:         " << this->sample_fasta << std::endl;
    out << "K-mer length:   " << this->kmer_length << std::endl;
    out << "Window length:  " << this->window_length << std::endl;
    out << "True distance:  " << this->distance << std::endl;
    out << "Queries:        " << this->queries << std::endl;
    out << "Threads:        " << this->threads << std::endl;
}

//------------------------------------------------------------------------------

std::vector<std::string> read_fasta(const std::string& filename) {
    std::vector<std::string> result;

    std::ifstream input(filename, std::ios_base::binary);
    if(!input) {
        std::cerr << "Error: Cannot open FASTA file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    while(std::getline(input, line)) {
        if(line.empty()) { continue; }
        if(line[0] == '>') { result.push_back(std::string()); continue; }
        if(result.empty()) {
            std::cerr << "Error: FASTA file " << filename << " does not start with a header line" << std::endl; std::exit(EXIT_FAILURE);
        }
        result.back().append(line);
    }
    input.close();

    return result;
}

//------------------------------------------------------------------------------

Sampler::Sampler(const std::vector<std::string>& sequences, size_t window_length, std::uint64_t seed) :
    rng(seed)
{
    size_t offset = 0;
    for(size_t i = 0; i < sequences.size(); i++) {
        if(sequences[i].length() < window_length) { continue; }
        this->positions[offset] = i;
        offset += sequences[i].length() - window_length + 1;
    }
    this->size = offset;
    if(this->size == 0) {
        std::cerr << "Error: No sequences long enough for sampling" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::pair<size_t, size_t> Sampler::sample() {
    size_t offset = this->rng() % this->size;
    auto iter = this->positions.upper_bound(offset);
    --iter;
    return std::make_pair(iter->second, offset - iter->first);
}

//------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& out, const Position& position) {
    out << "(node " << position.node << ", offset " << position.node_offset << ", sequence offset " << position.sequence_offset << ")";
    return out;
}


Position find_position(const gbwtgraph::GBZ& graph, const gbwtgraph::MinimizerIndex<gbwtgraph::Key64, gbwtgraph::Position>& index, const std::string& sequence, size_t offset, size_t read_length) {
    // Find the minimizers.
    auto minimizers = index.minimizers(sequence.begin() + offset, sequence.begin() + offset + read_length);
    if(minimizers.empty()) { return { gbwt::ENDMARKER, 0, 0 }; }

    // Find the first uniquely mapping minimizer.
    for(const auto& minimizer : minimizers) {
        std::pair<const gbwtgraph::Position*, size_t> hits = index.find(minimizer);
        if(hits.second != 1) { continue; }
        gbwtgraph::pos_t pos = hits.first->decode();
        if(minimizer.is_reverse) {
            size_t node_length = graph.graph.get_length(graph.graph.get_handle(gbwtgraph::id(pos)));
            pos = gbwtgraph::reverse_base_pos(pos, node_length);
        }
        return { gbwt::Node::encode(gbwtgraph::id(pos), gbwtgraph::is_rev(pos)), gbwtgraph::offset(pos), offset + minimizer.offset };
    }

    return { gbwt::ENDMARKER, 0, 0 };
}

size_t graph_distance(const gbwtgraph::GBZ& graph, const Position& first, const Position& second) {
    size_t result = Distances::NOT_REACHED;
    if(first.empty() || second.empty()) { return result; }

    gbwtgraph::handle_t first_handle = gbwtgraph::GBWTGraph::node_to_handle(first.node);
    gbwtgraph::handle_t second_handle = gbwtgraph::GBWTGraph::node_to_handle(second.node);

    handlegraph::algorithms::dijkstra(&graph.graph, first_handle, [&](const gbwtgraph::handle_t& handle, size_t distance) -> bool {
        if(handle == second_handle) {
            // The reported distance is from the end of the initial node to the start of the current node.
            result = distance + graph.graph.get_length(first_handle) + second.node_offset - first.node_offset;
            return false;
        }
        return true;
    }, false, false, false);

    return result;
}

std::pair<size_t, double> haplotype_distance(const gbwtgraph::GBZ& graph, const pathsprefixsumarrays::PathsPrefixSumArrays& index, const Position& first, const Position& second) {
    std::pair<size_t, double> result = { Distances::NOT_REACHED, -1.0 };
    if(first.empty() || second.empty()) { return result; }

    std::shared_ptr<std::vector<size_t>> distances = index.get_all_nodes_distances(first.node, second.node);
    if(distances->empty()) { return result; }

    // The reported distances are from the end of the initial node to the start of the current node.
    size_t first_node_length = graph.graph.get_length(gbwtgraph::GBWTGraph::node_to_handle(first.node));
    for(size_t& distance : *distances) {
        distance += second.node_offset + first_node_length - first.node_offset;
    }

    result.first = *std::min_element(distances->begin(), distances->end());
    double sum = 0.0;
    for(size_t distance : *distances) { sum += distance; }
    result.second = sum / distances->size();

    return result;
}

//------------------------------------------------------------------------------

void Distances::full_report(std::ostream& out) const {
    for(size_t i = 0; i < this->truth.size(); i++) {
        out << this->truth[i] << '\t';
        out << this->base_graph_min[i] << '\t';
        out << this->base_haplotype_min[i] << '\t';
        out << this->base_haplotype_mean[i] << '\t';
        out << this->sampled_graph_min[i] << '\t';
        out << this->sampled_haplotype_min[i] << '\t';
        out << this->sampled_haplotype_mean[i] << std::endl;
    }
}

void report(std::vector<double>& distances, std::ostream& out, const std::string& name, size_t indent) {
    std::sort(distances.begin(), distances.end());
    double mean = 0.0, variance = 0.0;
    for(double distance : distances) { mean += distance; variance += distance * distance; }
    mean /= distances.size();
    variance = variance / distances.size() - mean * mean;
    double stdev = std::sqrt(variance);
    double median = (distances.size() % 2 == 0 ? (distances[distances.size() / 2 - 1] + distances[distances.size() / 2]) / 2 : distances[distances.size() / 2]);

    std::string gap;
    if(indent > name.length() + 1) { gap = std::string(indent - name.length() - 1, ' '); }
    out << name << ":" << gap << "median " << median << ", mean " << mean << ", stdev " << stdev << " (" << distances.size() << " measurements)" << std::endl;
}

void report_summary(const std::vector<size_t>& distances, std::ostream& out, const std::string& name, size_t indent = 24) {
    std::vector<double> copy;
    for(size_t distance : distances) {
        copy.push_back(static_cast<double>(distance));
    }
    report(copy, out, name, indent);
}

void report_normalized(const std::vector<size_t>& distances, const std::vector<size_t>& truth, std::ostream& out, const std::string& name, size_t indent = 24) {
    std::vector<double> copy;
    for(size_t i = 0; i < distances.size(); i++) {
        copy.push_back(static_cast<double>(distances[i]) / static_cast<double>(truth[i]));
    }
    report(copy, out, name, indent);
}

void report_normalized(const std::vector<double>& distances, const std::vector<size_t>& truth, std::ostream& out, const std::string& name, size_t indent = 24) {
    std::vector<double> copy;
    for(size_t i = 0; i < distances.size(); i++) {
        copy.push_back(distances[i] / static_cast<double>(truth[i]));
    }
    report(copy, out, name, indent);
}

void Distances::summary_report(std::ostream& out) const {
    report_summary(this->truth, out, "True distance");

    report_normalized(this->base_graph_min, this->truth, out, "Base graph min");
    report_normalized(this->base_haplotype_min, this->truth, out, "Base haplotype min");
    report_normalized(this->base_haplotype_mean, this->truth, out, "Base haplotype mean");

    report_normalized(this->sampled_graph_min, this->truth, out, "Sampled graph min");
    report_normalized(this->sampled_haplotype_min, this->truth, out, "Sampled haplotype min");
    report_normalized(this->sampled_haplotype_mean, this->truth, out, "Sampled haplotype mean");
}

//------------------------------------------------------------------------------
