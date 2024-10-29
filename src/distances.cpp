#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <getopt.h>
#include <unistd.h>

#include <omp.h>

#include <consensus_distance/paths_prefix_sum_arrays.h>

#include <gbwtgraph/gbz.h>
#include <gbwtgraph/index.h>

//------------------------------------------------------------------------------

struct Config
{
    constexpr static size_t KMER_LENGTH = gbwtgraph::Key64::KMER_LENGTH;
    constexpr static size_t WINDOW_LENGTH = gbwtgraph::Key64::WINDOW_LENGTH;
    constexpr static size_t READ_LENGTH = 150;
    constexpr static size_t DISTANCE = 1000;
    constexpr static size_t QUERIES = 1000;

    std::string base_graph;
    std::string sampled_graph;
    std::string sample_fasta;

    size_t read_length = READ_LENGTH;
    size_t kmer_length = KMER_LENGTH;
    size_t window_length = WINDOW_LENGTH;

    size_t distance = DISTANCE;
    size_t queries = QUERIES;

    int threads;

    // TODO: verbosity?

    Config(int argc, char** argv);

    static void usage(int exit_code);
    void print(std::ostream& out) const;

    // Minimizer window length.
    size_t window_bp() const { return this->kmer_length + this->window_length - 1; }

    static int max_threads() { return omp_get_max_threads(); }
};

std::vector<std::string> read_fasta(const std::string& filename);

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
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
    std::vector<std::string> sample = read_fasta(config.sample_fasta);
    size_t total_length = 0;
    for(const std::string& seq : sample) { total_length += seq.length(); }
    std::cerr << "The sample consists of " << sample.size() << " sequences of total length " << total_length << " bp" << std::endl;
    // TODO: Implement the sampling.

    // Queries. Note that we may have to abort if kmer length is too short and we cannot find uniquely mapping kmers from both reads.

    // Output: individual results to cout + summary to cerr.

    double seconds = gbwt::readTimer() - start;
    std::cerr << std::endl;
    std::cerr << "Used " << seconds << " seconds, " << gbwt::inGigabytes(gbwt::memoryUsage()) << " GiB" << std::endl;
    std::cerr << std::endl;

    return 0;
}

//------------------------------------------------------------------------------

Config::Config(int argc, char** argv)
{
    if(argc == 1) { Config::usage(EXIT_SUCCESS); }
    this->threads = Config::max_threads();

    // Data for getopt_long().
    int c = 0, option_index = 0;
    constexpr option OPTIONS[] =
    {
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
    while((c = getopt_long(argc, argv, "r:k:w:d:q:t:h", OPTIONS, &option_index)) != -1)
    {
        switch (c)
        {
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
    if(this->kmer_length == 0 || this->kmer_length > gbwtgraph::Key64::KMER_MAX_LENGTH)
    {
        std::cerr << "Error: K-mer length (" << this->kmer_length << ") must be between 1 and " << gbwtgraph::Key64::KMER_MAX_LENGTH << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->window_length == 0)
    {
        std::cerr << "Error: Window length cannot be 0" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->read_length < this->window_bp())
    {
        std::cerr << "Error: Read length (" << this->read_length << ") must be at least as long as a minimizer window (" << this->window_bp() << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->distance < this->read_length)
    {
        std::cerr << "Error: Distance (" << this->distance << ") between reads must be at least as long as a read (" << this->read_length << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(this->threads < 1 || this->threads > Config::max_threads())
    {
        std::cerr << "Error: Number of threads (" << this->threads << ") must be between 1 and " << Config::max_threads() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Apply the settings.
    omp_set_num_threads(this->threads);
    gbwt::Verbosity::set(gbwt::Verbosity::SILENT);
}

void Config::usage(int exit_code)
{
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

void Config::print(std::ostream& out) const
{
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

std::vector<std::string> read_fasta(const std::string& filename)
{
    std::vector<std::string> result;

    std::ifstream input(filename, std::ios_base::binary);
    if(!input)
    {
        std::cerr << "Error: Cannot open FASTA file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    while(std::getline(input, line))
    {
        if(line.empty()) { continue; }
        if(line[0] == '>') { result.push_back(std::string()); continue; }
        if(result.empty())
        {
            std::cerr << "Error: FASTA file " << filename << " does not start with a header line" << std::endl; std::exit(EXIT_FAILURE);
        }
        result.back().append(line);
    }
    input.close();

    return result;
}

//------------------------------------------------------------------------------
