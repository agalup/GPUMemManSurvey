#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>

#include "CSR.h"
#include "dCSR.h"
#include "COO.h"
#include "device/dynamicGraph_impl.cuh"
#include "device/EdgeUpdate.cuh"
#include "device/EdgeInsertion.cuh"
#include "device/EdgeDeletion.cuh"
#include "Definitions.h"
#include "Verification.h"

// Json Reader
#include "json.h"
using json = nlohmann::json;

// ########################
#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
using MemoryManager = MemoryManagerCUDA;
const std::string mem_name("CUDA");
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
using MemoryManager = MemoryManagerHalloc;
const std::string mem_name("HALLOC");
#elif TEST_XMALLOC
#include "xmalloc/Instance.cuh"
using MemoryManager = MemoryManagerXMalloc;
const std::string mem_name("XMALLOC");
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
using MemoryManager = MemoryManagerScatterAlloc;
const std::string mem_name("ScatterAlloc");
#elif TEST_FDG
#include "fdg/Instance.cuh"
using MemoryManager = MemoryManagerFDG;
const std::string mem_name("FDGMalloc");
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
	#ifdef TEST_PAGES
	#ifdef TEST_VIRTUALIZED_ARRAY
	using MemoryManager = MemoryManagerOuroboros<OuroVAPQ>;
	const std::string mem_name("Ouroboros-P-VA");
	#elif TEST_VIRTUALIZED_LIST
	using MemoryManager = MemoryManagerOuroboros<OuroVLPQ>;
	const std::string mem_name("Ouroboros-P-VL");
	#else
	using MemoryManager = MemoryManagerOuroboros<OuroPQ>;
	const std::string mem_name("Ouroboros-P-S");
	#endif
	#endif
	#ifdef TEST_CHUNKS
	#ifdef TEST_VIRTUALIZED_ARRAY
	using MemoryManager = MemoryManagerOuroboros<OuroVACQ>;
	const std::string mem_name("Ouroboros-C-VA");
	#elif TEST_VIRTUALIZED_LIST
	using MemoryManager = MemoryManagerOuroboros<OuroVLCQ>;
	const std::string mem_name("Ouroboros-C-VL");
	#else
	using MemoryManager = MemoryManagerOuroboros<OuroCQ>;
	const std::string mem_name("Ouroboros-C-S");
	#endif
	#endif
#elif TEST_REGEFF
#include "regeff/Instance.cuh"
	#ifdef TEST_ATOMIC
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::AtomicMalloc>;
	const std::string mem_name("RegEff-A");
	#elif TEST_ATOMIC_WRAP
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::AWMalloc>;
	const std::string mem_name("RegEff-AW");
	#elif TEST_CIRCULAR
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CMalloc>;
	const std::string mem_name("RegEff-C");
	#elif TEST_CIRCULAR_FUSED
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CFMalloc>;
	const std::string mem_name("RegEff-CF");
	#elif TEST_CIRCULAR_MULTI
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CMMalloc>;
	const std::string mem_name("RegEff-CM");
	#elif TEST_CIRCULAR_FUSED_MULTI
	using MemoryManager = MemoryManagerRegEff<RegEffVariants::CFMMalloc>;
	const std::string mem_name("RegEff-CFM");
	#endif
#endif

using DataType = float;
template <typename T>
std::string typeext();

template <>
std::string typeext<float>()
{
	return std::string("");
}

template <>
std::string typeext<double>()
{
	return std::string("d_");
}

template <typename MemoryManagerType, typename DataType>
void testrun(CSR<DataType>& input_graph, const json& config, std::ofstream& results);

int main(int argc, char* argv[])
{
    if (argc == 1)
	{
		std::cout << "Require config file as first argument" << std::endl;
		return -1;
    }

    std::string config_file;
    unsigned int testcase_index{0xFFFFFFFFU};
    if(argc >= 2)
	{
        config_file = std::string(argv[1]);
        if(argc >= 3)
        {
            testcase_index = atoi(argv[2]);
        }
    }

    printf("%sDynamic Graph Example\n%s", CLHighlight::break_line_blue_s, CLHighlight::break_line_blue_e);

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d__%H-%M-%S");
    auto time_string = oss.str();

    // Parse config
	std::ifstream json_input(config_file.c_str());
	json config;
    json_input >> config;

    const auto device{config.find("device").value().get<int>()};
	cudaSetDevice(device);
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const auto write_csv{config.find("write_csv").value().get<bool>()};
    const auto csv_filename{config.find("csv_filename").value().get<std::string>()};
    std::ofstream results;
    if(write_csv)
	{
        results.open((csv_filename + prop.name + "---" + time_string + ".csv").c_str(), std::ios_base::app);
    }

    auto graphs = *config.find("graphs");
    int index{0};
    for(auto const& elem : graphs)
	{
        // If we get passed just a single testcase, only execute this specific graph
        if(index++ != testcase_index && testcase_index != 0xFFFFFFFFU)
            continue;
        
        const auto path{config.find("path").value().get<std::string>()};
        std::string filename = path + elem.find("filename").value().get<std::string>();
        CSR<DataType> csr_mat;
        std::string csr_name = filename + typeext<DataType>() + ".csr";
        try
		{
			std::cout << "trying to load csr file \"" << csr_name << "\"\n";
			csr_mat = loadCSR<DataType>(csr_name.c_str());
			std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
			try
			{
				std::cout << "trying to load mtx file \"" << filename << "\"\n";
				auto coo_mat = loadMTX<DataType>(filename.c_str());
                convert(csr_mat, coo_mat);
                csr_mat.filename = csr_name;
				std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
				return -1;
			}
			try
			{
				std::cout << "write csr file for future use\n";
				storeCSR(csr_mat, csr_name.c_str());
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
			}
        }
        std::cout << "Testing: " << elem.find("filename").value().get<std::string>();
        std::cout << " with " << csr_mat.rows << " vertices and " << csr_mat.nnz << " edges\n";

        // #######################################################
        // #######################################################
        // #######################################################
        // Testcase
        // #######################################################
        // #######################################################
        // #######################################################
        std::cout << "--- " << mem_name << "---\n";
        
        testrun<MemoryManager, DataType>(csr_mat, config, results);
    }
    
    return 0;
}

template <typename MemoryManagerType, typename DataType>
void testrun(CSR<DataType>& input_graph, const json& config, std::ofstream& results)
{
    // Parameters
	const auto iterations{config.find("iterations").value().get<int>()};
	const auto update_iterations{config.find("update_iterations").value().get<int>()};
	const auto batch_size{config.find("batch_size").value().get<int>()};
	const auto realistic_deletion{config.find("realistic_deletion").value().get<bool>()};
    const auto verify_enabled{ config.find("verify").value().get<bool>() };
    const auto allocMB{ config.find("manageable_memory_mb").value().get<size_t>() }  ;
    size_t allocationSize{allocMB * 1024ULL * 1024ULL};
    const auto range{config.find("range").value().get<unsigned int>()};
    unsigned int offset{0};
    const auto write_csv{config.find("write_csv").value().get<bool>()};
    const bool printResults{true};

    PerfMeasure init_measure, insert_measure, delete_measure;
    
    // Instantiate graph
    DynGraph<VertexData, EdgeData, MemoryManagerType> dynamic_graph(allocationSize);

    for(auto round = 0; round < iterations; ++round)
    {
        if(printDebugMessages)
            std::cout << "Round: " << round + 1 << " / " << iterations << std::endl;

        Verification<DataType> verification(input_graph);

        // Initialization
        dynamic_graph.init(input_graph);

        if (verify_enabled)
        {
            // Test integrity
			CSR<DataType> test_graph;
            dynamic_graph.dynGraphToCSR(test_graph);
            std::string header = std::string("Initialization - ") + std::to_string(round);
			verification.verify(test_graph, header.c_str(), OutputCodes::VERIFY_INITIALIZATION);
        }

        for(auto update_round = 0; update_round < update_iterations; ++update_round, offset += range)
        {
            EdgeUpdateBatch<VertexData, EdgeData> insertion_updates(dynamic_graph.number_vertices);
            insertion_updates.generateEdgeUpdates(dynamic_graph.number_vertices, batch_size, (round * update_iterations) + update_round, range, offset);
            dynamic_graph.edgeInsertion(insertion_updates);

            if (verify_enabled)
			{
                CSR<DataType> test_graph;
				dynamic_graph.dynGraphToCSR(test_graph);
                verification.hostEdgeInsertion(insertion_updates);
                std::string header = std::string("Insertion - ") + std::to_string(update_round);
				verification.verify(test_graph, header.c_str(), OutputCodes::VERIFY_INSERTION);
            }

            if (realistic_deletion)
            {
                EdgeUpdateBatch<VertexData, EdgeData> deletion_updates(dynamic_graph.number_vertices);
                deletion_updates.generateEdgeUpdates(dynamic_graph, batch_size, (round * update_iterations) + update_round, range, offset);
                
                dynamic_graph.edgeDeletion(deletion_updates);

                if (verify_enabled)
                {
                    CSR<DataType> test_graph;
                    dynamic_graph.dynGraphToCSR(test_graph);
                    verification.hostEdgeDeletion(deletion_updates);
                    verification.printAdjacency(7);
                    std::string header = std::string("Deletion - ") + std::to_string(update_round);
                    verification.verify(test_graph, header.c_str(), OutputCodes::VERIFY_DELETION);
                }
            }
            else
            {
                dynamic_graph.edgeDeletion(insertion_updates);
                if (verify_enabled)
                {
                    CSR<DataType> test_graph;
                    dynamic_graph.dynGraphToCSR(test_graph);
                    verification.hostEdgeDeletion(insertion_updates);
                    std::string header = std::string("Deletion - ") + std::to_string(update_round);
                    verification.verify(test_graph, header.c_str(), OutputCodes::VERIFY_DELETION);
                }
            }
        }
        init_measure.addMeasure(dynamic_graph.init_performance);
        insert_measure.addMeasure(dynamic_graph.insert_performance);
        delete_measure.addMeasure(dynamic_graph.delete_performance);

        dynamic_graph.cleanup();
    }
    auto init_result = init_measure.generateResult();
    auto insert_result = insert_measure.generateResult();
    auto delete_result = delete_measure.generateResult();
    
    if(write_csv)
    {
        // Write results to CSV
        const auto stats = input_graph.rowStatistics();
        results << "Name;#vertices;#edges;minAdj;maxAdj;meanAdj;stddev\n";
        results 
            << input_graph.filename << ";"
            << input_graph.rows << ";"
            << input_graph.nnz << ";"
            << stats.min << ";"
            << stats.max << ";"
            << stats.mean << ";"
            << stats.std_dev << "\n";
        
        // Performance
        results << "\nInit-Timing (iter);mean;std_dev;median\n";
        results 
            << init_result.num_ << ";"
            << init_result.mean_ << ";"
            << init_result.std_dev_ << ";"
            << init_result.median_ << "\n";
        results << "\nInsert-Timing (iter);mean;std_dev;median\n";
        results 
            << insert_result.num_ << ";"
            << insert_result.mean_ << ";"
            << insert_result.std_dev_ << ";"
            << insert_result.median_ << "\n";
        results << "\nDelete-Timing (iter);mean;std_dev;median\n";
        results 
            << delete_result.num_ << ";"
            << delete_result.mean_ << ";"
            << delete_result.std_dev_ << ";"
            << delete_result.median_ << "\n";
    }
    
    if(printResults)
    {
        int width{10};
        std::cout  << std::setw(width) << "-" << " | "<< std::setw(width) << "iter" << " | " << std::setw(width) << "mean" << " | " << std::setw(width) << "std_dev" << " | " << std::setw(width) << "median\n";
        std::cout << std::setw(width) << "Init" << " | " << std::setw(width)
            << init_result.num_ << " | " << std::setw(width)
            << init_result.mean_ << " | " << std::setw(width)
            << init_result.std_dev_ << " | " << std::setw(width)
            << init_result.median_ << "\n";
        std::cout << std::setw(width) << "Insert" << " | " << std::setw(width)
            << insert_result.num_ << " | "  << std::setw(width)
            << insert_result.mean_ << " | " << std::setw(width)
            << insert_result.std_dev_ << " | " << std::setw(width)
            << insert_result.median_ << "\n";
        std::cout<< std::setw(width) << "Delete" << " | " << std::setw(width)
            << delete_result.num_ << " | " << std::setw(width)
            << delete_result.mean_ << " | " << std::setw(width)
            << delete_result.std_dev_ << " | " << std::setw(width)
            << delete_result.median_ << "\n";
    }
}