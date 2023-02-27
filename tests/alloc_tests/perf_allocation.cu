#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "UtilityFunctions.cuh"
#include "PerformanceMeasure.cuh"
#include "DevicePerformanceMeasure.cuh"
#include "runtime_system.cuh"

#define DIRECT_MM_PTR

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

template <typename Runtime, bool warp_based>
__global__ void d_testAllocation_RS(Runtime rs, volatile int** verification_ptr, unsigned int num_allocations, unsigned int allocation_size)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
		if(threadIdx.x % 32 == 0){
            rs.malloc((volatile int**)&verification_ptr[tid], allocation_size);
            assert(verification_ptr[tid]);
        }
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid == 0){
            printf("num allocations %d, allocation_size %d\n", num_allocations, allocation_size);
        }
		if(tid >= num_allocations)
			return;

        assert(verification_ptr);

        rs.malloc((volatile int**)&verification_ptr[tid], allocation_size);
        assert(verification_ptr[tid]);
        __threadfence();
        __syncthreads();
	}
}

template <typename Runtime>
__global__ void d_testAllocation_RS(Runtime rs, volatile int** verification_ptr, int num_allocations, int allocation_size, DevicePerfMeasure::Type* timing)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	DevicePerf perf_measure;
	
	// Start Measure
	perf_measure.startThreadMeasure();

    int* ptr;
    rs.malloc((volatile int**)&ptr, allocation_size);
	
	// Stop Measure
	timing[tid] = perf_measure.stopThreadMeasure();

	verification_ptr[tid] = ptr;
}

template <typename MemoryManagerType, bool warp_based>
__global__ void d_testFree_RS(Runtime<MemoryManagerType> rs, volatile int** verification_ptr, int num_allocations)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
	
		if(threadIdx.x % 32 == 0)
			rs.free(verification_ptr[tid]);
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;

		rs.free(verification_ptr[tid]);
	}
}

template <typename MemoryManagerType>
__global__ void d_testFree_RS(Runtime<MemoryManagerType> rs, volatile int** verification_ptr, int num_allocations, DevicePerfMeasure::Type* timing)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	DevicePerf perf_measure;

	// Start Measure
	perf_measure.startThreadMeasure();

	rs.free(verification_ptr[tid]);

	// Stop Measure
	timing[tid] = perf_measure.stopThreadMeasure();
}


__global__ void d_testWriteToMemory(volatile int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

__global__ void d_testReadFromMemory(volatile int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d | We got a wrong value here! %d vs %d\n", tid, ptr[i], tid);
			__trap();
		}
	}
}

int main(int argc, char* argv[])
{
	// Usage: <num_allocations> <size_of_allocation_in_byte> <num_iterations> <onDeviceMeasure> <warp-based> <generateoutput> <free_memory> <initial_path>
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte{8192};
	int num_iterations {100};
	bool warp_based{false};
	bool onDeviceMeasure{false};
	bool print_output{true};
	bool generate_output{false};
	bool free_memory{true};
	std::string alloc_csv_path{"../results/tmp/"};
	std::string free_csv_path{"../results/tmp/"};
	int allocSizeinGB{8};
	int device{0};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_byte = atoi(argv[2]);
			if(argc >= 4)
			{
				num_iterations = atoi(argv[3]);
				if(argc >= 5)
				{
					onDeviceMeasure = static_cast<bool>(atoi(argv[4]));
					if(argc >= 6)
					{
						warp_based = static_cast<bool>(atoi(argv[5]));
						if(onDeviceMeasure && warp_based)
						{
							std::cout << "OnDeviceMeasure and warp-based not possible at the same!" << std::endl;
							exit(-1);
						}
						if(argc >= 7)
						{
							generate_output = static_cast<bool>(atoi(argv[6]));
							if(argc >= 8)
							{
								free_memory = static_cast<bool>(atoi(argv[7]));
								if(argc >= 9)
								{
									alloc_csv_path = std::string(argv[8]);
									if(argc >= 10)
									{
										free_csv_path = std::string(argv[9]);
										if(argc >= 11)
										{
											allocSizeinGB = atoi(argv[10]);
											if(argc >= 12)
											{
												device = atoi(argv[11]);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
    }

    debug("num_allocations %d\n", num_allocations);
    debug("allocation size %d\n", allocation_size_byte);
    debug("num_iterations %d\n", num_iterations);
    debug("onDeviceMeasure %d\n", onDeviceMeasure);
    debug("warp_based %d\n", warp_based);
    debug("generate_output %d\n", generate_output);
    debug("free memory %d\n", free_memory);

	allocation_size_byte = Utils::alignment(allocation_size_byte, sizeof(int));
	if(print_output)
		std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << std::endl;

    /*int * tmp_dev;
    CHECK_ERROR(cudaMalloc((void**)&tmp_dev, sizeof(int)));*/
	CHECK_ERROR(cudaSetDevice(device));
	cudaDeviceProp prop;
	CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	std::cout << "--- " << mem_name << "---\n";

	volatile int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc((void**)&d_memory, sizeof(volatile int*) * (num_allocations+100)));


	std::ofstream results_alloc, results_free;
	if(generate_output)
	{
		results_alloc.open(alloc_csv_path.c_str(), std::ios_base::app);
		results_free.open(free_csv_path.c_str(), std::ios_base::app);
	}

	int blockSize {256};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};
    printf("blockSize %d, gridSize %d\n", blockSize, gridSize);
	if(warp_based)
		gridSize *= 32;

	PerfMeasure timing_allocation;
	PerfMeasure timing_free;

	DevicePerfMeasure per_thread_timing_allocation(num_allocations, num_iterations);
	DevicePerfMeasure per_thread_timing_free(num_allocations, num_iterations);

    //exit(1);
	for(auto i = 0; i < num_iterations; ++i)
    {
        debug("iteration %d\n", i);

#ifdef DIRECT_MM_PTR
    #ifdef TEST_OUROBOROS
        debug("mm with direct mm ptr type\n");
	    MemoryManager memory_manager(allocSizeinGB * 1024ULL * 1024ULL * 1024ULL);
        using MemoryManager2 = std::remove_pointer<decltype(memory_manager.d_memory_manager)>::type;
    #else
        debug("cuda mm\n");
	    MemoryManager* memory_manager;
        GUARD_CU(cudaMallocManaged((void**)&memory_manager, sizeof(MemoryManager)));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        new (memory_manager) MemoryManager(allocSizeinGB * 1024ULL * 1024ULL * 1024ULL);
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        using MemoryManager2 = MemoryManager;
    #endif
#else
        debug("mm without direct mm ptr type\n");
	    MemoryManager* memory_manager;
        GUARD_CU(cudaMallocManaged((void**)&memory_manager, sizeof(MemoryManager)));
        debug("mm, memory allocated, init to be\n");
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        new (memory_manager) MemoryManager(allocSizeinGB * 1024ULL * 1024ULL * 1024ULL);
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        debug("[%s:%d]mm address %x \n", __FUNCTION__, __LINE__, memory_manager);
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        //debug("[%s:%d]malloc address %x \n", __FUNCTION__, __LINE__, memory_manager->malloc);
        fflush(stdout);
        
	    //MemoryManager memory_manager(allocSizeinGB * 1024ULL * 1024ULL * 1024ULL);
        using MemoryManager2 = MemoryManager;
#endif
        Runtime<MemoryManager2> rs;
        int app_sm = 70;

#ifdef DIRECT_MM_PTR
    #ifdef TEST_OUROBOROS
        #ifdef CALLBACK__
            debug("RS with direct ptr to mm and callback!\n");
            rs.init(num_allocations, 0, memory_manager.d_memory_manager, 1, app_sm, 5, 4, 1, blockSize, 1);
        #else
            debug("RS with direct ptr to mm\n");
            rs.init(num_allocations, 0, memory_manager.d_memory_manager, app_sm, 5, 4, blockSize, 1);
        #endif
    #else
        #ifdef CALLBACK__
            debug("RS with direct ptr to mm and callback!\n");
            rs.init(num_allocations, 0, memory_manager, app_sm, 5, 4, 1, blockSize, 1);
        #else
            debug("RS with direct ptr to mm\n");
            rs.init(num_allocations, 0, memory_manager, app_sm, 5, 4, blockSize, 1);
        #endif
    #endif
#else
    #ifdef CALLBACK__
        debug("RS with callback\n");
        rs.init(num_allocations, 0, memory_manager, 1, app_sm, 5, 4, 1, blockSize, 1);
    #else
        debug("RS, no direct ptr to mm and no callback\n");
        rs.init(num_allocations, 0, memory_manager, app_sm, 5, 4, blockSize, 1);
    #endif
#endif
        CUcontext app_ctx; 
        CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int) app_sm};
        auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
        GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        CUcontext current_ctx;
        GUARD_CU((cudaError_t)cuCtxPopCurrent(&current_ctx));
        debug("current was %d\n", current_ctx);

        std::cout << "#" << std::flush;

        GUARD_CU((cudaError_t)cuCtxSynchronize());

        if(onDeviceMeasure)
        {
            void* args[] = {&rs, &d_memory, &num_allocations, &allocation_size_byte, per_thread_timing_allocation.getDevicePtr()};
            rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>>, gridSize, blockSize, args, app_ctx);
            CHECK_ERROR(cudaDeviceSynchronize());
            per_thread_timing_allocation.acceptResultsFromDevice();
        }
        else
        {
            void* args[] = {&rs, &d_memory, &num_allocations, &allocation_size_byte};
            timing_allocation.startMeasurement();
            if(warp_based){
                rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
            }else{
                rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
            }
            timing_allocation.stopMeasurement();
            CHECK_ERROR(cudaDeviceSynchronize());
        }
        debug("write\n");
        void* args2[] = {&d_memory, &num_allocations, &allocation_size_byte};
        rs.run_sync((void*)d_testWriteToMemory, gridSize, blockSize, args2, app_ctx);
        debug("read\n");
        void* args3[] = {&d_memory, &num_allocations, &allocation_size_byte};
        rs.run_sync((void*)d_testReadFromMemory, gridSize, blockSize, args3, app_ctx);
        debug("free\n");
        if(free_memory)
        {
            if(onDeviceMeasure)
            {
                void* args[] = {&rs, &d_memory, &num_allocations, per_thread_timing_allocation.getDevicePtr()};
                rs.run_sync((void*)d_testFree_RS<Runtime<MemoryManager2>>, gridSize, blockSize, args, app_ctx);
                //d_testFree_RS <<<gridSize, blockSize>>>(rs, d_memory, num_allocations, per_thread_timing_free.getDevicePtr());
                CHECK_ERROR(cudaDeviceSynchronize());
                per_thread_timing_free.acceptResultsFromDevice();
            }
            else
            {
                void* args[] = {&rs, &d_memory, &num_allocations};
                timing_free.startMeasurement();
                if(warp_based){
                    rs.run_sync((void*)d_testFree_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
                    //d_testFree_RS <MemoryManager2, true> <<<gridSize, blockSize>>>(rs, d_memory, num_allocations);
                }else{
                    rs.run_sync((void*)d_testFree_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
                }
                timing_free.stopMeasurement();
                CHECK_ERROR(cudaDeviceSynchronize());
            }
        }

        debug("stop services\n");
        rs.stop_services();
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        debug("clean memory\n");
        clean_memory(app_sm, blockSize, rs);
        GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        debug("memory cleaned\n");
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        debug("stop runtime\n");
        rs.stop_runtime();
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());

	}
	std::cout << std::endl;

	if(onDeviceMeasure)
	{
		auto alloc_result = per_thread_timing_allocation.generateResult();
		auto free_result = per_thread_timing_free.generateResult();

		if(print_output)
		{
			std::cout << "Timing Allocation: Mean:" << alloc_result.mean_ << "cycles | Median: " << alloc_result.median_ << " ms" << std::endl;
			std::cout << "Timing       Free: Mean:" << free_result.mean_ << "cycles | Median: " << free_result.median_ << " ms" << std::endl;
		}
		if(generate_output)
		{
			results_alloc << alloc_result.mean_ << "," << alloc_result.std_dev_ << "," << alloc_result.min_ << "," << alloc_result.max_ << "," << alloc_result.median_;
			results_free << free_result.mean_ << "," << free_result.std_dev_ << "," << free_result.min_ << "," << free_result.max_ << "," << free_result.median_;
		}
	}
	else
	{
		auto alloc_result = timing_allocation.generateResult();
		auto free_result = timing_free.generateResult();
		if(print_output)
		{
			std::cout << "Timing Allocation: Mean:" << alloc_result.mean_ << "ms" << std::endl;// " | Median: " << alloc_result.median_ << " ms" << std::endl;
			std::cout << "Timing       Free: Mean:" << free_result.mean_ << "ms" << std::endl;// "  | Median: " << free_result.median_ << " ms" << std::endl;
		}
		if(generate_output)
		{
			results_alloc << alloc_result.mean_ << "," << alloc_result.std_dev_ << "," << alloc_result.min_ << "," << alloc_result.max_ << "," << alloc_result.median_;
			results_free << free_result.mean_ << "," << free_result.std_dev_ << "," << free_result.min_ << "," << free_result.max_ << "," << free_result.median_;
		}
	}
	
	return 0;
}
