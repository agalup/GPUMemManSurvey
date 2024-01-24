#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "UtilityFunctions.cuh"
#include "PerformanceMeasure.cuh"
#include "DevicePerformanceMeasure.cuh"
#include "runtime_system_one_srvc.cuh"

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

//Allocation - a monolithic application test
template <typename xyz, bool warp_based>
//__launch_bounds__(512, 1)
__global__ void d_testAllocation(xyz mm, int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
		if(threadIdx.x % 32 == 0)
			verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size));
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;

		verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size));
	}
}

//Allocation - a fissioned application in the modular runtime system (warp based allocation available)
template <typename Runtime, bool warp_based>
__global__  
void d_testAsyncAllocation_RS(Runtime rs, typename Runtime::Future* future_ptr, unsigned int num_allocations, unsigned int allocation_size)
{
    int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
		if(threadIdx.x % 32 == 0){ 
            rs.malloc_async(&future_ptr[tid], allocation_size);
        }
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;
        rs.malloc_async(&future_ptr[tid], allocation_size);
    }
}

//Allocation - a fissioned application in the modular runtime system (warp based allocation available)
template <typename Runtime, bool warp_based>
__global__  
void d_testAllocation_RS(Runtime rs, int** verification_ptr, unsigned int num_allocations, unsigned
int allocation_size)
{
    
    int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
		if(threadIdx.x % 32 == 0){ rs.malloc((volatile int**)&verification_ptr[tid], allocation_size);
        }
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;

        rs.malloc((volatile int**)&verification_ptr[tid], allocation_size);
    }
}

template <typename MemoryManagerType, bool warp_based>
__global__ void d_testFree(MemoryManagerType mm, int** verification_ptr, int num_allocations)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
	
		if(threadIdx.x % 32 == 0)
			mm.free(verification_ptr[tid]);
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;

		mm.free(verification_ptr[tid]);
	}
}
//Free - a fissioned application in the modular runtime system (warp based allocation available)
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

//Free - a fissioned application in the modular runtime system (warp based allocation available)
template <typename Runtime, bool warp_based>
__global__ void d_testFutureFree_RS(Runtime rs, typename Runtime::Future* future_ptr, int num_allocations)
{
	int tid{0};
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
	
		if(threadIdx.x % 32 == 0)
            rs.free(future_ptr[tid].ptr);
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;

        rs.free(future_ptr[tid].ptr);
	}
}
template <typename Runtime>
__global__ void d_testFutureWriteToMemory(typename Runtime::Future* future_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = future_ptr[tid].get();
    assert(ptr);

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

__global__ void d_testWriteToMemory(int** verification_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];
    assert(ptr);

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}
template <typename Runtime>
__global__ void d_testFutureReadFromMemory(typename Runtime::Future* future_ptr, int num_allocations, int allocation_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = future_ptr[tid].get();

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d | We got a wrong value here! %d vs %d\n", tid, ptr[i], tid);
			__trap();
		}
	}
}
__global__ void d_testReadFromMemory(int** verification_ptr, int num_allocations, int allocation_size)
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

void get_kernel_attributes(cudaFuncAttributes& funcAttribMalloc){
	printf("binaryVersion = %d\n",                   funcAttribMalloc.binaryVersion);
	printf("cacheModeCA = %d\n",                     funcAttribMalloc.cacheModeCA);
	printf("constSizeBytes = %d\n",                  funcAttribMalloc.constSizeBytes);
	printf("localSizeBytes = %d\n",                  funcAttribMalloc.localSizeBytes);
	printf("maxDynamicSharedSizeBytes = %d\n",       funcAttribMalloc.maxDynamicSharedSizeBytes);
	printf("maxThreadsPerBlock = %d\n",              funcAttribMalloc.maxThreadsPerBlock);
	printf("numRegs = %d\n",                         funcAttribMalloc.numRegs);
	printf("preferredShmemCarvout = %d\n",           funcAttribMalloc.preferredShmemCarveout);
	printf("ptxVersion = %d\n",                      funcAttribMalloc.ptxVersion);
	printf("sharedSizeBytes = %d\n",                 funcAttribMalloc.sharedSizeBytes);
}

void MPS_single_kernel_launch(void* kernel_app, unsigned int gridSize, unsigned int blockSize, void** args, unsigned int
sm_app, unsigned int device, CUcontext& app_ctx){
    std::thread app_thread{[&]{
        GUARD_CU((cudaError_t)cuCtxPushCurrent(app_ctx));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        //GUARD_CU(cudaLaunchCooperativeKernel(kernel_app, gridSize, blockSize, args));
        //printf("for app: blockSize %d, gridSize %d\n", blockSize, gridSize);
        GUARD_CU(cudaLaunchKernel(kernel_app, gridSize, blockSize, args));
        GUARD_CU(cudaPeekAtLastError());    
        GUARD_CU(cudaGetLastError());
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        GUARD_CU((cudaError_t)cuCtxPopCurrent(&app_ctx));
        GUARD_CU(cudaPeekAtLastError());    
        GUARD_CU(cudaGetLastError());
    }};
    app_thread.join();
}

int main(int argc, char* argv[])
{
    /*printf("[%d]driver test\n", __LINE__);
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaPeekAtLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
    printf("driver test done\n");*/
	// Usage: <num_allocations> <size_of_allocation_in_byte> <num_iterations> <onDeviceMeasure> <warp-based> <generateoutput> <free_memory> <initial_path>
	printf(" Usage: <num_allocations> <size_of_allocation_in_byte> <num_iterations> <runtime> <app_sm> <blockSize> <mm_sm> <mm_gridSize> <mm_blockSize> <onDeviceMeasure> <warp-based> <generateoutput> <free_memory> <initial_path>\n");
	unsigned int num_allocations{10000};
	unsigned int allocation_size_byte{8192};
	int num_iterations {100};
	bool warp_based{false};
	bool onDeviceMeasure{false};
	bool print_output{true};
	bool generate_output{false};
	bool free_memory{true};
    //int factor{16};
    int mm_sm {26};
    int app_sm {82};
    int mm_blockSize {1024};
    int mm_gridSize {26};
    int blockSize {1024};
	std::string alloc_csv_path{"../results/tmp/"};
	std::string write_csv_path{"../results/tmp/"};
	std::string free_csv_path{"../results/tmp/"};
	//int allocSizeinGB{8};
	int allocSizeinGB{16};
    int device{0};
    int runtime{0};
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
                    runtime = atoi(argv[4]);
                    if(argc >= 6)
                    {
                        //factor = atoi(argv[4]);
                        app_sm = atoi(argv[5]);
                        if(argc >= 7)
                        {
                            blockSize = atoi(argv[6]);
                            if (argc >= 8)
                            {
                                mm_sm = atoi(argv[7]);
                                if (argc >= 9)
                                {
                                    mm_gridSize = atoi(argv[8]);
                                    if (argc >= 10)
                                    {
                                        mm_blockSize = atoi(argv[9]);
                                        if (argc >= 11)
                                        {
                                            onDeviceMeasure = static_cast<bool>(atoi(argv[10]));
                                            if(argc >= 12)
                                            {
                                                warp_based = static_cast<bool>(atoi(argv[11]));
                                                if(onDeviceMeasure && warp_based)
                                                {
                                                    std::cout << "OnDeviceMeasure and warp-based not possible at the same!" << std::endl;
                                                    exit(-1);
                                                }
                                                if(argc >= 13)
                                                {
                                                    generate_output = static_cast<bool>(atoi(argv[12]));
                                                    if(argc >= 14)
                                                    {
                                                        free_memory = static_cast<bool>(atoi(argv[13]));
                                                        if(argc >= 15)
                                                        {
                                                            alloc_csv_path = std::string(argv[14]);
                                                            if(argc >= 16)
                                                            {
                                                                write_csv_path = std::string(argv[15]);
                                                                if (argc >= 17)
                                                                {
                                                                    free_csv_path = std::string(argv[16]);
                                                                    if(argc >= 18)
                                                                    {
                                                                        allocSizeinGB = atoi(argv[17]);
                                                                        if(argc >= 19)
                                                                        {
                                                                            device = atoi(argv[18]);
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

	CHECK_ERROR(cudaSetDevice(device));
	cudaDeviceProp prop;
	CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
    CHECK_ERROR(cudaPeekAtLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";
	std::cout << "--- " << mem_name << "---\n";
    
    //int best_factor;
    //double min_time = 100.0;
    //int factor = 20;
    //for (int factor = 10; factor < 30; factor+=1){
    std::ofstream results_alloc, results_write, results_free;
	if(generate_output)
	{
		results_alloc.open(alloc_csv_path.c_str(), std::ios_base::app);
        results_write.open(write_csv_path.c_str(), std::ios_base::app);
		results_free.open(free_csv_path.c_str(), std::ios_base::app);
	}
	//int blockSize {1024};
	//int blockSize {256};
    //int mm_blockSize {256};
    //int mm_blocks_per_sm {1};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};
	if(warp_based)
		gridSize *= 32;

	PerfMeasure timing_allocation;
	PerfMeasure timing_free;
	PerfMeasure timing_write;

    DevicePerfMeasure per_thread_timing_allocation(num_allocations, num_iterations);
    DevicePerfMeasure per_thread_timing_free(num_allocations, num_iterations);
    size_t mem_pool_size = (size_t)(allocSizeinGB*1024ULL*1024ULL*1024ULL);
    debug("memory pool available to allocate: %lu, app is going to allocate in total %u\n", mem_pool_size, num_allocations*allocation_size_byte); 

    MemoryManager memory_manager(mem_pool_size);

#ifdef TEST_OUROBOROS
    debug("mm with direct mm ptr type\n");
    using MemoryManager2 = std::remove_pointer<decltype(memory_manager.d_memory_manager)>::type;
#else
    using MemoryManager2 = MemoryManager;
#endif

  /*  
    printf("block size %d, # blocks %d\n", blockSize, gridSize);
{
	struct cudaFuncAttributes funcAttribMalloc;
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribMalloc, d_testAllocation<MemoryManager, false>));
    get_kernel_attributes(funcAttribMalloc);
}
    printf("\n");
{
    int numBlocks;
    CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, d_testAllocation<MemoryManager, false>, blockSize, 0));
    printf("Occupancy for Mono\n");
    printf("block size %d, max active blocks per SM: %d\n", blockSize, numBlocks);
    int activeWarps = numBlocks*blockSize/prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    printf("occupancy %lf %\n", (double)activeWarps/maxWarps * 100);
}
*/
    Runtime<MemoryManager2> rs;
    int multi_processor_count = prop.multiProcessorCount;
    fflush(stdout);
 /*   
    printf("\n");
{
	struct cudaFuncAttributes funcAttribMalloc;
    printf("(RS) d_testAllocation_RS<Runtime<MemoryManager2>, false>\n");
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribMalloc, d_testAllocation_RS<Runtime<MemoryManager2>, false>));
    get_kernel_attributes(funcAttribMalloc);
}
    printf("\n");
{
    int numBlocks;
    CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, d_testAllocation_RS<Runtime<MemoryManager2>, false>,
    blockSize, 0));
    printf("Occupancy for RS\n");
    printf("block size %d, max active blocks per SM: %d\n", blockSize, numBlocks);
    int activeWarps = numBlocks*blockSize/prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    printf("warpSize %d, max threads/SM %d, active warps/SM %d, max active warps/SM %d\n", prop.warpSize,
    prop.maxThreadsPerMultiProcessor, activeWarps, maxWarps);
    printf("occupancy %lf %\n", (double)activeWarps/maxWarps * 100);
}
    printf("\n");
{
	struct cudaFuncAttributes funcAttribMalloc;
    printf("Memory Manager Service <MemoryManager>\n");
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribMalloc, mem_manager_service<MemoryManager>));
    get_kernel_attributes(funcAttribMalloc);
}
    printf("\n");
{
	struct cudaFuncAttributes funcAttribMalloc;
    printf("Memory Manager Service <MemoryManager2>\n");
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribMalloc, mem_manager_service<MemoryManager2>));
    get_kernel_attributes(funcAttribMalloc);
}
    printf("\n");
{
    int numBlocks;
    CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, mem_manager_service<MemoryManager2>, mm_blockSize, 0));
    printf("Occupancy for Memory Manager Service\n");
    printf("block size %d, max active blocks per SM: %d\n", mm_blockSize, numBlocks);
    if (mm_gridSize > mm_sm * numBlocks){
        printf("%d blocks is not able to reside on %d SMs, ", mm_gridSize, mm_sm);
        mm_gridSize = numBlocks * mm_sm;
        printf("new # blocks %d\n", mm_gridSize);
    }
    int activeWarps = numBlocks*mm_blockSize/prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    printf("occupancy %lf %\n", (double)activeWarps/maxWarps * 100);
}
    printf("\n");
    */

    // Sync Runtime: runtime == 1
    // Async Runtime: runtime == 2
    if (runtime == 1 or runtime == 2){
        rs.init(num_allocations, 0, memory_manager.d_memory_manager, mem_pool_size, app_sm/*, factor*/, multi_processor_count, mm_sm, mm_blockSize, mm_gridSize);//, mm_blocks_per_sm);
    }
    printf("\napp #threads %d x #blocks %d = #total threads %d\n", blockSize, gridSize, blockSize*gridSize); 

    CUcontext current;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&current));
    debug("current ctx %p\n", current);
    fflush(stdout);

    CUcontext app_ctx; 
    CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int) app_sm};
    auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
    GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
    GUARD_CU((cudaError_t)cuCtxGetExecAffinity(&app_param, CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT));
    printf("app sm %d/%d\n", app_param.param.smCount.val, multi_processor_count);
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&app_ctx));
    debug("app ctx %p\n", app_ctx);
    GUARD_CU((cudaError_t)cuCtxSynchronize());

    int active_app_warps = app_sm * 2048;
    int active_mm_warps = mm_blockSize * mm_gridSize; //for persistent kernel all its blocks are always active.

    if (runtime == 2){
        printf("Async Fissioned Application\n");
    }else if (runtime == 1){
        printf("Fissioned Application\n");
    }else if (runtime == 0){
        printf("Monolithic Application\n");
    }

    //printf("# active app warps per active service warp %d\n", gridSize*blockSize/(mm_gridSize*mm_blockSize));
    printf("# active app warps per active service warp %d\n", active_app_warps/active_mm_warps);

    int** d_memory{nullptr};
    if (runtime == 1 or runtime == 0){
        CHECK_ERROR(cudaMalloc((void**)&d_memory, sizeof(volatile int*) * (num_allocations)));
        CHECK_ERROR(cudaDeviceSynchronize());
    }

    Runtime<MemoryManager2>::Future* d_memory_f{nullptr};
    if (runtime == 2){
        CHECK_ERROR(cudaMalloc((void**)&d_memory_f, sizeof(Runtime<MemoryManager2>::Future) * (num_allocations)));
        CHECK_ERROR(cudaDeviceSynchronize());
    }

    for(auto it = 0; it < num_iterations; ++it)
    {
        std::cout << "#" << std::flush;// << it << "\n" << std::flush;
        if (runtime == 1){ //runtime
            {
                void* args[] = {&rs, &d_memory, &num_allocations, &allocation_size_byte};
                timing_allocation.startMeasurement();
                if(warp_based){
                    rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
                }else{
                    CHECK_ERROR(cudaProfilerStart());
                    rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
                    CHECK_ERROR(cudaProfilerStop());
                }
                timing_allocation.stopMeasurement();
            }
            CHECK_ERROR(cudaDeviceSynchronize());

            void* args2[] = {&d_memory, &num_allocations, &allocation_size_byte};
            
            timing_write.startMeasurement();
            rs.run_sync((void*)d_testWriteToMemory, gridSize, blockSize, args2, app_ctx);
            timing_write.stopMeasurement();
            CHECK_ERROR(cudaDeviceSynchronize());
            
            rs.run_sync((void*)d_testReadFromMemory, gridSize, blockSize, args2, app_ctx);
            CHECK_ERROR(cudaDeviceSynchronize());

            {
                void* args[] = {&rs, &d_memory, &num_allocations};
                timing_free.startMeasurement();
                if(warp_based){
                    rs.run_sync((void*)d_testFree_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
                }else{
                    rs.run_sync((void*)d_testFree_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
                }
                timing_free.stopMeasurement();
            }
            CHECK_ERROR(cudaDeviceSynchronize());

        }if (runtime == 2){ //async runtime

            {
                void* args[] = {&rs, &d_memory_f, &num_allocations, &allocation_size_byte};
                timing_allocation.startMeasurement();
                if(warp_based){
                    rs.run_sync((void*)d_testAsyncAllocation_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
                }else{
                    CHECK_ERROR(cudaProfilerStart());
                    rs.run_sync((void*)d_testAsyncAllocation_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
                    CHECK_ERROR(cudaProfilerStop());
                }
                timing_allocation.stopMeasurement();
            }
            CHECK_ERROR(cudaDeviceSynchronize());
            {
                void* args2[] = {&d_memory_f, &num_allocations, &allocation_size_byte};
                
                timing_write.startMeasurement();
                rs.run_sync((void*)d_testFutureWriteToMemory<Runtime<MemoryManager2>>, gridSize, blockSize, args2, app_ctx);
                timing_write.stopMeasurement();
                CHECK_ERROR(cudaDeviceSynchronize());
                
                rs.run_sync((void*)d_testFutureReadFromMemory<Runtime<MemoryManager2>>, gridSize, blockSize, args2, app_ctx);
                CHECK_ERROR(cudaDeviceSynchronize());
            }
            {
                void* args[] = {&rs, &d_memory_f, &num_allocations};
                timing_free.startMeasurement();
                if(warp_based){
                    rs.run_sync((void*)d_testFutureFree_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
                }else{
                    rs.run_sync((void*)d_testFutureFree_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
                }
                timing_free.stopMeasurement();
            }
            CHECK_ERROR(cudaDeviceSynchronize());
        }else{  //no runtime

            {
                void* args[] = {&memory_manager, &d_memory, &num_allocations, &allocation_size_byte};
                timing_allocation.startMeasurement();
                if(warp_based){
                    MPS_single_kernel_launch((void*)d_testAllocation<decltype(memory_manager), true>, gridSize,
                    blockSize, args, app_sm, device, app_ctx);
                }else{
                    CHECK_ERROR(cudaProfilerStart());
                    MPS_single_kernel_launch((void*)d_testAllocation<decltype(memory_manager), false>, gridSize,
                    blockSize, args, app_sm, device, app_ctx);
                    CHECK_ERROR(cudaProfilerStop());
                }
                timing_allocation.stopMeasurement();
            }
            CHECK_ERROR(cudaDeviceSynchronize());
            {
                void* args[] = {&d_memory, &num_allocations, &allocation_size_byte};
                timing_write.startMeasurement();
                MPS_single_kernel_launch((void*)d_testWriteToMemory, gridSize, blockSize, args, app_sm, device, app_ctx);
                timing_write.stopMeasurement();
                CHECK_ERROR(cudaDeviceSynchronize());

                //d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);
                MPS_single_kernel_launch((void*)d_testReadFromMemory, gridSize, blockSize, args, app_sm, device,
                app_ctx);
                CHECK_ERROR(cudaDeviceSynchronize());
            }

            {
                void* args[] = {&memory_manager, &d_memory, &num_allocations};
                timing_free.startMeasurement();
                if(warp_based){
                    //d_testFree <decltype(memory_manager), true> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
                    MPS_single_kernel_launch((void*)d_testFree<decltype(memory_manager), true>, gridSize, blockSize,
                    args, app_sm, device, app_ctx);
                }else{
                    //d_testFree <decltype(memory_manager), false> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
                    MPS_single_kernel_launch((void*)d_testFree<decltype(memory_manager), false>, gridSize, blockSize,
                    args, app_sm, device, app_ctx);
                }
                timing_free.stopMeasurement();
                CHECK_ERROR(cudaDeviceSynchronize());
            }
        }
    }
    if (runtime == 1 or runtime == 0){
        GUARD_CU(cudaFree(d_memory));
    }else if (runtime == 2){
        GUARD_CU(cudaFree(d_memory_f));
    }
    CHECK_ERROR(cudaDeviceSynchronize());

    if (runtime == 1 or runtime == 2){ 
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
	/*if(onDeviceMeasure)
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
	else*/
	{
		auto alloc_result = timing_allocation.generateResult();
		auto free_result = timing_free.generateResult();
        auto write_result = timing_write.generateResult();
		if(print_output)
		{
			std::cout << "Timing Allocation: Mean:" << alloc_result.mean_ << "ms" << std::endl;// " | Median: " << alloc_result.median_ << " ms" << std::endl;
			std::cout << "Timing      Write: Mean:" << write_result.mean_ << "ms" << std::endl;// "  | Median: " << free_result.median_ << " ms" << std::endl;
			std::cout << "Timing       Free: Mean:" << free_result.mean_ << "ms" << std::endl;// "  | Median: " << free_result.median_ << " ms" << std::endl;
		}
		if(generate_output)
		{
			results_alloc << alloc_result.mean_ << "," << alloc_result.std_dev_ << "," << alloc_result.min_ << "," << alloc_result.max_ << "," << alloc_result.median_;
			results_write << write_result.mean_ << "," << write_result.std_dev_ << "," << write_result.min_ << "," << write_result.max_ << "," << write_result.median_;
			results_free << free_result.mean_ << "," << free_result.std_dev_ << "," << free_result.min_ << "," << free_result.max_ << "," << free_result.median_;
        }
        /*if (alloc_result.mean_ < min_time){
            min_time = alloc_result.mean_;
            best_factor = factor;
        }*/
	}
    //printf("the best factor %d achieves %lf\n", best_factor, min_time);
    //}

	
	return 0;
}
