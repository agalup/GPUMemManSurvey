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


template <typename MemoryManagerType, bool warp_based>
//__launch_bounds__(256, 1)
__launch_bounds__(512, 1)
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, int allocation_size)
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

extern "C" 
__noinline__
//__forceinline__
__device__
int //__attribute__ ((regparm(30))) 
callee(long long int a1,  long long int a2,  long long int a3,  long long int a4,  long long int a5, 
       long long int a6,  long long int a7,  long long int a8,  long long int a9,  long long int a10,
       long long int a11, long long int a12, long long int a13, long long int a14){/*, long long int a15
       long long int a16, long long int a17, long long int a18, long long int a19, long long int a20
       long long int a21, long long int a22, long long int a23, long long int a24, long long int a25, 
       long long int a26, long long int a27, long long int a28, long long int a29, long long int a30){*/

     //  __shared__ long long int tab[10];

 //       #pragma unroll
 //       for (int i=0; i<20; ++i){
      /*      tab[0] = a1;
            tab[1] = a2;
            tab[2] = a3;
            tab[3] = a4;
            tab[4] = a5;
            tab[5] = a6;*/
        /*asm volatile("mov.u32 %0, %laneid;" : "=r"(a1));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a2));
        a1 += a2;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a3));
        a3 += a1 * a2;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a4));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a5));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a6));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a7));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a8));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a9));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a10));*/
        /*asm volatile("mov.u32 %0, %laneid;" : "=r"(a11));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a12));
        a1 += a2;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a13));
        a3 += a1 * a2;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a14));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a15));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a16));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a17));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a18));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a19));
        asm volatile("mov.u32 %0, %laneid;" : "=r"(a20));*/
  //      }
        //unsigned all = a1 + a2 + a3;//+ a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14;//+ a15;
        // + a16 + a17 +a18 + a19 + a20;
        /*
        asm volatile(".reg .u32 t1;\n\t"
                    ".reg .u32 t2;\n\t"
                       "mul.lo.u32 t1, %1, %1;\n\t"
                       "mul.lo.u32 t2, %2, %1;\n\t"
                       "mul.lo.u32 %0, t1, %1;"
                       : "=r"(all) : "r"(all), "r"(a4));
*/

    return a1;//all;
}

template <typename Runtime, bool warp_based>
__global__  
//__launch_bounds__(512, 1)
__launch_bounds__(512, 1)
void d_testAllocation_RS(Runtime rs, volatile int** verification_ptr, unsigned int num_allocations, unsigned
int allocation_size)
{
    
	/*register*/ int tid{0};
   // register 
   // int x1{0}, x2{0}, x3{0}, x4{0}, x5{0}, x6{0}, x7{0}, x8{0}, x9{0}, x10{0};;
	if(warp_based)
	{
		tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
		if(tid >= num_allocations)
			return;
		if(threadIdx.x % 32 == 0){
            rs.malloc((volatile int**)&verification_ptr[tid], allocation_size);
        }
	}
	else
	{
		tid = threadIdx.x + blockIdx.x * blockDim.x;
		if(tid >= num_allocations)
			return;

        rs.malloc((volatile int**)&verification_ptr[tid], allocation_size);
        //int result = callee(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);
        //printf("result = %d\n", result);

        __shared__ long long int ret, ret2, ret3, ret4, ret5, ret6, ret7, ret8,
        ret9, ret10, ret11, ret12, ret13, ret14;
        //ret15, ret16, ret17, ret18, ret19, ret20;//, ret21, ret22, ret23, ret24, ret25, ret26, ret27, ret28;
        //, ret29;
        /*ret30;*/
        //register unsigned ret = tid, ret2=allocation_size, ret3=verification_ptr[tid][0], ret4=blockIdx.x,
        //ret5=blockDim.x, ret6=ret5, ret7=tid, ret8=allocation_size, ret9, ret10, ret11, ret12, ret13, ret14,
        //ret15, ret16, ret17, ret18, ret19, ret20, ret21, ret22, ret23, ret24, ret25, ret26, ret27, ret28, ret29, ret30;
        //int result = callee(ret, ret2, ret3, ret4, ret5);
        /*, ret6, ret7, ret8, ret9, ret10, ret11, ret12, ret13, ret14, ret15, ret16, ret17, ret18, ret19, ret20);*/

        __shared__ long long int result;
        asm volatile(
                    "{\n\t"
                    ".reg .u64 t1;\n\t"
                    ".reg .u64 t2;\n\t"
                    ".reg .u64 t3;\n\t"
                    ".reg .u64 t4;\n\t"
                    ".reg .u64 t5;\n\t"
                    ".reg .u64 t6;\n\t"
                    ".reg .u64 t7;\n\t"
                    ".reg .u64 t8;\n\t"
                    ".reg .u64 t9;\n\t"
                    ".reg .u64 t10;\n\t"
                    ".reg .u64 t11;\n\t"
                    ".reg .u64 t12;\n\t"
                    ".reg .u64 t13;\n\t"
                    ".reg .u64 t14;\n\t"
                    /*".reg .u64 t15;\n\t"
                    ".reg .u64 t16;\n\t"
                    ".reg .u64 t17;\n\t"
                    ".reg .u64 t18;\n\t"
                    ".reg .u64 t19;\n\t"
                    ".reg .u64 t20;\n\t"
                    ".reg .u64 t21;\n\t"
                    ".reg .u64 t22;\n\t"
                    ".reg .u64 t23;\n\t"
                    ".reg .u64 t24;\n\t"
                    ".reg .u64 t25;\n\t"
                    ".reg .u64 t26;\n\t"
                    ".reg .u64 t27;\n\t"
                    ".reg .u64 t28;\n\t"
                    ".reg .u64 t29;\n\t"
                    ".reg .u64 t30;\n\t"*/
                    "mov.u64 t1, %1;\n\t" 
                    "mov.u64 t2, %2;\n\t" 
                    "mov.u64 t3, %3;\n\t" 
                    "mov.u64 t4, %4;\n\t" 
                    "mov.u64 t5, %5;\n\t"
                    "mov.u64 t6, %6;\n\t" 
                    "mov.u64 t7, %7;\n\t" 
                    "mov.u64 t8, %8;\n\t" 
                    "mov.u64 t9, %9;\n\t" 
                    "mov.u64 t10, %10;\n\t"
                    "mov.u64 t11, %11;\n\t" 
                    "mov.u64 t12, %12;\n\t" 
                    "mov.u64 t13, %13;\n\t" 
                    "mov.u64 t14, %14;\n\t" 
                    /*"mov.u64 t15, %15;\n\t"
                    "mov.u64 t16, %16;\n\t" 
                    "mov.u64 t17, %17;\n\t" 
                    "mov.u64 t18, %18;\n\t" 
                    "mov.u64 t19, %19;\n\t" 
                    "mov.u64 t20, %20;\n\t"
                    "mov.u64 t21, %21;\n\t" 
                    "mov.u64 t22, %22;\n\t" 
                    "mov.u64 t23, %23;\n\t" 
                    "mov.u64 t24, %24;\n\t" 
                    "mov.u64 t25, %25;\n\t"
                    "mov.u64 t26, %26;\n\t" 
                    "mov.u64 t27, %27;\n\t" 
                    "mov.u64 t28, %28;\n\t" 
                    "mov.u64 t9, %29;\n\t" 
                    "mov.u64 t10, %30;\n\t"*/
                    ".param .b32 retval0;\n\t"
                    "call.uni (retval0), callee, (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14);\n\t"
                    /*t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,t30);\n\t"*/
                    "ld.param.b32 %0, [retval0+0];\n\t"
                    "}\n\t"
                    : "=l"(ret) : "l"(result), "l"(ret2), "l"(ret3), "l"(ret4), "l"(ret5), "l"(ret6), "l"(ret7),
                    "l"(ret8),  "l"(ret9),  "l"(ret10), "l"(ret11), "l"(ret12), "l"(ret13), "l"(ret14));//, "l"(ret15));
                    /*"l"(ret16), "l"(ret17), "l"(ret18), "l"(ret19), "l"(ret20), "l"(ret21), "l"(ret22), "l"(ret23),
                    "l"(ret24), "l"(ret25), "l"(ret26), "l"(ret27), "l"(ret28), "l"(ret29));, "l"(ret30)
                    );*/
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
    assert(ptr);

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
	//int allocSizeinGB{8};
	int allocSizeinGB{16};
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

	CHECK_ERROR(cudaSetDevice(device));
	cudaDeviceProp prop;
	CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";
	std::cout << "--- " << mem_name << "---\n";

	volatile int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc((void**)&d_memory, sizeof(volatile int*) * (num_allocations)));

    int factor = 20;
    std::ofstream results_alloc, results_free;
	if(generate_output)
	{
		results_alloc.open(alloc_csv_path.c_str(), std::ios_base::app);
		results_free.open(free_csv_path.c_str(), std::ios_base::app);
	}

	//int blockSize {256};
	//int blockSize {1024};
	int blockSize {512};
	int gridSize {Utils::divup<int>(num_allocations, blockSize)};
	if(warp_based)
		gridSize *= 32;

	PerfMeasure timing_allocation;
	PerfMeasure timing_free;

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
       
    Runtime<MemoryManager2> rs;
    int app_sm = 0;
    int multi_processor_count = prop.multiProcessorCount;
    fflush(stdout);

    printf("\n");
{
	struct cudaFuncAttributes funcAttribMalloc;
    printf("(MONOLITHIC) d_testAllocation<MemoryManager, false>\n");
	CHECK_ERROR(cudaFuncGetAttributes(&funcAttribMalloc, d_testAllocation<MemoryManager, false>));
    get_kernel_attributes(funcAttribMalloc);
}
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
    CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, d_testAllocation<MemoryManager, false>,
    blockSize, 0));
    printf("Occupancy for monolithic\n");
    printf("max active blocks per SM: %d\n", numBlocks);
    int activeWarps = numBlocks*blockSize/prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    printf("warpSize %d, max threads/SM %d, active warps/SM %d, max active warps/SM %d\n", prop.warpSize,
    prop.maxThreadsPerMultiProcessor, activeWarps, maxWarps);
    printf("occupancy %lf %\n", (double)activeWarps/maxWarps * 100);
}
    printf("\n");
{
    int numBlocks;
    CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, d_testAllocation_RS<Runtime<MemoryManager2>, false>,
    blockSize, 0));
    printf("Occupancy for RS\n");
    printf("max active blocks per SM: %d\n", numBlocks);
    int activeWarps = numBlocks*blockSize/prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    printf("warpSize %d, max threads/SM %d, active warps/SM %d, max active warps/SM %d\n", prop.warpSize,
    prop.maxThreadsPerMultiProcessor, activeWarps, maxWarps);
    printf("occupancy %lf %\n", (double)activeWarps/maxWarps * 100);
}
    printf("\n");
{
    printf("Monolithic\n");
    printf("Max ptential block size\n");
    int potential_blockSize, min_grid_size;
    CHECK_ERROR(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &potential_blockSize,
    d_testAllocation<MemoryManager, false>)); 
    printf("potential block size %d\n", potential_blockSize);
    printf("min grid size %d\n", min_grid_size);
}
    printf("\n");
{
    printf("RS\n");
    printf("Max ptential block size\n");
    int potential_blockSize, min_grid_size;
    CHECK_ERROR(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &potential_blockSize,
    d_testAllocation_RS<Runtime<MemoryManager2>, false>)); 
    printf("potential block size %d\n", potential_blockSize);
    printf("min grid size %d\n", min_grid_size);
}

//return;

#ifdef TEST_OUROBOROS
    debug("RS with direct ptr to mm\n");
    rs.init(num_allocations, 0, memory_manager.d_memory_manager, mem_pool_size, app_sm, factor, multi_processor_count, blockSize, 1);
#else
    debug("RS without direct ptr to mm\n");
    rs.init(num_allocations, 0, (MemoryManager*)NULL, mem_pool_size, app_sm, factor, multi_processor_count, blockSize, 1);
#endif

    CUcontext current;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&current));
    debug("current ctx %p\n", current);

    fflush(stdout);

    CUcontext app_ctx; 
    CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int) app_sm};
    auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
    GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&app_ctx));
    debug("app ctx %p\n", app_ctx);
    GUARD_CU((cudaError_t)cuCtxSynchronize());


for(auto it = 0; it < num_iterations; ++it)
    {
        std::cout << "#" << std::flush;// << it << "\n" << std::flush;

        if(onDeviceMeasure)
        {
            void* args[] = {&rs, &d_memory, &num_allocations, &allocation_size_byte, per_thread_timing_allocation.getDevicePtr()};
            rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>>, gridSize, blockSize, args, app_ctx);
            per_thread_timing_allocation.acceptResultsFromDevice();
        }
        else
        {
            CHECK_ERROR(cudaProfilerStart());
            void* args[] = {&rs, &d_memory, &num_allocations, &allocation_size_byte};
            timing_allocation.startMeasurement();
            if(warp_based){
                rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>, true>, gridSize, blockSize, args, app_ctx);
            }else{
                //CHECK_ERROR(cudaProfilerStart());
                rs.run_sync((void*)d_testAllocation_RS<Runtime<MemoryManager2>, false>, gridSize, blockSize, args, app_ctx);
                //CHECK_ERROR(cudaProfilerStop());
            }
            timing_allocation.stopMeasurement();
            CHECK_ERROR(cudaProfilerStop());
        }
        
        void* args2[] = {&d_memory, &num_allocations, &allocation_size_byte};
        rs.run_sync((void*)d_testWriteToMemory, gridSize, blockSize, args2, app_ctx);
        rs.run_sync((void*)d_testReadFromMemory, gridSize, blockSize, args2, app_ctx);
        
        if(free_memory)
        {
            if(onDeviceMeasure)
            {
                void* args[] = {&rs, &d_memory, &num_allocations, per_thread_timing_allocation.getDevicePtr()};
                rs.run_sync((void*)d_testFree_RS<Runtime<MemoryManager2>>, gridSize, blockSize, args, app_ctx);
                per_thread_timing_free.acceptResultsFromDevice();
            }
            else
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

    GUARD_CU(cudaFree(d_memory));
	
	return 0;
}
