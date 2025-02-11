cmake_minimum_required(VERSION 3.16)
project(GPUMemoryManagers LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_PATH "../../")

include(${BASE_PATH}BaseCMake.cmake)

# Which frameworks to test
#option(TEST_CUDA "Test CUDA" True)
option(TEST_CUDA "Test CUDA" False)
option(TEST_HALLOC "Test Halloc" False)
option(TEST_SCATTERALLOC "Test ScatterAlloc" False)
#option(TEST_OUROBOROS "Test Ouroboros" False)
option(TEST_OUROBOROS "Test Ouroboros" True)
option(TEST_FDG "Test FDGMalloc" False)
option(TEST_REG "Test Register-Efficient" False)
option(TEST_XMALLOC "Test Xmalloc" False)

if(SYNC_BUILD)
set(TEST_CUDA OFF)
set(TEST_OUROBOROS OFF)
else()
set(TEST_HALLOC OFF)
set(TEST_SCATTERALLOC OFF)
set(TEST_FDG OFF)
set(TEST_REG OFF)
set(TEST_XMALLOC OFF)
endif()

##########################################################################
##########################################################################
##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_PATH}include)
include_directories(${BASE_PATH}frameworks)
include_directories(${BASE_PATH}externals/runtime-system)

##########################################################################
##########################################################################
##########################################################################
# CUDA
##########################################################################
##########################################################################
##########################################################################
if(TEST_CUDA)

	add_executable(c_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(c_alloc_test PUBLIC TEST_CUDA)
	target_include_directories(c_alloc_test PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
    target_link_libraries(c_alloc_test cuda)
	set_property(TARGET c_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET c_alloc_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#	add_executable(c_mixed_alloc_test
#		perf_mixed_allocation.cu
#	)
#	target_compile_definitions(c_mixed_alloc_test PUBLIC TEST_CUDA)
#	target_include_directories(c_mixed_alloc_test PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_link_libraries(c_mixed_alloc_test cuda)
#	set_property(TARGET c_mixed_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET c_mixed_alloc_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

endif()

##########################################################################
##########################################################################
##########################################################################
# Halloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_HALLOC AND NOT CUDA_BUILD_CC70_ASYNC)
	
	add_executable(h_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(h_alloc_test PUBLIC TEST_HALLOC)
	target_include_directories(h_alloc_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)
    target_include_directories(h_alloc_test PUBLIC externals)
	set_property(TARGET h_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(h_mixed_alloc_test
		perf_mixed_allocation.cu
	)
	target_compile_definitions(h_mixed_alloc_test PUBLIC TEST_HALLOC)
    target_include_directories(h_mixed_alloc_test PUBLIC externals)
	target_include_directories(h_mixed_alloc_test PUBLIC ${BASE_PATH}frameworks/halloc/repository/src)
	set_property(TARGET h_mixed_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# ScatterAlloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_SCATTERALLOC)

	add_executable(s_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(s_alloc_test PUBLIC TEST_SCATTERALLOC)
    target_include_directories(s_alloc_test PUBLIC externals)
	target_include_directories(s_alloc_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)
	set_property(TARGET s_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(s_mixed_alloc_test
		perf_mixed_allocation.cu
	)
	target_compile_definitions(s_mixed_alloc_test PUBLIC TEST_SCATTERALLOC)
    target_include_directories(s_mixed_alloc_test PUBLIC externals)
	target_include_directories(s_mixed_alloc_test PUBLIC ${BASE_PATH}frameworks/scatteralloc/repository/src/include/mallocMC)
	set_property(TARGET s_mixed_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# XMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_XMALLOC)

	add_executable(x_alloc_test
		perf_allocation.cu
	)
	target_compile_definitions(x_alloc_test PUBLIC TEST_XMALLOC)
	target_include_directories(x_alloc_test PUBLIC ${BASE_PATH}frameworks/xmalloc)
    target_include_directories(x_alloc_test PUBLIC externals)
	set_property(TARGET x_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(x_mixed_alloc_test
		perf_mixed_allocation.cu
	)
	target_compile_definitions(x_mixed_alloc_test PUBLIC TEST_XMALLOC)
    target_include_directories(x_mixed_alloc_test PUBLIC externals)
	target_include_directories(x_mixed_alloc_test PUBLIC ${BASE_PATH}frameworks/xmalloc)
	set_property(TARGET x_mixed_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

endif ()

##########################################################################
##########################################################################
##########################################################################
# Ouroboros
##########################################################################
##########################################################################
##########################################################################
if(TEST_OUROBOROS)

	add_executable(o_alloc_test_p
	perf_allocation.cu
	)
	target_compile_definitions(o_alloc_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_alloc_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_alloc_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
    target_include_directories(o_alloc_test_p PUBLIC ${BASE_PATH}externals/ouroboros/)
    target_link_libraries(o_alloc_test_p cuda)
	set_property(TARGET o_alloc_test_p PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET o_alloc_test_p PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	add_executable(o_async_alloc_test_p
	perf_async_allocation.cu
	)
	target_compile_definitions(o_async_alloc_test_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_async_alloc_test_p PUBLIC TEST_PAGES)
	target_include_directories(o_async_alloc_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
    target_include_directories(o_async_alloc_test_p PUBLIC ${BASE_PATH}externals/ouroboros/)
    target_link_libraries(o_async_alloc_test_p cuda)
	set_property(TARGET o_async_alloc_test_p PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET o_async_alloc_test_p PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	#add_executable(o_async_alloc_test_correctness_p
	#perf_async_allocation_correctness.cu
	#)
	#target_compile_definitions(o_async_alloc_test_correctness_p PUBLIC TEST_OUROBOROS)
	#target_compile_definitions(o_async_alloc_test_correctness_p PUBLIC TEST_PAGES)
	#target_include_directories(o_async_alloc_test_correctness_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
    #target_include_directories(o_async_alloc_test_correctness_p PUBLIC ${BASE_PATH}externals/ouroboros/)
    #target_link_libraries(o_async_alloc_test_correctness_p cuda)
	#set_property(TARGET o_async_alloc_test_correctness_p PROPERTY CUDA_ARCHITECTURES OFF)
    #set_property(TARGET o_async_alloc_test_correctness_p PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	add_executable(o_unified_kernel_launch_p
	perf_unified_kernel_launch.cu
	)
	target_compile_definitions(o_unified_kernel_launch_p PUBLIC TEST_OUROBOROS)
	target_compile_definitions(o_unified_kernel_launch_p PUBLIC TEST_PAGES)
	target_include_directories(o_unified_kernel_launch_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
    target_include_directories(o_unified_kernel_launch_p PUBLIC ${BASE_PATH}externals/ouroboros/)
    target_link_libraries(o_unified_kernel_launch_p cuda)
	set_property(TARGET   o_unified_kernel_launch_p PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET   o_unified_kernel_launch_p PROPERTY CUDA_SEPARABLE_COMPILATION ON)


#	add_executable(o_alloc_test_c
#	perf_allocation.cu
#	)
#	target_compile_definitions(o_alloc_test_c PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_alloc_test_c PUBLIC TEST_CHUNKS)
#	target_include_directories(o_alloc_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_alloc_test_c PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_alloc_test_c cuda)
#	set_property(TARGET o_alloc_test_c PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_alloc_test_c PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_alloc_test_vap
#	perf_allocation.cu
#	)
#	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_PAGES)
#	target_compile_definitions(o_alloc_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
#	target_include_directories(o_alloc_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_alloc_test_vap PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_alloc_test_vap cuda)
#	set_property(TARGET o_alloc_test_vap PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_alloc_test_vap PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_alloc_test_vac
#	perf_allocation.cu
#	)
#	target_compile_definitions(o_alloc_test_vac PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_alloc_test_vac PUBLIC TEST_CHUNKS)
#	target_compile_definitions(o_alloc_test_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
#	target_include_directories(o_alloc_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_alloc_test_vac PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_alloc_test_vac cuda)
#	set_property(TARGET o_alloc_test_vac PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_alloc_test_vac PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_alloc_test_vlp
#	perf_allocation.cu
#	)
#	target_compile_definitions(o_alloc_test_vlp PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_alloc_test_vlp PUBLIC TEST_PAGES)
#	target_compile_definitions(o_alloc_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
#	target_include_directories(o_alloc_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_alloc_test_vlp PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_alloc_test_vlp cuda)
#	set_property(TARGET o_alloc_test_vlp PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_alloc_test_vlp PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_alloc_test_vlc
#	perf_allocation.cu
#	)
#	target_compile_definitions(o_alloc_test_vlc PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_alloc_test_vlc PUBLIC TEST_CHUNKS)
#	target_compile_definitions(o_alloc_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
#	target_include_directories(o_alloc_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_alloc_test_vlc PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_alloc_test_vlc cuda)
#	set_property(TARGET o_alloc_test_vlc PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_alloc_test_vlc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	##########################################################################
#	##########################################################################
#	##########################################################################
#
#	add_executable(o_mixed_alloc_test_p
#	perf_mixed_allocation.cu
#	)
#	target_compile_definitions(o_mixed_alloc_test_p PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_mixed_alloc_test_p PUBLIC TEST_PAGES)
#	target_include_directories(o_mixed_alloc_test_p PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_mixed_alloc_test_p PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_mixed_alloc_test_p cuda)
#	set_property(TARGET o_mixed_alloc_test_p PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_mixed_alloc_test_p PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_mixed_alloc_test_c
#	perf_mixed_allocation.cu
#	)
#	target_compile_definitions(o_mixed_alloc_test_c PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_mixed_alloc_test_c PUBLIC TEST_CHUNKS)
#	target_include_directories(o_mixed_alloc_test_c PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_mixed_alloc_test_c PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_mixed_alloc_test_c cuda)
#	set_property(TARGET o_mixed_alloc_test_c PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_mixed_alloc_test_c PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_mixed_alloc_test_vap
#	perf_mixed_allocation.cu
#	)
#	target_compile_definitions(o_mixed_alloc_test_vap PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_mixed_alloc_test_vap PUBLIC TEST_PAGES)
#	target_compile_definitions(o_mixed_alloc_test_vap PUBLIC TEST_VIRTUALIZED_ARRAY)
#	target_include_directories(o_mixed_alloc_test_vap PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_mixed_alloc_test_vap PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_mixed_alloc_test_vap cuda)
#	set_property(TARGET o_mixed_alloc_test_vap PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_mixed_alloc_test_vap PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_mixed_alloc_test_vac
#	perf_mixed_allocation.cu
#	)
#	target_compile_definitions(o_mixed_alloc_test_vac PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_mixed_alloc_test_vac PUBLIC TEST_CHUNKS)
#	target_compile_definitions(o_mixed_alloc_test_vac PUBLIC TEST_VIRTUALIZED_ARRAY)
#	target_include_directories(o_mixed_alloc_test_vac PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_mixed_alloc_test_vac PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_mixed_alloc_test_vac cuda)
#	set_property(TARGET o_mixed_alloc_test_vac PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_mixed_alloc_test_vac PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_mixed_alloc_test_vlp
#	perf_mixed_allocation.cu
#	)
#	target_compile_definitions(o_mixed_alloc_test_vlp PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_mixed_alloc_test_vlp PUBLIC TEST_PAGES)
#	target_compile_definitions(o_mixed_alloc_test_vlp PUBLIC TEST_VIRTUALIZED_LIST)
#	target_include_directories(o_mixed_alloc_test_vlp PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_mixed_alloc_test_vlp PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_mixed_alloc_test_vlp cuda)
#	set_property(TARGET o_mixed_alloc_test_vlp PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_mixed_alloc_test_vlp PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#
#	add_executable(o_mixed_alloc_test_vlc
#	perf_mixed_allocation.cu
#	)
#	target_compile_definitions(o_mixed_alloc_test_vlc PUBLIC TEST_OUROBOROS)
#	target_compile_definitions(o_mixed_alloc_test_vlc PUBLIC TEST_CHUNKS)
#	target_compile_definitions(o_mixed_alloc_test_vlc PUBLIC TEST_VIRTUALIZED_LIST)
#	target_include_directories(o_mixed_alloc_test_vlc PUBLIC ${BASE_PATH}frameworks/ouroboros/repository/include)
#    target_include_directories(o_mixed_alloc_test_vlc PUBLIC ${BASE_PATH}externals/ouroboros/)
#    target_link_libraries(o_mixed_alloc_test_vlc cuda)
#	set_property(TARGET o_mixed_alloc_test_vlc PROPERTY CUDA_ARCHITECTURES OFF)
#    set_property(TARGET o_mixed_alloc_test_vlc PROPERTY CUDA_SEPARABLE_COMPILATION ON)

endif ()

##########################################################################
##########################################################################
##########################################################################
# FDGMalloc
##########################################################################
##########################################################################
##########################################################################
if(TEST_FDG)

	add_executable(f_alloc_test
	perf_allocation.cu
	)
	target_compile_definitions(f_alloc_test PUBLIC TEST_FDG)
	set_property(TARGET f_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(f_mixed_alloc_test
	perf_mixed_allocation.cu
	)
	target_compile_definitions(f_mixed_alloc_test PUBLIC TEST_FDG)
	set_property(TARGET f_mixed_alloc_test PROPERTY CUDA_ARCHITECTURES OFF)

endif()

##########################################################################
##########################################################################
##########################################################################
# Register-Efficient Allocation
##########################################################################
##########################################################################
##########################################################################
if(TEST_REG)

	add_executable(r_alloc_test_a
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_a PUBLIC TEST_ATOMIC)
	set_property(TARGET r_alloc_test_a PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_alloc_test_aw
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_aw PUBLIC TEST_ATOMIC_WRAP)
	set_property(TARGET r_alloc_test_aw PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_alloc_test_c
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_c PUBLIC TEST_CIRCULAR)
	set_property(TARGET r_alloc_test_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_alloc_test_cf
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_cf PUBLIC TEST_CIRCULAR_FUSED)
	set_property(TARGET r_alloc_test_cf PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_alloc_test_cm
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_cm PUBLIC TEST_CIRCULAR_MULTI)
	set_property(TARGET r_alloc_test_cm PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_alloc_test_cfm
		perf_allocation.cu
	)
	target_compile_definitions(r_alloc_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_alloc_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)
	set_property(TARGET r_alloc_test_cfm PROPERTY CUDA_ARCHITECTURES OFF)

	##########################################################################
	##########################################################################
	##########################################################################

	add_executable(r_mixed_alloc_test_a
		perf_mixed_allocation.cu
	)
	target_compile_definitions(r_mixed_alloc_test_a PUBLIC TEST_REGEFF)
	target_compile_definitions(r_mixed_alloc_test_a PUBLIC TEST_ATOMIC)
	set_property(TARGET r_mixed_alloc_test_a PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_mixed_alloc_test_aw
		perf_mixed_allocation.cu
	)
	target_compile_definitions(r_mixed_alloc_test_aw PUBLIC TEST_REGEFF)
	target_compile_definitions(r_mixed_alloc_test_aw PUBLIC TEST_ATOMIC_WRAP)
	set_property(TARGET r_mixed_alloc_test_aw PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_mixed_alloc_test_c
		perf_mixed_allocation.cu
	)
	target_compile_definitions(r_mixed_alloc_test_c PUBLIC TEST_REGEFF)
	target_compile_definitions(r_mixed_alloc_test_c PUBLIC TEST_CIRCULAR)
	set_property(TARGET r_mixed_alloc_test_c PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_mixed_alloc_test_cf
		perf_mixed_allocation.cu
	)
	target_compile_definitions(r_mixed_alloc_test_cf PUBLIC TEST_REGEFF)
	target_compile_definitions(r_mixed_alloc_test_cf PUBLIC TEST_CIRCULAR_FUSED)
	set_property(TARGET r_mixed_alloc_test_cf PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_mixed_alloc_test_cm
		perf_mixed_allocation.cu
	)
	target_compile_definitions(r_mixed_alloc_test_cm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_mixed_alloc_test_cm PUBLIC TEST_CIRCULAR_MULTI)
	set_property(TARGET r_mixed_alloc_test_cm PROPERTY CUDA_ARCHITECTURES OFF)

	add_executable(r_mixed_alloc_test_cfm
		perf_mixed_allocation.cu
	)
	target_compile_definitions(r_mixed_alloc_test_cfm PUBLIC TEST_REGEFF)
	target_compile_definitions(r_mixed_alloc_test_cfm PUBLIC TEST_CIRCULAR_FUSED_MULTI)
	set_property(TARGET r_mixed_alloc_test_cfm PROPERTY CUDA_ARCHITECTURES OFF)

endif()
