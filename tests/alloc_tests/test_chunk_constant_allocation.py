import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromFileAllocation
# from Helper import plotMean
import csv
import argparse
from sendEmail import EmailAlert

def main():
    print("##############################################################################")
    print("Callable as: python test_allocation.py -h")
    print("##############################################################################")
    
    # Specify which test configuration to use
    testcases = {}
    num_iterations = 100
    free_memory = 1
    filetype = "pdf"
    time_out_val = 10
    mm_sm = 22
    app_sm = 86
    mm_bl = 22
    mm_th = 1024
    app_th = 1024
    if os.name == 'nt': # If on Windows

        build_path = os.path.join("build", "Release")
        sync_build_path = os.path.join("sync_build", "Release")
    else:
        build_path = "build/"
        sync_build_path = "sync_build/"

    parser = argparse.ArgumentParser(description='Test allocation performance for various frameworks')
    parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
    parser.add_argument('-num', type=int, help='How many allocations to perform')
    parser.add_argument('-range', type=str, help='Specify Allocation Range, e.g. 4-1024')
    parser.add_argument('-iter', type=int, help='How many iterations?')
    parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
    parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
    parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
    parser.add_argument('-cleantemp', action='store_true', default=False, help='Clean up temporary files')
    parser.add_argument('-warp', action='store_true', default=False, help='Start testcases warp-based')
    parser.add_argument('-devmeasure', action='store_true', default=False, help='Measure performance on device in cycles')
    parser.add_argument('-plotscale', type=str, help='log/linear')
    parser.add_argument('-timeout', type=int, help='Timeout Value in Seconds, process will be killed after as many seconds')
    parser.add_argument('-filetype', type=str, help='png or pdf')
    parser.add_argument('-allocsize', type=int, help='How large is the manageable memory in GiB?', default=8)
    parser.add_argument('-device', type=int, help='Which device to use', default=0)
    parser.add_argument('-mailpass', type=str, help='Supply the mail password if you want to be notified', default=None)
    parser.add_argument('-mm_sm', type=int, help='# sm per MM', default=None)
    parser.add_argument('-app_sm', type=int, help='# sm per APP', default=None)
    parser.add_argument('-mm_bl', type=int, help='# blocks per MM', default=None)
    parser.add_argument('-mm_th', type=int, help='# threads per block MM', default=None)
    parser.add_argument('-app_th', type=int, help='# threads per block APP', default=None)

    args = parser.parse_args()

    executable_extension = ""
    if os.name == 'nt': # If on Windows
        executable_extension = ".exe"
    # Parse approaches
    if(args.t):
        if any("c" in s for s in args.t):
            testcases["CUDA"] = os.path.join(build_path, str("c_alloc_test") + executable_extension)
        if any("x" in s for s in args.t):
            testcases["XMalloc"] = os.path.join(sync_build_path, str("x_alloc_test") + executable_extension)
        if any("h" in s for s in args.t):
            testcases["Halloc"] = os.path.join(sync_build_path, str("h_alloc_test") + executable_extension)
        if any("s" in s for s in args.t):
            testcases["ScatterAlloc"] = os.path.join(sync_build_path, str("s_alloc_test") + executable_extension)
        if any("o" in s for s in args.t):
            testcases["Ouroboros-P-S"] = os.path.join(build_path, str("o_unified_kernel_launch_p") + executable_extension)
            #testcases["Ouroboros-P-S"] = os.path.join(build_path, str("o_async_alloc_test_p") + executable_extension)
            #testcases["Ouroboros-P-S"] = os.path.join(build_path, str("o_alloc_test_p") + executable_extension)
            #testcases["Ouroboros-P-VA"] = os.path.join(build_path, str("o_alloc_test_vap") + executable_extension)
            #testcases["Ouroboros-P-VL"] = os.path.join(build_path, str("o_alloc_test_vlp") + executable_extension)
            #testcases["Ouroboros-C-S"] = os.path.join(build_path, str("o_alloc_test_c") + executable_extension)
            #testcases["Ouroboros-C-VA"] = os.path.join(build_path, str("o_alloc_test_vac") + executable_extension)
            #testcases["Ouroboros-C-VL"] = os.path.join(build_path, str("o_alloc_test_vlc") + executable_extension)
        if any("f" in s for s in args.t):
            testcases["FDGMalloc"] = os.path.join(sync_build_path, str("f_alloc_test") + executable_extension)
        if any("r" in s for s in args.t):
            # testcases["RegEff-A"] = os.path.join(sync_build_path, str("r_alloc_test_a") + executable_extension)
            # testcases["RegEff-AW"] = os.path.join(sync_build_path, str("r_alloc_test_aw") + executable_extension)
            testcases["RegEff-C"] = os.path.join(sync_build_path, str("r_alloc_test_c") + executable_extension)
            testcases["RegEff-CF"] = os.path.join(sync_build_path, str("r_alloc_test_cf") + executable_extension)
            testcases["RegEff-CM"] = os.path.join(sync_build_path, str("r_alloc_test_cm") + executable_extension)
            testcases["RegEff-CFM"] = os.path.join(sync_build_path, str("r_alloc_test_cfm") + executable_extension)
    
    # Parse num allocation
    if(args.num):
        num_allocations = args.num

    # Parse num SMs for mm
    if(args.mm_sm):
        mm_sm = args.mm_sm
    
    # Parse num SMs for app
    if(args.app_sm):
        app_sm = args.app_sm

    # Parse num blocks for mm
    if(args.mm_bl):
        mm_bl = args.mm_bl
    
    # Parse num threads per block for mm
    if(args.mm_th):
        mm_th = args.mm_th
    
    # Parse num threads per block for mm
    if(args.app_th):
        app_th = args.app_th
    
    # Parse range
    if(args.range):
        selected_range = args.range.split('-')
        smallest_allocation_size = int(selected_range[0])
        largest_allocation_size = int(selected_range[1])

    # Parse num iterations
    if(args.iter):
        num_iterations = args.iter

    # Generate results
    warpstring = str("")
    if args.warp:
        test_warp_based = 1
        warpstring = "warp_"
    else:
        test_warp_based = 0
    
    # Run Testcases
    run_testcases = args.runtest
    
    # Generate results
    generate_results = args.genres

    # Generate plots
    generate_plots = args.genplot

    # Plot Axis scaling
    plotscale = args.plotscale

    # Clean temporary files
    clean_temporary_files = args.cleantemp

    # Measure on device
    if args.devmeasure:
        measure_on_device = 1
    else:
        measure_on_device = 0

    # Currently we cannot measure on the device when running warp-based
    if measure_on_device and test_warp_based:
        print("Cannot measure on device and warp-based at the same time!")
        exit(-1)

    # Timeout (in seconds)
    if(args.timeout):
        time_out_val = args.timeout
    
    if(args.filetype):
        filetype = args.filetype

    #mailalert = EmailAlert(args.mailpass)

    ####################################################################################################
    ####################################################################################################
    # Run testcases
    ####################################################################################################
    ####################################################################################################
    now = datetime.now()
    time_string = now.strftime("%b-%d-%Y_%H-%M-%S")
    
    if run_testcases:
        for i in [0, 1, 2]:
            runtime = i
            runtime_string = ""
            if runtime == 0: 
                runtime_string = "monolithic"
            elif runtime == 1:
                runtime_string = "fissioned"
            elif runtime == 2:
                runtime_string = "fissioned_async"
            # Run Testcase
            for name, path in testcases.items():
                allocation_size = 8
                num_allocations = 6400000 # -> 6250 chunks
                prefix = "results/performance/" + runtime_string 
                date = time_string + "_" + warpstring + ".csv"
                test_conf = "_" + str(num_allocations) + "_" + name 
                app_conf = "__app_sm" + str(app_sm) + "_th" + str(app_th)
                mm_conf  = "__mm_sm" + str(mm_sm) + "_bl" + str(mm_bl) + "_th" + str(mm_th)
                sufix = app_conf + mm_conf + test_conf 
                csv_path_alloc = prefix + "perf_alloc_" + sufix + date  
                csv_path_write = prefix + "perf_write_" + sufix + date 
                csv_path_free =  prefix + "perf_free_"  + sufix + date 
                if(os.path.isfile(csv_path_alloc)):
                    print("This file <" + csv_path_alloc + "x> already exists, do you really want to OVERWRITE?")
                    inputfromconsole = input()
                    if not (inputfromconsole == "yes" or inputfromconsole == "y"):
                        continue
                with open(csv_path_alloc, "w", newline='') as csv_file:
                    csv_file.write("AllocationSize (in Byte), mean, std-dev, min, max, median")
                with open(csv_path_write, "w", newline='') as csv_file:
                    csv_file.write("AllocationSize (in Byte), mean, std-dev, min, max, median")
                with open(csv_path_free, "w", newline='') as csv_file:
                    csv_file.write("AllocationSize (in Byte), mean, std-dev, min, max, median")
                
                while allocation_size <= 8192:
                    with open(csv_path_alloc, "a", newline='') as csv_file:
                        csv_file.write("\n" + str(allocation_size) + ",")
                    with open(csv_path_write, "a", newline='') as csv_file:
                        csv_file.write("\n" + str(allocation_size) + ",")
                    with open(csv_path_free, "a", newline='') as csv_file:
                        csv_file.write("\n" + str(allocation_size) + ",")
                    run_config = str(num_allocations) + " " + str(allocation_size) + " " + str(num_iterations) + " " + str(runtime) + " " + str(app_sm) + " " + str(app_th) + " " + str(mm_sm) + " " + str(mm_bl) + " " + str(mm_th) + " " + str(measure_on_device) + " " + str(test_warp_based) + " 1 " + str(free_memory) + " " + csv_path_alloc + " " + csv_path_write + " " + csv_path_free + " " + str(args.allocsize) + " " + str(args.device)
                    executecommand = "{0} {1}".format(path, run_config)
                    print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
                    print("Running " + name + " with command -> " + executecommand)
                    print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
                    _, process_killed = Command(executecommand).run(timeout=time_out_val)
                    if process_killed :
                        print("We killed the process!")
                        with open(csv_path_alloc, "a", newline='') as csv_file:
                            csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val))
                        with open(csv_path_write, "a", newline='') as csv_file:
                            csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val))
                        with open(csv_path_free, "a", newline='') as csv_file:
                            csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val))
                    else:
                        print("Success!")
                    num_allocations /= 2
                    allocation_size *= 2

    ####################################################################################################
    ####################################################################################################
    # Generate new Results
    ####################################################################################################
    ####################################################################################################
    #if generate_results:
    #    if not os.path.exists("results/performance/aggregate"):
    #        os.mkdir("results/performance/aggregate")
    #    generateResultsFromFileAllocation(testcases, "results/performance", num_allocations, smallest_allocation_size, largest_allocation_size, "Bytes", "perf", 2 if not args.warp else 3)
    
    ####################################################################################################
    ####################################################################################################
    # Generate new plots
    ####################################################################################################
    ####################################################################################################
    
    if generate_plots:
       result_alloc = list(list())
       result_free = list(list())
       # Get Timestring
       now = datetime.now()
       time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

       if plotscale == "log":
           time_string += "_log"
       else:
           time_string += "_lin"

       for file in os.listdir("results/performance/aggregate"):
           filename = str("results/performance/aggregate/") + os.fsdecode(file)
           if(os.path.isdir(filename)):
               continue
           if filename.split("_")[2] != "perf" or str(num_allocations) != filename.split('_')[4] or str(smallest_allocation_size) + "-" + str(largest_allocation_size) != filename.split('_')[5].split(".")[0]:
               continue
           # We want the one matching our input
           with open(filename) as f:
               reader = csv.reader(f)
               if "free" in filename:
                   result_free = list(reader)
               else:
                   result_alloc = list(reader)

       ####################################################################################################
       # Alloc - Mean - Std-dev
       ####################################################################################################
       plotMean(result_alloc, 
           testcases,
           plotscale,
           False, 
           'Bytes', 
           'ms', 
           "Allocation performance for " + str(num_allocations) + " allocations (mean)", 
           str("results/plots/performance/") + time_string + "_alloc." + filetype,
           "stddev")
       print("---------------------------------------")
       plotMean(result_alloc, 
           testcases,
           plotscale,
           True, 
           'Bytes', 
           'ms', 
           "Allocation performance for " + str(num_allocations) + " allocations (mean + std-dev)", 
           str("results/plots/performance/") + time_string + "_alloc_stddev." + filetype,
           "stddev")
       print("---------------------------------------")
       ####################################################################################################
       # Free - Mean - Std-dev
       ####################################################################################################
       plotMean(result_free, 
           testcases,
           plotscale,
           False,
           'Bytes', 
           'ms', 
           "Free performance for " + str(num_allocations) + " allocations (mean)", 
           str("results/plots/performance/") + time_string + "_free." + filetype,
           "stddev")
       print("---------------------------------------")
       plotMean(result_free, 
           testcases,
           plotscale,
           True,
           'Bytes', 
           'ms', 
           "Free performance for " + str(num_allocations) + " allocations (mean + std-dev)", 
           str("results/plots/performance/") + time_string + "_free_stddev." + filetype,
           "stddev")
       print("---------------------------------------")
       ####################################################################################################
       # Alloc - Mean - Min/Max
       ####################################################################################################
       plotMean(result_alloc, 
           testcases,
           plotscale,
           True,
           'Bytes', 
           'ms', 
           "Allocation performance for " + str(num_allocations) + " allocations (mean + min/max)", 
           str("results/plots/performance/") + time_string + "_alloc_min_max." + filetype,
           "minmax")
       print("---------------------------------------")
       ####################################################################################################
       # Free - Mean - Min/Max
       ####################################################################################################
       plotMean(result_free, 
           testcases,
           plotscale,
           True,
           'Bytes', 
           'ms', 
           "Free performance for " + str(num_allocations) + " allocations (mean + min/max)", 
           str("results/plots/performance/") + time_string + "_free_min_max." + filetype,
           "minmax")
       print("---------------------------------------")
       ####################################################################################################
       # Alloc - Median
       ####################################################################################################
       plotMean(result_alloc, 
           testcases,
           plotscale,
           False,
           'Bytes', 
           'ms', 
           "Allocation performance for " + str(num_allocations) + " allocations (median)", 
           str("results/plots/performance/") + time_string + "_alloc_median." + filetype,
           "median")
       print("---------------------------------------")
       ####################################################################################################
       # Free - Median
       ####################################################################################################
       plotMean(result_free, 
           testcases,
           plotscale,
           False,
           'Bytes', 
           'ms', 
           "Free performance for " + str(num_allocations) + " allocations (median)", 
           str("results/plots/performance/") + time_string + "_free_median." + filetype,
           "median")
       print("---------------------------------------")

    ####################################################################################################
    ####################################################################################################
    # Clean temporary files
    ####################################################################################################
    ####################################################################################################
    if clean_temporary_files:
       print("Do you REALLY want to delete all temporary files?:")
       inputfromconsole = input()
       if not (inputfromconsole == "yes" or inputfromconsole == "y"):
           exit(-1)
       for file in os.listdir("results/tmp"):
           filename = str("results/tmp/") + os.fsdecode(file)
           if(os.path.isdir(filename)):
               continue
           os.remove(filename)
       for file in os.listdir("results/tmp/aggregate"):
           filename = str("results/tmp/aggregate/") + os.fsdecode(file)
           if(os.path.isdir(filename)):
               continue
           os.remove(filename)
       for file in os.listdir("results/plots"):
           filename = str("results/plots/") + os.fsdecode(file)
           if(os.path.isdir(filename)):
               continue
           os.remove(filename)
    if args.mailpass:
        message = "Test Allocation finished!"
        print("no mail to be send")
        #mailalert.sendAlert(message)
    print("Done")

if __name__ == "__main__":
    main()
