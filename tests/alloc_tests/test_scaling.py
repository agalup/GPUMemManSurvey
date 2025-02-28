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

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_scaling.py -h")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = {}
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	smallest_num_threads = 2 ** 0
	largest_num_threads = 2 ** 10
	num_iterations = 25
	filetype = "pdf"
	time_out_val = 10
	free_memory = 1
	if os.name == 'nt': # If on Windows
		build_path = os.path.join("build", "Release")
		sync_build_path = os.path.join("sync_build", "Release")
	else:
		build_path = "build/"
		sync_build_path = "sync_build/"

	parser = argparse.ArgumentParser(description='Test allocation performance for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-byterange', type=str, help='Specify Allocation Range, e.g. 16-8192')
	parser.add_argument('-threadrange', type=str, help='Specify number of threads, given as powers of two, e.g. 0-5 -> results in 1-32')
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
			testcases["Ouroboros-P-S"] = os.path.join(build_path, str("o_alloc_test_p") + executable_extension)
			testcases["Ouroboros-P-VA"] = os.path.join(build_path, str("o_alloc_test_vap") + executable_extension)
			testcases["Ouroboros-P-VL"] = os.path.join(build_path, str("o_alloc_test_vlp") + executable_extension)
			testcases["Ouroboros-C-S"] = os.path.join(build_path, str("o_alloc_test_c") + executable_extension)
			testcases["Ouroboros-C-VA"] = os.path.join(build_path, str("o_alloc_test_vac") + executable_extension)
			testcases["Ouroboros-C-VL"] = os.path.join(build_path, str("o_alloc_test_vlc") + executable_extension)
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = os.path.join(sync_build_path, str("f_alloc_test") + executable_extension)
		if any("r" in s for s in args.t):
			# testcases["RegEff-A"] = os.path.join(sync_build_path, str("r_alloc_test_a") + executable_extension)
			testcases["RegEff-AW"] = os.path.join(sync_build_path, str("r_alloc_test_aw") + executable_extension)
			testcases["RegEff-C"] = os.path.join(sync_build_path, str("r_alloc_test_c") + executable_extension)
			testcases["RegEff-CF"] = os.path.join(sync_build_path, str("r_alloc_test_cf") + executable_extension)
			testcases["RegEff-CM"] = os.path.join(sync_build_path, str("r_alloc_test_cm") + executable_extension)
			testcases["RegEff-CFM"] = os.path.join(sync_build_path, str("r_alloc_test_cfm") + executable_extension)
	
	# Parse allocation size
	if(args.byterange):
		selected_range = args.byterange.split('-')
		smallest_allocation_size = int(selected_range[0])
		largest_allocation_size = int(selected_range[1])

	# Parse range
	if(args.threadrange):
		selected_range = args.threadrange.split('-')
		smallest_num_threads = 2 ** int(selected_range[0])
		largest_num_threads = 2 ** int(selected_range[1])

	# Parse num iterations
	if(args.iter):
		num_iterations = args.iter

	# Generate results
	if args.warp:
		test_warp_based = 1
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

	if measure_on_device and test_warp_based:
		print("Cannot measure on device and warp-based at the same time!")
		exit(-1)

	# Timeout (in seconds)
	if(args.timeout):
		time_out_val = args.timeout
	
	if(args.filetype):
		filetype = args.filetype

	####################################################################################################
	####################################################################################################
	# Run testcases
	####################################################################################################
	####################################################################################################
	if run_testcases:
		allocation_size = smallest_allocation_size
		while allocation_size <= largest_allocation_size:
			for name, path in testcases.items():
				csv_path_alloc = "results/scaling/scale_alloc_" + name + "_" + str(allocation_size) + "_" + str(smallest_num_threads) + "-" + str(largest_num_threads) + ".csv"
				csv_path_free = "results/scaling/scale_free_" + name + "_" + str(allocation_size) + "_" + str(smallest_num_threads) + "-" + str(largest_num_threads) + ".csv"
				if(os.path.isfile(csv_path_alloc)):
					print("This file already exists, do you really want to OVERWRITE?")
					inputfromconsole = input()
					if not (inputfromconsole == "yes" or inputfromconsole == "y"):
						continue
				with open(csv_path_alloc, "w", newline='') as csv_file:
					csv_file.write("NumThreads, mean, std-dev, min, max, median")
				with open(csv_path_free, "w", newline='') as csv_file:
					csv_file.write("NumThreads, mean, std-dev, min, max, median")
				num_threads = smallest_num_threads
				while num_threads <= largest_num_threads:
					with open(csv_path_alloc, "a", newline='') as csv_file:
						csv_file.write("\n" + str(num_threads) + ",")
					with open(csv_path_free, "a", newline='') as csv_file:
						csv_file.write("\n" + str(num_threads) + ",")
					run_config = str(num_threads) + " " + str(allocation_size) + " " + str(num_iterations) + " " + str(measure_on_device) + " " + str(test_warp_based) + " 1 " + str(free_memory) + " " + csv_path_alloc + " " + csv_path_free + " " + str(args.allocsize) + " " + str(args.device)
					executecommand = "{0} {1}".format(path, run_config)
					print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
					print("Running " + name + " with command -> " + executecommand)
					print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
					_, process_killed = Command(executecommand).run(timeout=time_out_val)
					if process_killed :
						print("We killed the process!")
						with open(csv_path_alloc, "a", newline='') as csv_file:
							csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val * 1000))
						with open(csv_path_free, "a", newline='') as csv_file:
							csv_file.write("0.00,0.00,0.00,0.00,0.00,-------------------> Ran longer than " + str(time_out_val * 1000))
					num_threads *= 2
			allocation_size *= 2

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		if not os.path.exists("results/scaling/aggregate"):
			os.mkdir("results/scaling/aggregate")
		allocation_size = smallest_allocation_size
		while allocation_size <= largest_allocation_size:
			generateResultsFromFileAllocation(testcases, "results/scaling", allocation_size, smallest_num_threads, largest_num_threads, "Threads", "scale", 2)
			allocation_size *= 2

	# ####################################################################################################
	# ####################################################################################################
	# # Generate plots
	# ####################################################################################################
	# ####################################################################################################
	# if generate_plots:
	# 	# Get Timestring
	# 	now = datetime.now()
	# 	time_string = now.strftime("%b-%d-%Y_%H-%M-%S")
	# 	if plotscale == "log":
	# 		time_string += "_log"
	# 	else:
	# 		time_string += "_lin"
	# 	# Generate plots for each byte size
	# 	byte_range = []
	# 	start_val = smallest_allocation_size
	# 	while start_val <= largest_allocation_size:
	# 		byte_range.append(start_val)
	# 		start_val *= 2

	# 	for num_bytes in byte_range:
	# 		# Generate plots for this byte size
	# 		print("Generate plots for size: " + str(num_bytes) + " Bytes")
	# 		result_alloc = list(list())
	# 		result_free = list(list())			

	# 		for file in os.listdir("results/scaling/aggregate"):
	# 			filename = str("results/scaling/aggregate/") + os.fsdecode(file)
	# 			if(os.path.isdir(filename)):
	# 				continue
	# 			if filename.split("_")[2] != "scale" or str(num_bytes) != filename.split('_')[4] or str(smallest_num_threads) + "-" + str(largest_num_threads) != filename.split('_')[5].split(".")[0]:
	# 				continue
	# 			# We want the one matching our input
	# 			with open(filename) as f:
	# 				reader = csv.reader(f)
	# 				if "free" in filename:
	# 					result_free = list(reader)
	# 				else:
	# 					result_alloc = list(reader)

	# 		####################################################################################################
	# 		# Alloc - Mean - Std-dev
	# 		####################################################################################################
	# 		print("Generate mean/stddev alloc plot for " + str(num_bytes))
	# 		plotMean(result_alloc, 
	# 			testcases,
	# 			plotscale,
	# 			False,
	# 			'Threads', 
	# 			'ms', 
	# 			"Allocation Scaling for " + str(num_bytes) + " Bytes (mean)", 
	# 			str("results/plots/scaling/") + time_string + "_alloc_scale_" + str(num_bytes) + "_mean." + filetype,
	# 			"stddev")
	# 		plotMean(result_alloc, 
	# 			testcases,
	# 			plotscale,
	# 			True,
	# 			'Threads', 
	# 			'ms', 
	# 			"Allocation Scaling for " + str(num_bytes) + " Bytes (mean + std-dev)", 
	# 			str("results/plots/scaling/") + time_string + "_alloc_scale_" + str(num_bytes) + "_mean_stddev." + filetype,
	# 			"stddev")

	# 		####################################################################################################
	# 		# Free - Mean - Std-dev
	# 		####################################################################################################
	# 		print("Generate mean/stddev free plot for " + str(num_bytes))
	# 		plotMean(result_free, 
	# 			testcases,
	# 			plotscale,
	# 			False,
	# 			'Threads', 
	# 			'ms', 
	# 			"Free scaling for " + str(num_bytes) + " Bytes (mean)", 
	# 			str("results/plots/scaling/") + time_string + "_free_scale_" + str(num_bytes) + "_mean." + filetype,
	# 			"stddev")
	# 		plotMean(result_free, 
	# 			testcases,
	# 			plotscale,
	# 			True,
	# 			'Threads', 
	# 			'ms', 
	# 			"Free scaling for " + str(num_bytes) + " Bytes (mean + std-dev)", 
	# 			str("results/plots/scaling/") + time_string + "_free_scale_" + str(num_bytes) + "_mean_stddev." + filetype,
	# 			"stddev")

	# 		####################################################################################################
	# 		# Alloc - Mean - Min/Max
	# 		####################################################################################################
	# 		print("Generate mean/min/max alloc plot for " + str(num_bytes))
	# 		plotMean(result_alloc, 
	# 			testcases,
	# 			plotscale,
	# 			True,
	# 			'Threads', 
	# 			'ms', 
	# 			"Allocation scaling for " + str(num_bytes) + " Bytes (mean + min/max)", 
	# 			str("results/plots/scaling/") + time_string + "_alloc_scale_" + str(num_bytes) + "_min_max." + filetype,
	# 			"minmax")

	# 		####################################################################################################
	# 		# Free - Mean - Min/Max
	# 		####################################################################################################
	# 		print("Generate mean/min/max free plot for " + str(num_bytes))
	# 		plotMean(result_free, 
	# 			testcases,
	# 			plotscale,
	# 			True,
	# 			'Threads', 
	# 			'ms', 
	# 			"Free scaling for " + str(num_bytes) + " Bytes (mean + min/max)", 
	# 			str("results/plots/scaling/") + time_string + "_free_scale_" + str(num_bytes) + "_min_max." + filetype,
	# 			"minmax")

	# 		####################################################################################################
	# 		# Alloc - Median
	# 		####################################################################################################
	# 		print("Generate median alloc plot for " + str(num_bytes))
	# 		plotMean(result_alloc, 
	# 			testcases,
	# 			plotscale, 
	# 			False,
	# 			'Threads', 
	# 			'ms', 
	# 			"Allocation scaling for " + str(num_bytes) + " Bytes (median)", 
	# 			str("results/plots/scaling/") + time_string + "_alloc_scale_" + str(num_bytes) + "_median." + filetype,
	# 			"median")

	# 		####################################################################################################
	# 		# Free - Median
	# 		####################################################################################################
	# 		print("Generate median free plot for " + str(num_bytes))
	# 		plotMean(result_free, 
	# 			testcases,
	# 			plotscale, 
	# 			False,
	# 			'Threads', 
	# 			'ms', 
	# 			"Free scaling for " + str(num_bytes) + " Bytes (median)", 
	# 			str("results/plots/scaling/") + time_string + "_free_scale_" + str(num_bytes) + "_median." + filetype,
	# 			"median")


	# ####################################################################################################
	# ####################################################################################################
	# # Clean temporary files
	# ####################################################################################################
	# ####################################################################################################
	# if clean_temporary_files:
	# 	print("Do you REALLY want to delete all temporary files?:")
	# 	inputfromconsole = input()
	# 	if not (inputfromconsole == "yes" or inputfromconsole == "y"):
	# 		exit(-1)
	# 	for file in os.listdir("results/tmp"):
	# 		filename = str("results/tmp/") + os.fsdecode(file)
	# 		if(os.path.isdir(filename)):
	# 			continue
	# 		os.remove(filename)
	# 	for file in os.listdir("results/tmp/aggregate"):
	# 		filename = str("results/tmp/aggregate/") + os.fsdecode(file)
	# 		if(os.path.isdir(filename)):
	# 			continue
	# 		os.remove(filename)
	# 	for file in os.listdir("results/plots"):
	# 		filename = str("results/plots/") + os.fsdecode(file)
	# 		if(os.path.isdir(filename)):
	# 			continue
	# 		os.remove(filename)

	print("Done")

if __name__ == "__main__":
	main()