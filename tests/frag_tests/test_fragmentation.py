import sys
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
from Helper import generateResultsFromFileFragmentation
from Helper import plotLineRange
import csv
import argparse
import numpy as np

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_fragmentation.py")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = {}
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	alloc_size = 8
	num_iterations = 1
	free_memory = 1
	generate_results = True
	generate_plots = True
	clean_temporary_files = True
	test_warp_based = False
	filetype = "pdf"
	time_out_val = 100
	build_path = "build/"
	sync_build_path = "sync_build/"

	parser = argparse.ArgumentParser(description='Test fragmentation for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r+x ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient | x : xmalloc')
	parser.add_argument('-num', type=int, help='How many allocations to perform')
	parser.add_argument('-range', type=str, help='Sepcify Allocation Range, e.g. 4-1024')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-runtest', action='store_true', default=False, help='Run testcases')
	parser.add_argument('-genres', action='store_true', default=False, help='Generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-timeout', type=int, help='Timeout Value in Seconds, process will be killed after as many seconds')
	parser.add_argument('-plotscale', type=str, help='log/linear')
	parser.add_argument('-filetype', type=str, help='png or pdf')
	parser.add_argument('-allocsize', type=int, help='How large is the manageable memory in GiB?', default=8)

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("c" in s for s in args.t):
			testcases["CUDA"] = build_path + str("c_frag_test")
		if any("x" in s for s in args.t):
			testcases["XMalloc"] = sync_build_path + str("x_frag_test")
		if any("h" in s for s in args.t):
			testcases["Halloc"] = sync_build_path + str("h_frag_test")
		if any("s" in s for s in args.t):
			testcases["ScatterAlloc"] = sync_build_path + str("s_frag_test")
		if any("o" in s for s in args.t):
			testcases["Ouroboros-P-S"] = build_path + str("o_frag_test_p")
			testcases["Ouroboros-P-VA"] = build_path + str("o_frag_test_vap")
			testcases["Ouroboros-P-VL"] = build_path + str("o_frag_test_vlp")
			testcases["Ouroboros-C-S"] = build_path + str("o_frag_test_c")
			testcases["Ouroboros-C-VA"] = build_path + str("o_frag_test_vac")
			testcases["Ouroboros-C-VL"] = build_path + str("o_frag_test_vlc")
		if any("f" in s for s in args.t):
			testcases["FDGMalloc"] = sync_build_path + str("f_frag_test")
		if any("r" in s for s in args.t):
			# testcases["RegEff-A"] = sync_build_path + str("r_frag_test_a")
			testcases["RegEff-AW"] = sync_build_path + str("r_frag_test_aw")
			testcases["RegEff-C"] = sync_build_path + str("r_frag_test_c")
			testcases["RegEff-CF"] = sync_build_path + str("r_frag_test_cf")
			testcases["RegEff-CM"] = sync_build_path + str("r_frag_test_cm")
			testcases["RegEff-CFM"] = sync_build_path + str("r_frag_test_cfm")
	
	# Parse num allocation
	if(args.num):
		num_allocations = args.num

	# Parse range
	if(args.range):
		selected_range = args.range.split('-')
		smallest_allocation_size = int(selected_range[0])
		largest_allocation_size = int(selected_range[1])
	
	# Parse num iterations
	if(args.iter):
		num_iterations = args.iter

	# Run Testcases
	run_testcases = args.runtest
	
	# Generate results
	generate_results = args.genres

	# Generate plots
	generate_plots = args.genplot

	# Plot Axis scaling
	plotscale = args.plotscale

	# Timeout (in seconds)
	if(args.timeout):
		time_out_val = args.timeout

	if(args.filetype):
		filetype = args.filetype

	if(args.allocsize):
		alloc_size = args.allocsize 

	####################################################################################################
	####################################################################################################
	# Run testcases
	####################################################################################################
	####################################################################################################
	if run_testcases:
		for name, executable in testcases.items():
			csv_path = "results/frag_" + name + "_" + str(num_allocations) + "_" + str(smallest_allocation_size) + "-" + str(largest_allocation_size) + ".csv"
			if(os.path.isfile(csv_path)):
				print("This file already exists, do you really want to OVERWRITE?")
				inputfromconsole = input()
				if not (inputfromconsole == "yes" or inputfromconsole == "y"):
					continue
			with open(csv_path, "w", newline='') as csv_file:
				csv_file.write("AllocationSize (in Byte)")
				for i in range(num_iterations):
					csv_file.write(",range, static range")
			allocation_size = smallest_allocation_size
			while allocation_size <= largest_allocation_size:
				with open(csv_path, "a", newline='') as csv_file:
					csv_file.write("\n" + str(allocation_size))
				run_config = str(num_allocations) + " " + str(allocation_size) + " " + str(num_iterations) + " 0 " + csv_path + " " + str(alloc_size)
				executecommand = "{0} {1}".format(executable, run_config)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print("Running " + name + " with command -> " + executecommand)
				print("#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#")
				print(executecommand)
				_, process_killed = Command(executecommand).run(timeout=time_out_val)
				if process_killed :
					print("We killed the process!")
					with open(csv_path, "a", newline='') as csv_file:
						csv_file.write(",0,0,-------------------> Ran longer than " + str(time_out_val))
				else:
					print("Success!")
				allocation_size += 4

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		if not os.path.exists("results/aggregate"):
			os.mkdir("results/aggregate")
		generateResultsFromFileFragmentation("results", num_allocations, smallest_allocation_size, largest_allocation_size, "Bytes", 1, num_iterations)

	####################################################################################################
	####################################################################################################
	# Generate new plots
	####################################################################################################
	####################################################################################################
	if generate_plots:
		result_frag = list()
		# Get Timestring
		now = datetime.now()
		time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

		if plotscale == "log":
			time_string += "_log"
		else:
			time_string += "_lin"

		for file in os.listdir("results/aggregate"):
			filename = str("results/aggregate/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			if filename.split("_")[2] != "frag" or str(num_allocations) != filename.split('_')[3] or str(smallest_allocation_size) + "-" + str(largest_allocation_size) != filename.split('_')[4].split(".")[0]:
				continue
			# We want the one matching our input
			with open(filename) as f:
				result_frag = list(csv.reader(f))

		####################################################################################################
		# Lineplot
		####################################################################################################
		plotLineRange(result_frag, 
			testcases,
			plotscale,
			False, 
			'Bytes', 
			'Byte - Range', 
			"Fragmentation: Byte-Range for " + str(num_allocations) + " allocations", 
			str("results/plots/") + time_string + "_frag." + filetype)

		####################################################################################################
		# Lineplot with range
		####################################################################################################
		plotLineRange(result_frag, 
			testcases,
			plotscale,
			True, 
			'Bytes', 
			'Byte - Range',
			"Fragmentation: Byte-Range for " + str(num_allocations) + " allocations", 
			str("results/plots/") + time_string + "_frag_range." + filetype)

	print("Done")

if __name__ == "__main__":
	main()