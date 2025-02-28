import time
import os
from datetime import datetime
#from os import PathLike
import csv
import math
# import numpy as np
# import matplotlib.pyplot as plt

colours = {
	'ActualSize' : 'black',
	'BaseLine' : 'black',
	'Halloc' : 'orange' , 
    'XMalloc' : 'silver',
	'Ouroboros-P-VA' : 'lightcoral' , 'Ouroboros-P-VL' : 'darkred' , 'Ouroboros-P-S' : 'red' ,
	'Ouroboros-C-VA' : 'deepskyblue' , 'Ouroboros-C-VL' : 'royalblue' , 'Ouroboros-C-S' : 'navy' ,
	'CUDA' : 'green' , 
	'ScatterAlloc' : 'blue' , 
	'FDGMalloc' : 'gold' , 
	'RegEff-A' : 'mediumvioletred' , 'RegEff-AW' : 'orchid',
	'RegEff-C' : 'purple' , 'RegEff-CF' : 'violet' , 'RegEff-CM' : 'indigo' , 'RegEff-CFM' : 'blueviolet'
}

linestyles = {
	'ActualSize' : 'solid',
	'BaseLine' : 'solid',
	'Halloc' : 'solid' , 
    'XMalloc' : 'solid',
	'Ouroboros-P-VA' : 'dotted' , 'Ouroboros-P-VL' : 'dashed' , 'Ouroboros-P-S' : 'solid' ,
	'Ouroboros-C-VA' : 'dotted' , 'Ouroboros-C-VL' : 'dashed' , 'Ouroboros-C-S' : 'solid' ,
	'CUDA' : 'solid' , 
	'ScatterAlloc' : 'solid' , 
	'FDGMalloc' : 'solid' , 
	'RegEff-A' : 'solid' , 'RegEff-AW' : 'dashed',
	'RegEff-C' : 'solid' , 'RegEff-CF' : 'dashed' , 'RegEff-CM' : 'dotted' , 'RegEff-CFM' : 'dashed'
}

lineplot_width = 20
lineplot_height = 10
barplot_width = 15
barplot_height = 10

####################################################################################################
####################################################################################################
# Generate new Results
####################################################################################################
####################################################################################################
def generateResultsFromFileAllocation(testcases, folderpath, param1, param2, param3, dimension_name, output_name_short, approach_pos):
	print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
	# Gather results
	result_alloc = list(list())
	result_free = list(list())

	# Go over files, read data and generate new
	written_header_free = False
	written_header_alloc = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		#filename = folderpath + str("/") + os.PathLike(file)
		if(os.path.isdir(filename)):
			continue
		if str(param1) != filename.split('_')[approach_pos+1] or str(param2) + "-" + str(param3) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		approach_name = filename.split('_')[approach_pos]
		if approach_name not in testcases:
			continue
		print("Processing -> " + str(filename))
		with open(filename, newline='') as csv_file:
			csvreader = list(csv.reader(csv_file, delimiter=',', quotechar='|'))
			cols = list()
			num_cols = 6
			for i in range(num_cols):
				cols.append([row[i] if len(row) > i else "" for row in csvreader])
			if "free" in filename:
				if not written_header_free:
					result_free.append(cols[0][1:])
					result_free[-1].insert(0, dimension_name)
					written_header_free = True
				result_free.append(cols[1][1:])
				result_free[-1].insert(0, approach_name + " - mean")
				result_free.append(cols[2][1:])
				result_free[-1].insert(0, approach_name + " - std_dev")
				result_free.append(cols[3][1:])
				result_free[-1].insert(0, approach_name + " - min")
				result_free.append(cols[4][1:])
				result_free[-1].insert(0, approach_name + " - max")
				result_free.append(cols[5][1:])
				result_free[-1].insert(0, approach_name + " - median")
			else:
				if not written_header_alloc:
					result_alloc.append(cols[0][1:])
					result_alloc[-1].insert(0, dimension_name)
					written_header_alloc = True
				result_alloc.append(cols[1][1:])
				result_alloc[-1].insert(0, approach_name + " - mean")
				result_alloc.append(cols[2][1:])
				result_alloc[-1].insert(0, approach_name + " - std_dev")
				result_alloc.append(cols[3][1:])
				result_alloc[-1].insert(0, approach_name + " - min")
				result_alloc.append(cols[4][1:])
				result_alloc[-1].insert(0, approach_name + " - max")
				result_alloc.append(cols[5][1:])
				result_alloc[-1].insert(0, approach_name + " - median")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_") + output_name_short + str("_alloc_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	alloc_name = folderpath + str("/aggregate/") + time_string + str("_") + output_name_short + str("_alloc_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(alloc_name, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_alloc:
			writer.writerow(row)

	print("Generating -> " + time_string + str("_") + output_name_short + str("_free_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	free_name = folderpath + str("/aggregate/")  + time_string + str("_") + output_name_short + str("_free_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(free_name, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_free:
			writer.writerow(row)
	print("####################")

def generateResultsFromFileFragmentation(folderpath, param1, param2, param3, dimension_name, approach_pos, iter):
	print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
	# Gather results
	result_frag = list(list())

	# Go over files, read data and generate new
	written_header_frag = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str("frag") != filename.split('_')[0].split('/')[1] or str(param1) != filename.split('_')[approach_pos+1] or str(param2) + "-" + str(param3) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos]
		with open(filename, newline='') as csv_file:
			csvreader = list(csv.reader(csv_file, delimiter=',', quotechar='|'))
			if not written_header_frag:
				actual_size = [i for i in range(param2, param3 + 4, 4)]
				result_frag.append(list(actual_size))
				result_frag[-1].insert(0, dimension_name)
				actual_size = [i * param1 for i in range(param2, param3 + 4, 4)]
				result_frag.append(list(actual_size))
				result_frag[-1].insert(0, "ActualSize - range")
				result_frag.append(list(actual_size))
				result_frag[-1].insert(0, "ActualSize - static range")
				written_header_frag = True
			csvreader = csvreader[1:]
			result_frag.append([row[1] for row in csvreader])
			result_frag[-1].insert(0, approach_name + " - range")
			result_frag.append([row[-1] for row in csvreader])
			result_frag[-1].insert(0, approach_name + " - static range")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_frag_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	frag_name = folderpath + str("/aggregate/") + time_string +  str("_frag_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(frag_name, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_frag:
			writer.writerow(row)
	print("####################")

def generateResultsFromFileOOM(folderpath, testcases, param1, param2, param3, dimension_name, approach_pos, alloc_size):
	print("Generate Results for identifier " + str(param1) + "_" + str(param2) + "-" + str(param3))
	# Gather results
	result_oom = list(list())
	percent = 100

	max_rounds = []
	num = param2
	while num <= param3:
		max_rounds.append(int(alloc_size / (param1 * num)))
		num *= 2

	# Go over files, read data and generate new
	written_header = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str("oom") != filename.split('_')[0].split('/')[1] or str(param1) != filename.split('_')[approach_pos+1] or str(param2) + "-" + str(param3) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		approach_name = filename.split('_')[approach_pos]
		if approach_name not in testcases:
			continue
		print("Processing -> " + str(filename))
		with open(filename, newline='') as csv_file:
			csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
			if not written_header:
				header = []
				best_case = []
				num = param2
				while num <= param3:
					header.append(num)
					best_case.append(percent)
					num *= 2
				result_oom.append(list(header))
				result_oom[-1].insert(0, dimension_name)
				result_oom.append(list(best_case))
				result_oom[-1].insert(0, "BaseLine")
				written_header = True
			approach_rounds = [len(row)-1 for row in csvreader]
			approach_rounds = approach_rounds[1:]
			result = [(b / a)*percent for a,b in zip(max_rounds, approach_rounds)]
			result_oom.append(list(result))
			result_oom[-1].insert(0, approach_name)

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_oom_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv"))
	frag_name = folderpath + str("/aggregate/") + time_string +  str("_oom_") + str(param1) + "_" + str(param2) + "-" + str(param3) + str(".csv")
	with(open(frag_name, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_oom:
			writer.writerow(row)
	print("####################")

def generateResultsFromFileInit(folderpath, param1, dimension_name, approach_pos):
	print("Generate Results for identifier " + str(param1))
	# Gather results
	result_init = list(list())

	# Go over files, read data and generate new
	written_header_init = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str("init") != filename.split('_')[0].split("/")[1]:
			continue
		if str(param1) != filename.split('_')[approach_pos+1].split(".")[0]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos]
		with open(filename, newline='') as csv_file:
			csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
			if not written_header_init:
				result_init.append(["Approach", "Alloc Size", "Timing (ms) GPU", "Timing (ms) CPU"])
				written_header_init = True
			csvreader = list(csvreader)
			result_init.append(list(csvreader[1]))
			result_init[-1].insert(0, approach_name)

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_init_") + str(param1) + str(".csv"))
	init_name = folderpath + str("/aggregate/") + time_string +  str("_init_") + str(param1) + str(".csv")
	with(open(init_name, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_init:
			writer.writerow(row)
	print("####################")

def generateResultsFromFileRegisters(folderpath, dimension_name, approach_pos):
	# Gather results
	result_init = list(list())

	# Go over files, read data and generate new
	written_header_init = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str("reg") != filename.split('_')[0].split("/")[1]:
			continue
		print("Processing -> " + str(filename))
		approach_name = filename.split('_')[approach_pos].split(".")[0]
		with open(filename, newline='') as csv_file:
			csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
			if not written_header_init:
				result_init.append(["Approach", "Alloc Size", "Timing (ms) GPU", "Timing (ms) CPU"])
				written_header_init = True
			csvreader = list(csvreader)
			result_init.append(list(csvreader[1]))
			result_init[-1].insert(0, approach_name)

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	print("Generating -> " + time_string + str("_reg") + str(".csv"))
	init_name = folderpath + str("/aggregate/") + time_string +  str("_reg") + str(".csv")
	with(open(init_name, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in result_init:
			writer.writerow(row)
	print("####################")

def generateResultsFromSynthetic(testcases, folderpath, smallThread, largeThread, smallByte, largeByte, dimension_name, output_name_short, approach_pos, generateFull=False):
	print("Generate Results for identifier " + str(smallThread) + "-" + str(largeThread) + "_" + str(smallByte) + "-" + str(largeByte))
	# Gather results
	results = list(list())

	# Go over files, read data and generate new
	written_header = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if output_name_short == "synth":
			if str("synth") != filename.split('_')[0].split('/')[1]:
				continue
		else:
			if str("synth") != filename.split('_')[0].split('/')[1] or str("write") != filename.split('_')[1]:
				continue
		if str(smallThread) + "-" + str(largeThread) != filename.split('_')[approach_pos+1] or str(smallByte) + "-" + str(largeByte) != filename.split('_')[approach_pos+2].split(".")[0]:
			continue
		approach_name = filename.split('_')[approach_pos]
		if approach_name not in testcases:
			continue
		print("Processing -> " + str(filename))
		
		with open(filename, newline='') as csv_file:
			csvreader = list(csv.reader(csv_file, delimiter=',', quotechar='|'))
			cols = list()
			num_cols = 6
			for i in range(num_cols):
				cols.append([row[i] if len(row) > i else "" for row in csvreader])
			if not written_header:
				results.append(cols[0][1:])
				results[-1].insert(0, dimension_name)
				written_header = True
			results.append(cols[1][1:])
			results[-1].insert(0, approach_name + (" - mean") if generateFull else approach_name)
			if generateFull:
				results.append(cols[2][1:])
				results[-1].insert(0, approach_name + " - std_dev")
				results.append(cols[3][1:])
				results[-1].insert(0, approach_name + " - min")
				results.append(cols[4][1:])
				results[-1].insert(0, approach_name + " - max")
				results.append(cols[5][1:])
				results[-1].insert(0, approach_name + " - median")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	outputstr = time_string + str("_") + output_name_short + str("_") + str(smallThread) + "-" + str(largeThread) + "_" + str(smallByte) + "-" + str(largeByte) + str(".csv")
	print("Generating -> " + outputstr)
	filename = folderpath + str("/aggregate/") + outputstr
	with(open(filename, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in results:
			writer.writerow(row)

	print("####################")

def generateResultsFromGraph(testcases, folderpath, dimension_name, output_name_short, approach_pos, generateFull=False):
	print("Generate Results for graph " + output_name_short)
	# Gather results
	results = list(list())

	# Go over files, read data and generate new
	written_header = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str("init") != filename.split('_')[1]:
			continue
		approach_name = filename.split('_')[approach_pos].split(".")[0]
		if approach_name not in testcases:
			continue
		print("Processing -> " + str(filename))
		
		with open(filename, newline='') as csv_file:
			csvreader = list(csv.reader(csv_file, delimiter=',', quotechar='|'))
			cols = list()
			num_cols = 6
			for i in range(num_cols):
				cols.append([row[i] if len(row) > i else "" for row in csvreader])
			if not written_header:
				results.append(cols[0][1:])
				results[-1].insert(0, dimension_name)
				written_header = True
			results.append(cols[1][1:])
			results[-1].insert(0, approach_name + (" - mean") if generateFull else approach_name)
			if generateFull:
				results.append(cols[2][1:])
				results[-1].insert(0, approach_name + " - std_dev")
				results.append(cols[3][1:])
				results[-1].insert(0, approach_name + " - min")
				results.append(cols[4][1:])
				results[-1].insert(0, approach_name + " - max")
				results.append(cols[5][1:])
				results[-1].insert(0, approach_name + " - median")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	outputstr = time_string + str("_graph_") + output_name_short + str(".csv")
	print("Generating -> " + outputstr)
	filename = folderpath + str("/aggregate/") + outputstr
	with(open(filename, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in results:
			writer.writerow(row)

	print("####################")

def generateResultsFromGraphUpdate(testcases, folderpath, dimension_name, output_name_short, approach_pos, ranged, generateFull=False):
	print("Generate Results for graph " + output_name_short)
	# Gather results
	results_insert = list(list())
	results_delete = list(list())

	# Go over files, read data and generate new
	written_header_insert = False
	written_header_delete = False
	for file in os.listdir(folderpath):
		filename = folderpath + str("/") + os.fsdecode(file)
		if(os.path.isdir(filename)):
			continue
		if str("update") != filename.split('_')[1]:
				continue
		if ranged:
			if str("range") != filename.split('_')[-1].split(".")[0]:
				continue
		else:
			if str("range") == filename.split('_')[-1].split(".")[0]:
				continue
		approach_name = filename.split('_')[approach_pos].split(".")[0]
		if approach_name not in testcases:
			continue
		print("Processing -> " + str(filename))
		
		with open(filename, newline='') as csv_file:
			csvreader = list(csv.reader(csv_file, delimiter=',', quotechar='|'))
			cols = list()
			num_cols = 6
			for i in range(num_cols):
				cols.append([row[i] if len(row) > i else "" for row in csvreader])
			if "delete" in filename:
				if not written_header_delete:
					results_delete.append(cols[0][1:])
					results_delete[-1].insert(0, dimension_name)
					written_header_delete = True
				results_delete.append(cols[1][1:])
				results_delete[-1].insert(0, approach_name + (" - mean") if generateFull else approach_name)
				if generateFull:
					results_delete.append(cols[2][1:])
					results_delete[-1].insert(0, approach_name + " - std_dev")
					results_delete.append(cols[3][1:])
					results_delete[-1].insert(0, approach_name + " - min")
					results_delete.append(cols[4][1:])
					results_delete[-1].insert(0, approach_name + " - max")
					results_delete.append(cols[5][1:])
					results_delete[-1].insert(0, approach_name + " - median")
			else:
				if not written_header_insert:
					results_insert.append(cols[0][1:])
					results_insert[-1].insert(0, dimension_name)
					written_header_insert = True
				results_insert.append(cols[1][1:])
				results_insert[-1].insert(0, approach_name + (" - mean") if generateFull else approach_name)
				if generateFull:
					results_insert.append(cols[2][1:])
					results_insert[-1].insert(0, approach_name + " - std_dev")
					results_insert.append(cols[3][1:])
					results_insert[-1].insert(0, approach_name + " - min")
					results_insert.append(cols[4][1:])
					results_insert[-1].insert(0, approach_name + " - max")
					results_insert.append(cols[5][1:])
					results_insert[-1].insert(0, approach_name + " - median")

	# Get Timestring
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	# Generate output file
	print("------------------")
	suffix = ".csv"
	if ranged:
		suffix = "_range.csv"
	outputstr = time_string + str("_graph_") + output_name_short + str("_insert") + suffix
	print("Generating -> " + outputstr)
	filename = folderpath + str("/aggregate/") + outputstr
	with(open(filename, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in results_insert:
			writer.writerow(row)

	outputstr = time_string + str("_graph_") + output_name_short + str("_delete") + suffix
	print("Generating -> " + outputstr)
	filename = folderpath + str("/aggregate/") + outputstr
	with(open(filename, "w", newline='')) as f:
		writer = csv.writer(f, delimiter=',')
		for row in results_delete:
			writer.writerow(row)

	print("####################")


####################################################################################################
####################################################################################################
# Plot disabled for now
####################################################################################################
####################################################################################################

std_dev_offset = 1
min_offset = 2
max_offset = 3
median_offset = 4

# # Plot mean as a line plot with std-dev
# def plotMean(results, testcases, plotscale, plotrange, xlabel, ylabel, title, filename, variant):
# 	plt.figure(figsize=(lineplot_width, lineplot_height))
# 	x_values = np.asarray([float(i) for i in results[0][1:]])
# 	y_values = dict()
# 	y_min = dict()
# 	y_max = dict()
# 	y_stddev = dict()
# 	for i in range(1, len(results), 5):
# 		labelname = results[i][0].split(" ")[0]
# 		if labelname not in testcases:
# 			continue
# 		print("Generate plot for " + results[i][0] + " with " + variant)
# 		y = None
# 		if variant == "median":
# 			y = np.asarray([float(i) for i in results[i+median_offset][1:]])
# 			y_values[labelname] = y
# 		else:
# 			y = np.asarray([float(i) for i in results[i][1:]])
# 			y_values[labelname] = y
# 		if variant == "stddev":
# 			y_stddev = np.asarray([float(i) for i in results[i+std_dev_offset][1:]])
# 			y_min[labelname] = y - y_stddev
# 			y_max[labelname] = y + y_stddev
# 		else:
# 			y_min[labelname] = np.asarray([float(i) for i in results[i+min_offset][1:]])
# 			y_max[labelname] = np.asarray([float(i) for i in results[i+max_offset][1:]])

# 	y_values = sorted(y_values.items())
# 	y_min = sorted(y_min.items())
# 	y_max = sorted(y_max.items())

# 	for key, value in y_values:
# 		plt.plot(x_values, value, marker='', color=colours[key], linewidth=1, label=key, linestyle=linestyles[key])
# 	if plotrange:
# 		for key, value in y_values:
# 			min_values = y_min[key]
# 			max_values = y_max[key]
# 			plt.fill_between(x_values, min_values, max_values, alpha=0.5, edgecolor=colours[key], facecolor=colours[key])
# 	if plotscale == "log":
# 		plt.yscale("log")
# 	plt.ylabel(ylabel)
# 	plt.xlabel(xlabel)
# 	plt.title(title)
# 	plt.legend()
# 	plt.savefig(filename, dpi=600)

# 	# Clear Figure
# 	plt.clf()

# # Plot results as a bar plot with std-dev
# def plotBars(results, testcases, plotscale, plotrange, xlabel, ylabel, title, filename, variant):
# 	plt.figure(figsize=(barplot_width, barplot_height))
# 	num_approaches = int(len(results) / 5)
# 	width = 0.9 / num_approaches
# 	index = np.arange(len(results[0][1:]))
# 	placement = []
# 	alignlabel = ''
# 	approach_half = int(math.floor(num_approaches/2))
# 	error_offset = 0
# 	if num_approaches % 2 == 0:
# 		placement = [number - approach_half for number in range(0, num_approaches)]
# 		alignlabel = 'edge'
# 		error_offset = width / 2
# 	else:
# 		placement = [number - approach_half for number in range(0, num_approaches)]
# 		alignlabel = 'center'
# 	labels = []
# 	xticks = []
# 	for i in range(len(results[0][1:])):
# 		labels.append(results[0][1+i])
# 		xticks.append(index[i])
# 	x_values = np.asarray([str(i) for i in results[0][1:]])
# 	j = 0
# 	for i in range(1, len(results), 5):
# 		y_values = None
# 		if variant == "median":
# 			y_values = np.asarray([float(i) for i in results[i+median_offset][1:]])
# 		else:
# 			y_values = np.asarray([float(i) for i in results[i][1:]])
# 		y_min = None
# 		y_max = None
# 		if variant == "stddev":
# 			y_stddev = np.asarray([float(i) for i in results[i+std_dev_offset][1:]])
# 			y_min = y_values-y_stddev
# 			y_min = [max(val, 0) for val in y_min]
# 			y_max = y_values+y_stddev
# 		else:
# 			y_min = np.asarray([float(i) for i in results[i+min_offset][1:]])
# 			y_max = np.asarray([float(i) for i in results[i+max_offset][1:]])
# 		labelname = results[i][0].split(" ")[0]
# 		if labelname not in testcases:
# 			continue
# 		yerror = np.array([y_min,y_max])
# 		outputstring = "Generate plot for " + labelname
# 		if plotrange:
# 			outputstring += " with " + variant
# 		print(outputstring)
# 		plt.bar(index + (placement[j] * width), y_values, width=width, color=colours[labelname], align=alignlabel, edgecolor = "black", label=labelname, tick_label=x_values)
# 		if plotrange:
# 			plt.errorbar(index + (placement[j] * width) + error_offset, y_values, yerror, fmt='r^')
# 		j += 1
# 	if plotscale == "log":
# 		plt.yscale("log")
# 	plt.ylabel(ylabel)
# 	plt.xlabel(xlabel)
# 	plt.xticks(xticks)
# 	plt.tick_params(axis='x', which='major', labelsize=6)
# 	plt.tick_params(axis='y', which='major', labelsize=12)
# 	plt.title(title)
# 	plt.legend()
# 	plt.savefig(filename, dpi=600)		

# 	# Clear Figure
# 	plt.clf()


# # Lineplot
# def plotLine(results, testcases, plotscale, xlabel, ylabel, title, filename, xscale="linear"):
# 	plt.figure(figsize=(lineplot_width, lineplot_height))
# 	x_values = np.asarray([float(i) for i in results[0][1:]])
# 	results = results[1:]
# 	results.sort(key=lambda x: x[0])
# 	for i in range(0, len(results)):
# 		y_values = np.asarray([float(i) for i in results[i][1:]])
# 		labelname = results[i][0].split(" ")[0]
# 		print(labelname)
# 		if labelname not in testcases and labelname != "ActualSize":
# 			continue
# 		print("Generate plot for " + labelname)
# 		plt.plot(x_values, y_values, marker='', color=colours[labelname], linewidth=1, label=labelname, linestyle=linestyles[labelname])
# 	if plotscale == "log":
# 		plt.yscale("log")
# 	plt.xscale(xscale, base=2)
# 	plt.ylabel(ylabel)
# 	plt.xlabel(xlabel)
# 	plt.title(title)
# 	plt.legend()
# 	plt.savefig(filename, dpi=600)

# 	# Clear Figure
# 	plt.clf()

# # Lineplot with range
# def plotLineRange(results, testcases, plotscale, plotrange, xlabel, ylabel, title, filename):
# 	plt.figure(figsize=(lineplot_width, lineplot_height))
# 	x_values = np.asarray([float(i) for i in results[0][1:]])
# 	results = results[1:]
# 	results.sort(key=lambda x: x[0])
# 	for i in range(0, len(results), 2):
# 		y_values = np.asarray([float(i) for i in results[i][1:]])
# 		labelname = results[i][0].split(" ")[0]
# 		if labelname not in testcases and labelname != "ActualSize":
# 			continue
# 		print("Generate plot for " + labelname)
# 		plt.plot(x_values, y_values, marker='', color=colours[labelname], linewidth=1, label=labelname, linestyle=linestyles[labelname])
# 		if plotrange:
# 			y_max = np.asarray([float(i) for i in results[i+1][1:]])
# 			plt.fill_between(x_values, y_values, y_max, alpha=0.5, edgecolor=colours[labelname], facecolor=colours[labelname])
# 	if plotscale == "log":
# 		plt.yscale("log")
# 	plt.ylabel(ylabel)
# 	plt.xlabel(xlabel)
# 	plt.title(title)
# 	plt.legend()
# 	plt.savefig(filename, dpi=600)

# 	# Clear Figure
# 	plt.clf()


# def plotInit(results, testcases, plotscale, offset, xlabel, ylabel, title, filename):
# 	plt.figure(figsize=(barplot_width, barplot_height))
# 	results = results[1:]
# 	results.sort(key=lambda x: x[0])
# 	num_approaches = len(results)
# 	width = 0.9
# 	index = np.arange(num_approaches)
# 	labels = []
# 	xticks = []
# 	for i in range(len(results[0][1:])):
# 		labels.append(results[0][1+i])
# 		xticks.append(index[i])
# 	x_values = np.asarray([str(i[0]) for i in results])
# 	y_values = np.asarray([float(i[offset]) for i in results])
# 	y_pos = np.arange(len(x_values))
# 	colour = [colours[i] for i in x_values]
# 	plt.bar(y_pos, y_values, width=width, color=colour, align='center', edgecolor = "black")
# 	if plotscale == "log":
# 		plt.yscale("log")
# 	plt.ylabel(ylabel)
# 	plt.xlabel(xlabel)
# 	plt.xticks(y_pos, x_values)
# 	plt.tick_params(axis='x', which='major', labelsize=6)
# 	plt.tick_params(axis='y', which='major', labelsize=12)
# 	plt.title(title)
# 	plt.savefig(filename, dpi=600)

# 	# Clear Figure
# 	plt.clf()


# def plotRegisters(results, testcases, plotscale, xlabel, ylabel, title, filename, grouping):
# 	plt.figure(figsize=(barplot_width, barplot_height))
# 	results = results[1:]
# 	results.sort(key=lambda x: x[0])
# 	num_approaches = len(results)
# 	if grouping == 'per_approach':
# 		width = 0.9 / 2
# 		index = np.arange(num_approaches)
# 		labels = []
# 		xticks = []
# 		for i in range(len(results[0][1:])):
# 			labels.append(results[0][1+i])
# 			xticks.append(index[i])
# 		x_values = np.asarray([str(i[0]) for i in results])
# 		y_values_malloc = np.asarray([float(i[1]) for i in results])
# 		y_values_free = np.asarray([float(i[2]) for i in results])
# 		y_pos = np.arange(len(x_values))
# 		colour = [colours[i] for i in x_values]

# 		plt.bar(y_pos - width, y_values_malloc, width=width, color=colour, align='edge', edgecolor = "black", label='malloc')
# 		plt.bar(y_pos, y_values_free, width=width, color=colour, align='edge', edgecolor = "black", label='free')
		
# 		plt.ylabel(ylabel)
# 		plt.xlabel(xlabel)
# 		plt.xticks(y_pos, x_values)
# 	else:
# 		placement = []
# 		alignlabel = ''
# 		approach_half = int(math.floor(num_approaches/2))
# 		if num_approaches % 2 == 0:
# 			placement = [number - approach_half for number in range(0, num_approaches)]
# 			alignlabel = 'edge'
# 		else:
# 			placement = [number - approach_half for number in range(0, num_approaches)]
# 			alignlabel = 'center'
# 		width = 0.9 / num_approaches
# 		index = np.arange(2)
# 		x_values = np.asarray([str('malloc'), str('free')])
# 		for approach_num in range(num_approaches):
# 			y_values = np.asarray([float(i) for i in results[approach_num][1:]])
# 			y_pos = np.arange(len(x_values))
# 			labelname = results[approach_num][0]
# 			plt.bar(index + (placement[approach_num] * width), y_values, width=width, color=colours[labelname], align=alignlabel, edgecolor = "black", label=labelname)
# 		plt.ylabel(ylabel)
# 		plt.xlabel(xlabel)
# 		plt.xticks(y_pos, x_values)

# 	if plotscale == "log":
# 		plt.yscale("log")
# 	plt.tick_params(axis='x', which='major', labelsize=6)
# 	plt.tick_params(axis='y', which='major', labelsize=12)
# 	plt.title(title)
# 	plt.legend()
# 	plt.savefig(filename, dpi=600)

# 	# Clear Figure
# 	plt.clf()

