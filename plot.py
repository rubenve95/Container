import numpy as np
import matplotlib.pyplot as plt
import os
import json
import statistics
import yaml

def plot_inits(methods, results_folder='results_init'):
	base_path = os.path.join('data', results_folder, 'stats')
	files = os.listdir(base_path)
	files = [x for x in files if len(x.split('.')) > 1]
	results = {}
	for file in files:
		file_split = file.split('_')
		folder_name = "_".join(file_split[:2])
		method_name = ("_".join(file_split[2:])).split('.')[0]
		#print(folder_name, method_name)
		with open(os.path.join(base_path, file)) as f:
			if method_name not in results:
				results[method_name] = {}
			results[method_name][folder_name] = json.load(f)
	#print(results)

	#4 bar plots: each folder + average. Horizontally each method

	acc_mean = {}
	acc_std = {}
	miou_mean = {}
	miou_std = {}
	pck_mean = {}
	pck_std = {}
	#methods = ['random', 'generic', 'contour', 'gradient']
	acc_all, miou_all, pck_all = [],[],[]
	for i in range(len(methods)):
		acc_all.append([])
		miou_all.append([])
		pck_all.append([])
	#acc_all = [[]]*len(methods)
	#miou_all = [[]]*len(methods)
	#pck_all = [[]]*len(methods)
	#print(acc_all)
	for folder_name in results[methods[0]]:
		acc_mean[folder_name] = []
		acc_std[folder_name] = []
		miou_mean[folder_name] = []
		miou_std[folder_name] = []
		pck_mean[folder_name] = []
		pck_std[folder_name] = []
		for i,method_name in enumerate(methods):

			acc_list = [results[method_name][folder_name][x]['accuracy'] for x in results[method_name][folder_name]]
			miou_list = [results[method_name][folder_name][x]['miou'] for x in results[method_name][folder_name]]

			for a,m in zip(acc_list, miou_list):
				acc_all[i].append(a)
				miou_all[i].append(m)

			#print(method_name, len(pck_list), len(pck_all[i]))

			acc_mean[folder_name].append(statistics.mean(acc_list))
			acc_std[folder_name].append(statistics.stdev(acc_list))
			miou_mean[folder_name].append(statistics.mean(miou_list))
			miou_std[folder_name].append(statistics.stdev(miou_list))

			pck_list = []
			for x in results[method_name][folder_name]:
				pck = results[method_name][folder_name][x]['pck']
				#if pck != []:
				pck_list.extend(pck)
			#pck_list = [results[method_name][folder_name][x]['pck'] for x in results[method_name][folder_name]]
			if len(pck_list) == 0:
				pck_mean[folder_name].append('-')
				pck_std[folder_name].append('-')
			else:
				for p in pck_list:
					pck_all[i].append(p)
				pck_mean[folder_name].append(statistics.mean(pck_list))
				pck_std[folder_name].append(statistics.stdev(pck_list))

		base_fig_path = os.path.join("/".join(base_path.split('/')[:-1]), 'figures')
		if not os.path.exists(base_fig_path):
			os.mkdir(base_fig_path)
		#fig_path = os.path.join(base_fig_path, folder_name + '.png')
		#barplot(acc_mean[folder_name], miou_mean[folder_name], pck_mean[folder_name], methods, save_path=os.path.join(base_fig_path, folder_name.split('.')[0] + '_mean.png'), measurement='Mean')
		#barplot(acc_std[folder_name], miou_std[folder_name], pck_std[folder_name], methods, save_path=os.path.join(base_fig_path, folder_name.split('.')[0] + '_std.png'), measurement='Standard Deviation')

	acc_mean['all'] = []
	acc_std['all'] = []
	miou_mean['all'] = []
	miou_std['all'] = []
	pck_mean['all'] = []
	pck_std['all'] = []

	for i,method_name in enumerate(methods):
		acc_mean['all'].append(statistics.mean(acc_all[i]))
		acc_std['all'].append(statistics.stdev(acc_all[i]))
		miou_mean['all'].append(statistics.mean(miou_all[i]))
		miou_std['all'].append(statistics.stdev(miou_all[i]))
		if len(pck_all[i]) == 0:
			pck_mean['all'].append('-')
			pck_std['all'].append('-')
		else:
			pck_mean['all'].append(statistics.mean(pck_all[i]))
			pck_std['all'].append(statistics.stdev(pck_all[i]))

	fig_path = os.path.join(base_fig_path, 'all.png')
	#barplot(acc_mean['all'], miou_mean['all'], pck_mean['all'], methods, save_path=os.path.join(base_fig_path, 'all_mean.png'), measurement='Mean')
	#barplot(acc_std['all'], miou_std['all'], pck_std['all'], methods, save_path=os.path.join(base_fig_path, 'all_std.png'), measurement='Standard Deviation')
	print(methods)
	print('acc_mean\n', yaml.dump(acc_mean))
	print('acc_std\n', yaml.dump(acc_std))
	print('miou_mean\n', yaml.dump(miou_mean))
	print('miou_std\n', yaml.dump(miou_std))
	print('pck_mean\n', yaml.dump(pck_mean))
	print('pck_std\n', yaml.dump(pck_std))



	table = 'video & methods & accuracy & mIOU & PCK \\\\\n'
	# table = 'methods & acc mean & acc std & miou mean & miou std & pck mean & pck std \\\\\n'
	folders = ['MVI_3015.MP4', 'MVI_3018.MP4', 'MVI_4627.MP4', 'all']
	for folder_name in folders:
		print(folder_name)
		if folder_name == 'MVI_3015.MP4':
			vid_name = 'easy video'
		elif folder_name == 'MVI_3018.MP4':
			vid_name = 'semi-hard video'
		elif folder_name == 'MVI_4627.MP4':
			vid_name = 'hard video'
		elif folder_name == 'all':
			vid_name = 'average'

		best_acc,best_miou,best_pck = 0,0,0
		best_acc_index,best_miou_index,best_pck_index=-1,-1,-1

		for i in range(len(methods)):
			if acc_mean[folder_name][i] > best_acc:
				best_acc = acc_mean[folder_name][i]
				best_acc_index = i
			if miou_mean[folder_name][i] > best_miou:
				best_miou = miou_mean[folder_name][i]
				best_miou_index = i
			if pck_mean[folder_name][i] != '-' and pck_mean[folder_name][i] > best_pck:
				pck_acc = pck_mean[folder_name][i]
				best_pck_index = i

		for i in range(len(methods)):

			if methods[i] == 'DL_segmentation':
				method_name = 'DL segmentation'
			elif methods[i] == 'quadrilaterals':
				method_name = 'quadrilateral fitting'
			elif methods[i] == 'merged':
				method_name = 'combining quadrilaterals'
			elif methods[i] == 'cuboid_regular':
				method_name = 'optimizing cuboid without VP'
			elif methods[i] == 'cuboid_vp':
				method_name = 'optimizing cuboid with VP'
			elif methods[i] == 'cuboid_post':
				method_name = 'optimizing cuboid without and with VP'
			elif methods[i] == 'contour':
				method_name = 'contour-based'
			elif methods[i] == 'gradient':
				method_name = 'gradient-based'
			else:
				method_name = methods[i]

			#new_entry = '{} && {:.3f} && {:.3f} && {:.3f} && {:.3f} && {:.3f} && {:.3f} \\\\\n'.format(methods[i], acc_mean[folder_name][i], acc_std[folder_name][i], 
			#	miou_mean[folder_name][i], miou_std[folder_name][i], pck_mean[folder_name][i], pck_std[folder_name][i])
			acc_entry = '{:.3f}'.format(acc_mean[folder_name][i]) if best_acc_index != i else '\\textbf{{{:.3f}}}'.format(acc_mean[folder_name][i])
			miou_entry = '{:.3f}'.format(miou_mean[folder_name][i]) if best_miou_index != i else '\\textbf{{{:.3f}}}'.format(miou_mean[folder_name][i])
			if pck_mean[folder_name][i] == '-':
				pck_entry = '-'
			else:
				pck_entry = '{:.3f}'.format(pck_mean[folder_name][i]) if best_pck_index != i else '\\textbf{{{:.3f}}}'.format(pck_mean[folder_name][i])
			if best_pck_index == -1:
				new_entry = '{} & {} & {} & {} \\\\\n'.format(vid_name, method_name, acc_entry, miou_entry) 
			else:
				new_entry = '{} & {} & {} & {} & {} \\\\\n'.format(vid_name, method_name, acc_entry, miou_entry, pck_entry) 
			table += new_entry
		table += '\\hline\n'
	#for folder_name in folders:
		#table += folder_name + ' '

	with open(os.path.join(base_fig_path, 'table4'), 'w+') as f:
		f.write(table)

def read_data(folder_name, step):
	pck = []
	acc = []
	miou = []
	steps = []
	with open('data/results/metrics', 'r') as f:
		lines = f.readlines()
		lines = [l.strip() for l in lines]
	for l in lines[1:]:
		l = l.split('\t')
		if l[0] == folder_name:# and l[1] == step:
			acc.append(float(l[2]))
			miou.append(float(l[3]))
			if float(l[4]) == -1:
				pck.append(0)
			else:
				pck.append(float(l[4]))
			steps.append(l[1])
	return acc, miou, pck, steps

def barplot(acc, miou, pck, steps, save_path=None, measurement='Mean'):
	# set width of bar

	fontsize = 28 
	barWidth = 0.25 if pck is not None else 1/3
	fig = plt.subplots(figsize =(12, 8)) 
	   
	# Set position of bar on X axis 
	br1 = np.arange(len(acc)) 
	br2 = [x + barWidth for x in br1] 
	br3 = [x + barWidth for x in br2] 
	   
	# Make the plot 
	plt.bar(br1, acc, color ='r', width = barWidth, 
	        edgecolor ='grey', label ='accuracy') 
	plt.bar(br2, miou, color ='g', width = barWidth, 
	        edgecolor ='grey', label ='mIOU')
	if pck is not None:
		plt.bar(br3, pck, color ='b', width = barWidth, 
	        edgecolor ='grey', label ='PCK') 
	   
	# Adding Xticks  
	plt.xlabel('Method', fontweight ='bold', fontsize=fontsize) 
	plt.ylabel(measurement, fontweight ='bold', fontsize=fontsize)
	if pck is not None:
		plt.xticks([r + barWidth for r in range(len(acc))], steps, fontsize=fontsize)
	else:
		plt.xticks([r + barWidth/2 for r in range(len(acc))], steps, fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

	if measurement == 'Mean':
		plt.ylim(0,1)

	#plt.legend(loc="upper left", fontsize=fontsize)

	plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                borderaxespad=0, ncol=2, fontsize=fontsize)

	plt.tight_layout()

	if save_path is None:
		plt.show()
	else:
		#name = ".".join(save_path.split('/')[-1].split('.')[:-1])
		name = save_path.split('/')[-1].split('.')[-2].split('_')[-1]
		#plt.title(measurement, fontsize=14)
		plt.savefig(save_path)

def lineplot(metric, steps):
	fig = plt.figure()
	plt.plot(steps, metric)
	plt.show()

if __name__=="__main__":
	#acc, miou, pck, steps = read_data('MVI_4627.MP4', 'quadrilaterals')
	#barplot(acc, miou, pck, steps)
	#lineplot(miou, steps)

	for results_folder in ['results_init']:#['results_0', 'results_100', 'results_init']:

		methods = ['random', 'generic', 'contour', 'gradient', 'quadrilaterals'] if results_folder == 'results_init' else ['DL_segmentation', 'quadrilaterals', 'merged', 'cuboid_regular', 'cuboid_vp', 'cuboid_post']
		plot_inits(methods, results_folder=results_folder)