import yaml
import os

def pretty_print(dict):
	return yaml.dump(dict, default_flow_style=False)

def save_string(string, save_dir):
	with open(os.path.join(save_dir,'log.txt'), 'a+') as outfile:
		outfile.write('\n')
		outfile.write(string)

def average(output):
	out = torch.mean(torch.stack(my_list),0)
	return out