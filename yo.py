import numpy as np
import itertools
import time
import os


# temp = [[]]*3

temp = [[],[],[]]

for i in range(3):
	add = [i, i**2]
	for a in add:
		temp[i].append(add)
	#temp[i].extend([i,i**2])

print(temp)