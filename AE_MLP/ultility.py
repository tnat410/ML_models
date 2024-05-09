import numpy as np
np.set_printoptions(precision=15,suppress=True)

def file_to_array(file):
	fo = open(file, "r+")

	fo.readline()
	fo.readline()
	fo.readline()
	fo.readline()

	# Read number of samples
	line = fo.readline()
	num = int(line.split()[0])
	j = []

	for i in range(num):
		line = fo.readline()
		k = line.split()

		for i in k[2:-1]:
			j.append(float(i))

	t = np.reshape(j, (num, int(len(j)/num)))

	return t,num
