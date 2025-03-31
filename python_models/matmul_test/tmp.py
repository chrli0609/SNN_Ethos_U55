

import numpy as np



height = 4
width = 4
channel = 16

stride_x = 64
stride_y = 16
element_size = 1

arr = np.arange(height*width*channel)


for x in range(height):
	for y in range(width):
		for c in range(channel):
			print("arr["+str(x)+"]["+str(y)+"]["+str(c)+"] =", arr[y*stride_y + x * stride_x + c * element_size])
