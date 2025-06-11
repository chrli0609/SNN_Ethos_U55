



def compute_num_synaptic_operations(layer_sizes_list, layers_out_activity_list):

	num_syn_ops = 0
	
	for i in range(len(layer_sizes_list)):
		# Skip first, since only interested in output
		if i == 0:
			continue

		fanout = layer_sizes_list[i]
		activity = layers_out_activity_list[i]

		
		num_syn_ops += fanout * activity

		
		

		
		
		
	return num_syn_ops




def main():



	# Note that our NPU implementation will have a little more than this since sizes must be divisible by 8
	layer_sizes_list = [784, 32, 10]


	layers_out_activity_list = [1, 0.5, 0.2]


	# energy per synaptic operation
	truenorth_energy_per_synops = 26 #pJ



	tot_num_synops = compute_num_synaptic_operations(layer_sizes_list, layers_out_activity_list)


	tot_energy = tot_num_synops * truenorth_energy_per_synops




	print("Total Number of Synaptic Operations in model:", tot_num_synops)
	print("Total Theoretical Energy consumption for each model inference", tot_energy)


	
	






if __name__ == '__main__':
	main()	

