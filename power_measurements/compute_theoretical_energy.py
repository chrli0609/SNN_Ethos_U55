



def compute_num_synaptic_operations(layer_sizes_list, layers_out_activity_list):

	num_syn_ops = 0
	
	for i in range(1, len(layer_sizes_list)):
		# Skip first, since only interested in output
		if i == 0:
			continue

		fanout = layer_sizes_list[i]
		activity = layers_out_activity_list[i]

		
		num_syn_ops += fanout * activity

		
		

		
		
		
	return num_syn_ops




def main():



	# Note that our NPU implementation will have a little more than this since sizes must be divisible by 8
	layer_sizes_list = [
		[784, 72, 10],
		[784, 64, 64, 10],
		[784, 56, 56, 56, 10],
		[784, 48, 48, 48, 48, 10]
	]
	
	layers_out_activity_list = [
		[None, 1, 0.823111],
		[None, 1, 0.918222, 0.592],
		[None, 1, 0.780444, 0.716444, 0.548444],
		[None, 1, 0.882667, 0.715556, 0.660444, 0.536]
	]


	# Inference frequency
	inference_frequency = 1 / (10 * 10**-3)	# 10 ms period



	# energy per synaptic operation (J)
	truenorth_energy_per_synops = 26 * 10**-12 #pJ



	for i in range(len(layer_sizes_list)):
		
		print("For model", layer_sizes_list[i])
		tot_num_synops = compute_num_synaptic_operations(layer_sizes_list[i], layers_out_activity_list[i])


		tot_energy = tot_num_synops * truenorth_energy_per_synops

		#to watt
		tot_power_consumption = tot_energy * inference_frequency



		print("Total Number of Synaptic Operations in model:", tot_num_synops)
		print("Total Theoretical Energy consumption for each model inference (J):", tot_energy)
		print("Total Power Consumption (mW):", tot_power_consumption * 10**3)


	
	






if __name__ == '__main__':
	main()	

