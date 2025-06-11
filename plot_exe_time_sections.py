import matplotlib.pyplot as plt

# Your data
sections = [
"ait_reset_model_for_new_sample",
"ait_set_test_pattern_pointer_to_model",
"ait_get_time_since_last_update",
"ait_ethosu_reserve_driver",
"ait_check_npu_nn_op_validity",
"ait_process_cms_preamble",
"ait_verify_base_addr",
"ait_ethosu_flush_dcache",
"ait_ethosu_request_power",
"ait_ethosu_dev_run_command_stream",
"ait_wait_npu_task_complete_irq",
"ait_ethosu_release_power",
"ait_ethosu_invalidate_dcache",
"ait_ethosu_release_driver",
"ait_arg_max"
]

times = [
8.044444,
0.044444,
48.933333,
3.066667,
24.111111,
46.977778,
0.422222,
1546.488889,
136.066667,
101.088889,
2321.4,
0.066667,
1580.2,
6.466667,
2.044444
]

# Sort by execution time
sections_sorted, times_sorted = zip(*sorted(zip(sections, times), key=lambda x: x[1], reverse=True))

# Plot
plt.figure(figsize=(10, 8))
bars = plt.barh(sections_sorted, times_sorted, color='skyblue')
plt.xlabel("Execution Time (μs)")
plt.title("Execution Time per Code Section (MNIST 784x32x10 Rate Encoding)")

# Add labels
for bar, time in zip(bars, times_sorted):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f"{time:.1f} μs", va='center')

plt.gca().invert_yaxis()  # Highest first
plt.tight_layout()
plt.show()

