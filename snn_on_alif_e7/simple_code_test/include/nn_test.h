

void create_tensors(uint8_t*** base_addrs, size_t** base_addrs_size, int* num_tensors);

void free_tensors(uint8_t** base_addrs, size_t* base_addrs_size, int num_tensors);

int create_n_run_cmd_stream();