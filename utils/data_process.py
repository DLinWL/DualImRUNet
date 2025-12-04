import numpy as np
from utils.SparAlign import circular_cross_correlation

def process_data(trainval_filename, test_filename, spalign_flag):
    trainval_data = np.load(trainval_filename, allow_pickle=True).item()['feature_vector_data']
    test_data = np.load(test_filename, allow_pickle=True).item()['feature_vector_data']
    
    index_combinations = [[0, 4], [1, 5], [2, 6], [3, 7]]
    trainval_data = trainval_data[:, index_combinations[0], :, :]
    test_data = test_data[:, index_combinations[0], :, :]
    
    baseline_sample = trainval_data[5, :, :, :]
    
    if len(trainval_data) < 140000:
        repeat_times = 140000 // len(trainval_data) + (1 if 140000 % len(trainval_data) != 0 else 0)
        trainval_data = np.repeat(trainval_data, repeat_times, axis=0)
    
    local_rng = np.random.default_rng(seed=0)
    trainval_data = local_rng.choice(trainval_data, 140000, replace=False)
    
    data = np.concatenate((trainval_data, test_data), axis=0)
    baseline_sample = baseline_sample / np.max(np.abs(data))
    data = data / np.max(np.abs(data))
    data = np.roll(data, 12, axis=2)
    data = np.roll(data, 2, axis=3)
    baseline_sample = np.roll(baseline_sample, 12, axis=1)
    baseline_sample = np.roll(baseline_sample, 2, axis=2)
    
    if spalign_flag == 1:
        x_data_mag = abs(data[:, 0, :, :] + 1j * (data[:, 1, :, :]))
        baseline_sample_mag = abs(baseline_sample[0, :, :] + 1j * (baseline_sample[1, :, :]))
        
        compressed_sample0 = np.sum(baseline_sample_mag, axis=0)
        shifts = np.zeros(x_data_mag.shape[0])
        for i in range(x_data_mag.shape[0]):
            compressed_sample = np.sum(x_data_mag[i], axis=0)
            shifts[i] = circular_cross_correlation(compressed_sample, compressed_sample0)
            data[i] = np.roll(data[i], int(shifts[i]), axis=2)

        compressed_sample0 = np.sum(baseline_sample_mag, axis=1)
        for i in range(x_data_mag.shape[0]):
            compressed_sample = np.sum(x_data_mag[i], axis=1)
            shifts[i] = circular_cross_correlation(compressed_sample, compressed_sample0)
            data[i] = np.roll(data[i], int(shifts[i]), axis=1)
    
    total_size = len(data)
    test_size = total_size // 10 * 3
    train_val_size = total_size - test_size
    
    train_val_data = data[:train_val_size]
    test_data = data[train_val_size:]
    
    sample_interval = 5
    val_indices = np.arange(0, len(train_val_data), sample_interval)
    val_data = train_val_data[val_indices]
    train_indices = np.setdiff1d(np.arange(len(train_val_data)), val_indices)
    train_data = train_val_data[train_indices]
    
    return train_data, val_data, test_data
