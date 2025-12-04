import numpy as np

img_height = 32
img_width = 32
img_channels = 2


def circular_cross_correlation(sample1, sample2):
    max_corr = -np.inf
    max_shift = 0

    for shift in range(len(sample1)):
        shifted_sample1 = np.roll(sample1, shift)
        corr = np.correlate(shifted_sample1, sample2, 'valid')

        if corr > max_corr:
            max_corr = corr
            max_shift = shift

    return max_shift


def preprocess_data(data):
    data = data.astype('float32')
    return np.reshape(data, (len(data), img_channels, img_height, img_width))
