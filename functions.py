from collections import Counter

import cv2
import numpy as np

# Re-implement
def bgr_to_ycrcb(image):
    """
    Convert a BGR image to YCrCb color space.

    Parameters:
    image (numpy.ndarray): BGR image

    Returns:
    numpy.ndarray: YCrCb image
    """
    # Define transformation matrices
    bgr_to_y = np.array([0.299, 0.587, 0.114])
    bgr_to_cb = np.array([-0.169, -0.331, 0.500])
    bgr_to_cr = np.array([0.500, -0.419, -0.081])
    
    # Initialize the output image
    ycrcb_image = np.zeros_like(image, dtype=np.float32)

    # Apply transformation
    ycrcb_image[:, :, 0] = np.dot(image, bgr_to_y.T) + 0
    ycrcb_image[:, :, 1] = np.dot(image, bgr_to_cb.T) + 128
    ycrcb_image[:, :, 2] = np.dot(image, bgr_to_cr.T) + 128
    
    return ycrcb_image.astype(np.uint8)

# box_filter
def box_filter(image, ksize=(2, 2)):
  height, width = image.shape
  filtered_image = np.zeros_like(image)
  for i in range(height):
    for j in range(width):
      # Define the kernel boundaries
      start_i = max(i - ksize[0] // 2, 0)
      end_i = min(i + ksize[0] // 2 + 1, height)
      start_j = max(j - ksize[1] // 2, 0)
      end_j = min(j + ksize[1] // 2 + 1, width)

      # Calculate the sum of pixel values within the kernel
      kernel_sum = np.sum(image[start_i:end_i, start_j:end_j])

      # Apply the mean filter
      filtered_image[i, j] = kernel_sum / ((end_i - start_i) * (end_j - start_j))
  return filtered_image

# def box_filter(image, ksize=(2, 2)):
#     height, width = image.shape
#     filtered_image = np.zeros_like(image)
#     for i in range(height):
#         for j in range(width):
#             # Define the kernel boundaries
#             start_i = max(i - ksize[0] // 2, 0)
#             end_i = min(i + ksize[0] // 2 + 1, height)
#             start_j = max(j - ksize[1] // 2, 0)
#             end_j = min(j + ksize[1] // 2 + 1, width)
            
#             # Apply the mean filter within the kernel
#             filtered_image[i, j] = np.mean(image[start_i:end_i, start_j:end_j])
#     return filtered_image

# Subsampling function
def subsample_chrominance(cr, cb, SSH=2, SSV=2):
    # Apply box filter first
    cr_filtered = box_filter(cr, ksize=(2, 2))
    cb_filtered = box_filter(cb, ksize=(2, 2))
    
    # Now subsample
    cr_subsampled = cr_filtered[::SSV, ::SSH]
    cb_subsampled = cb_filtered[::SSV, ::SSH]

    return round(cr_subsampled, 2), round(cb_subsampled, 2)

# def subsample_chrominance(cr, cb, SSH=2, SSV=2):
#     height, width = cr.shape

#     # Calculate dimensions after subsampling
#     subsampled_height = height // SSV + 1
#     subsampled_width = width // SSH + 1

#     # Initialize subsampled arrays
#     cr_subsampled = np.zeros((subsampled_height, subsampled_width), dtype=np.float32)
#     cb_subsampled = np.zeros((subsampled_height, subsampled_width), dtype=np.float32)

#     # Perform subsampling
#     for i in range(subsampled_height):
#         for j in range(subsampled_width):
#             # Calculate the start and end indices for averaging
#             start_i = i * SSV
#             end_i = start_i + SSV
#             start_j = j * SSH
#             end_j = start_j + SSH

#             # Average over the block defined by start/end indices
#             cr_subsampled[i, j] = np.mean(cr[start_i:end_i, start_j:end_j])
#             cb_subsampled[i, j] = np.mean(cb[start_i:end_i, start_j:end_j])

#     return cr_subsampled, cb_subsampled

def zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    computes the zigzag of a quantized block
    :param numpy.ndarray matrix: quantized matrix
    :returns: zigzag vectors in an array
    """
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    i = 0
    output = np.zeros((v_max * h_max))

    while (v < v_max) and (h < h_max):
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                output[i] = matrix[v, h]  # first line
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                output[i] = matrix[v, h]
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                output[i] = matrix[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                output[i] = matrix[v, h]
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                output[i] = matrix[v, h]
                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                output[i] = matrix[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            output[i] = matrix[v, h]
            break
    return output


def trim(array: np.ndarray) -> np.ndarray:
    """
    in case the trim_zeros function returns an empty array, add a zero to the array to use as the DC component
    :param numpy.ndarray array: array to be trimmed
    :return numpy.ndarray:
    """
    trimmed = np.trim_zeros(array, 'b')
    if len(trimmed) == 0:
        trimmed = np.zeros(1)
    return trimmed


def run_length_encoding(array: np.ndarray) -> list:
    """
    finds the intermediary stream representing the zigzags
    format for DC components is <size><amplitude>
    format for AC components is <run_length, size> <Amplitude of non-zero>
    :param numpy.ndarray array: zigzag vectors in array
    :returns: run length encoded values as an array of tuples
    """
    encoded = list()
    run_length = 0
    eob = ("EOB",)

    for i in range(len(array)):
        for j in range(len(array[i])):
            trimmed = trim(array[i])
            if j == len(trimmed):
                encoded.append(eob)  # EOB
                break
            if i == 0 and j == 0:  # for the first DC component
                encoded.append((int(trimmed[j]).bit_length(), trimmed[j]))
            elif j == 0:  # to compute the difference between DC components
                diff = int(array[i][j] - array[i - 1][j])
                if diff != 0:
                    encoded.append((diff.bit_length(), diff))
                else:
                    encoded.append((1, diff))
                run_length = 0
            elif trimmed[j] == 0:  # increment run_length by one in case of a zero
                run_length += 1
            else:  # intermediary steam representation of the AC components
                encoded.append((run_length, int(trimmed[j]).bit_length(), trimmed[j]))
                run_length = 0
            # send EOB
        if not (encoded[len(encoded) - 1] == eob):
            encoded.append(eob)
    return encoded


def get_freq_dict(array: list) -> dict:
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)
    result = {k: d / len(array) for k, d in data.items()}
    return result


def find_huffman(p: dict) -> dict:
    """
    returns a Huffman code for an ensemble with distribution p
    :param dict p: frequency table
    :returns: huffman code for each symbol
    """
    # Base case of only two symbols, assign 0 or 1 arbitrarily; frequency does not matter
    if len(p) == 2:
        return dict(zip(p.keys(), ['0', '1']))

    # Create a new distribution by merging lowest probable pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = find_huffman(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + '0', ca1a2 + '1'

    return c


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]
