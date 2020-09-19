import numpy as np


def get_K_from_txt(path):
    """Get The K-Matrix

    covert a .txt containing Intrinsic Matrix (K)
    values to a 2D Float-Value Array

    :param path: .txt file path to K-Matrix values
    :return: K-matrix as a 2D np-array
    """
    K = []
    print("Loading K matrix from -> %s" % path)
    file = open(path, 'r')
    for line in file:
        line = line.strip(',').split(',')
        K = K + line
    return np.genfromtxt(K, dtype=np.float).reshape((3, 3))


def write_pose_to_file(file, R, t):
    pass

