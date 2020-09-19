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


def write_pose_to_file(file, R, T):
    """Write [R | T] pose as a single line in a text file rounded to 6 decimals

    :param file: .txt file to write
    :param R: Rotation matrix (Shape : 3 x 3)
    :param T: Translation vector (Shape : 3 X 1)
    """
    T = np.round(T, 6)
    R = np.round(R, 6)
    file.write('%s %s %s ' % (str(R[0, 0]), str(R[0, 1]), str(R[0, 2])))
    file.write('%s ' % str(T[0, 0]))
    file.write('%s %s %s ' % (str(R[1, 0]), str(R[1, 1]), str(R[1, 2])))
    file.write('%s ' % str(T[1, 0]))
    file.write('%s %s %s ' % (str(R[2, 0]), str(R[2, 1]), str(R[2, 2])))
    file.write('%s' % str(T[2, 0]))
    file.write('\n')
