import os
import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt


def fundamental_matrix(matches):
    """This function computes the fundamental matrix F given the data provided.
    It should return F as well as the residual res err. The residual is defined
    as the mean squared distance between the points in the two images and the
    corresponding epipolar lines.

    F : the 3x3 fundamental matrix
    res_err : mean squared distance between points in the two images and their
        their corresponding epipolar lines

    """
    # normalize points
    p1 = np.concatenate([matches[:, :2], np.ones((len(matches), 1))], axis=1)
    p2 = np.concatenate([matches[:, 2:], np.ones((len(matches), 1))], axis=1)
    mu1 = np.mean(p1, axis=0)
    mu2 = np.mean(p2, axis=0)
    std1 = np.std(p1, axis=0)
    std2 = np.std(p2, axis=0)
    T1 = np.array([
        [1 / std1[0], 0, -mu1[0] / std1[0]],
        [0, 1 / std1[1], -mu1[1] / std1[1]],
        [0, 0, 1]
    ])
    T2 = np.array([
        [1 / std2[0], 0, -mu2[0] / std2[0]],
        [0, 1 / std2[1], -mu2[1] / std2[1]],
        [0, 0, 1]
    ])
    x1 = np.dot(T1, p1.T).T
    x2 = np.dot(T2, p2.T).T

    # construct A matrix
    A = np.empty((len(matches), 9))
    for i in range(len(matches)):
        u, v, _ = x1[i]
        a, b, _ = x2[i]
        A[i] = np.array([a*u, a*v, a, b*u, b*v, b, u, v, 1])

    # SVD of A
    _, _, V = np.linalg.svd(A)
    # find minimum right eigenvector -- this is our f because when multiplied
    # with the other eigenvectors we get zero, which is what we want
    f = V[2]
    F = f.reshape((3, 3))
    # enforce that it is rank 2
    U, s, V = np.linalg.svd(F)
    S = np.diag(s)
    S[2, 2] = 0
    F = np.dot(U, np.dot(S, V))

    # de-normalize F
    F = np.dot(T2.T, np.dot(F, T1))

    # compute residual error
    d12 = np.empty(len(matches))
    d21 = np.empty(len(matches))
    for i in range(len(matches)):
        d12[i] = np.abs(np.dot(p1[i], np.dot(F, p2[i]))) / np.linalg.norm(np.dot(F, p2[i]))
        d21[i] = np.abs(np.dot(p2[i], np.dot(F, p1[i]))) / np.linalg.norm(np.dot(F, p1[i]))
    res_err = np.sum(d12 ** 2 + d21 ** 2) / (2 * len(matches))

    return F, res_err


def find_rotation_translation(E):
    """This function estimates the extrinsic parameters of the second camera.
    The function should return a cell array R of all the possible rotation
    matrices and a cell array t with all the possible translation vectors.

    R : 3D numpy array with the possible rotation matrices of second camera
    t : 2D numpy array of the possible translation vectors of second camera

    """
    R = np.array([np.eye(3)])
    t = np.array([np.zeros(3)])
    return R, t


def find_3d_points(E, P1, P2):
    """This function reconstructs the 3D point cloud. In particular, it returns
    a N × 3 array called points, where N is the number of the corresponding
    matches. It also returns the reconstruction error rec err, which is defined
    as the mean distance between the 2D points and the projected 3D points in
    the two images.

    """
    points = np.array([np.zeros(3)])
    err = 0
    return points, err


def plot_3d(points):
    """This function plots the 3D points in a 3D plot and displays the camera
    centers for both cameras.

    """
    pass


def reconstruct_3d(name, plot=True):
    """
    Homework 2: 3D reconstruction from two Views
    This function takes as input the name of the image pairs (i.e. 'house' or
    'library') and returns the 3D points as well as the camera matrices...but
    some functions are missing.

    NOTES
    (1) The code has been written so that it can be easily understood. It has
    not been written for efficiency.
    (2) Don't make changes to this main function since I will run my
    reconstruct_3d.py and not yours. I only want from you the missing
    functions and they should be able to run without crashing with my
    reconstruct_3d.py
    (3) Keep the names of the missing functions as they are defined here,
    otherwise things will crash

    """

    ## Load images, K matrices and matches
    data_dir = os.path.join('..', 'data', name)

    # images
    I1 = scipy.misc.imread(os.path.join(data_dir, "{}1.jpg".format(name)))
    I2 = scipy.misc.imread(os.path.join(data_dir, "{}2.jpg".format(name)))

    # K matrices
    K1 = np.array(scipy.io.loadmat(os.path.join(data_dir, "{}1_K.mat".format(name)))["K"], order='C')
    K2 = np.array(scipy.io.loadmat(os.path.join(data_dir, "{}2_K.mat".format(name)))["K"], order='C')

    # corresponding points
    # this is a N x 4 where:
    # matches[i,0:2] is a point in the first image
    # matches[i,2:4] is the corresponding point in the second image
    matches = np.loadtxt(os.path.join(data_dir, "{}_matches.txt".format(name)))

    # visualize matches (disable or enable this whenever you want)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(np.concatenate([I1, I2], axis=1))
        ax.plot(matches[:, 0], matches[:, 1], 'r+')
        ax.plot(matches[:, 2] + I1.shape[1], matches[:, 3], 'r+')
        ax.plot(np.array([matches[:, 0], matches[:, 2] + I1.shape[1]]), matches[:, [1, 3]].T, 'r')

    # compute the fundamental matrix
    (F, res_err) = fundamental_matrix(matches)
    print('Residual in F = {}'.format(res_err))

    # compute the essential matrix
    E = np.dot(np.dot(K2.T, F), K1)

    # compute the rotation and translation matrices
    (R, t) = find_rotation_translation(E)

    # Find R2 and t2 from R, t such that largest number of points lie in front
    # of the image planes of the two cameras
    P1 = np.dot(K1, np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1))

    # the number of points in front of the image planes for all combinations
    num_points = np.zeros((len(t), len(R)))

    # the reconstruction error for all combinations
    errs = np.empty((len(t), len(R)))

    for ti, t2 in enumerate(t):
        t2 = t[ti]
        for ri, R2 in enumerate(R):
            R2 = R[ri]
            P2 = np.dot(K2, np.concatenate([R2, t2[:, None]], axis=1))

            points_3d, errs[ti, ri] = find_3d_points()

            Z1 = points_3d[:, 2]
            Z2 = (np.dot(R2[2], points_3d.T) + t2[2]).T
            num_points[ti, ri] = np.sum((Z1 > 0) & (Z2 > 0))

    j = 0 # pick one out the best combinations
    (ti, ri) = np.nonzero(num_points == np.max(num_points))
    print('Reconstruction error = {}'.format(errs[ti[j], ri[j]]))

    t2 = t[ti[j]]
    R2 = R[ri[j]]
    P2 = np.dot(K2, np.concatenate([R2, t2[:, None]], axis=1))

    # compute the 3D points with the final P2
    points = find_3d_points()

    plot_3d()
