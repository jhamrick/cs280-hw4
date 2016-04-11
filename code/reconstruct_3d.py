# coding=utf-8

import os
import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


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
    F = V[-1].reshape((3, 3))
    # enforce that it is rank 2
    U, s, V = np.linalg.svd(F)
    S = np.diag(s)
    S[-1, -1] = 0
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
    The function should return a numpy array R of all the possible rotation
    matrices and a numpy array t with all the possible translation vectors.

    R : 3D numpy array with the possible rotation matrices of second camera
    t : 2D numpy array of the possible translation vectors of second camera

    """
    # t is the third left singular vector of E
    U, _, V = np.linalg.svd(E)
    t = np.array([U[:, 2], -U[:, 2]])

    # R = U * R90.T * V
    R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    all_R = np.array([
        np.dot(U, np.dot(R90.T, V)),
        -np.dot(U, np.dot(R90.T, V)),
        np.dot(U, np.dot(R90, V)),
        -np.dot(U, np.dot(R90, V))
    ])

    R = []
    for i in range(4):
        if np.allclose(np.linalg.det(all_R[i]), 1):
            R.append(all_R[i])
    R = np.array(R)

    return R, t


def find_3d_points(matches, P1, P2, R, t):
    """This function reconstructs the 3D point cloud. In particular, it returns
    a N × 3 array called points, where N is the number of the corresponding
    matches. It also returns the reconstruction error rec err, which is defined
    as the mean distance between the 2D points and the projected 3D points in
    the two images.

    """
    points = np.empty((len(matches), 3))
    reconstructed = np.empty(matches.shape)
    for i in range(len(matches)):
        x1 = matches[i, :2]
        x2 = matches[i, 2:]

        # setup A
        A = np.array([
            P1[0] - P1[2] * x1[0],
            P1[1] - P1[2] * x1[1],
            P2[0] - P2[2] * x2[0],
            P2[1] - P2[2] * x2[1]
        ])

        # solve for x
        _, _, V = np.linalg.svd(A)
        points[i] = V[-1, :-1] / V[-1, -1]

        # reproject points and compute error
        r1 = np.dot(P1, np.append(points[i], [1]))
        r2 = np.dot(P2, np.append(points[i], [1]))
        reconstructed[i, :2] = r1[:2] / r1[2]
        reconstructed[i, 2:] = r2[:2] / r2[2]

    # compute reconstruction error only for points that actually lie in front
    # of the cameras
    err1 = np.sqrt(np.sum((reconstructed[:, :2] - matches[:, :2]) ** 2, axis=1))
    err2 = np.sqrt(np.sum((reconstructed[:, 2:] - matches[:, 2:]) ** 2, axis=1))
    err = np.mean((err1 + err2) / 2)

    return points, err


def plot_3d(points, R, t, colors=None):
    """This function plots the 3D points in a 3D plot and displays the camera
    centers for both cameras.

    """
    # find points that are in front of the cameras
    Z1 = points[:, 2]
    Z2 = (np.dot(R[2], points.T) + t[2]).T
    ok = ((Z1 > 0) & (Z2 > 0))

    if colors is not None:
        colors = colors[ok]
    else:
        colors = cm.jet(np.linspace(0, 1, len(points)))[ok]

    # x, y, and z of the points
    X = points[ok, 0]
    Y = points[ok, 1]
    Z = points[ok, 2]

    # camera centers
    o1 = np.array([0, 0, 0])
    o2 = np.dot(R, o1) + t

    # compute axis limits so that the scale is the same on all axes, but the
    # axis is centered on the data
    xmin = min(0, o1[0], o2[0], X.min())
    xmax = max(0, o1[0], o2[0], X.max())
    xmid = (xmin + xmax) / 2
    ymin = min(0, o1[1], o2[1], Y.min())
    ymax = max(0, o1[1], o2[1], Y.max())
    ymid = (ymin + ymax) / 2
    zmin = min(0, o1[2], o2[2], Z.min())
    zmax = max(0, o1[2], o2[2], Z.max())
    zmid = (zmin + zmax) / 2
    vmin = min(xmin, ymin, zmin)
    vmax = max(xmax, ymax, zmax)
    vscale = vmax - vmin
    xmin = xmid - vscale / 2
    xmax = xmid + vscale / 2
    ymin = ymid - vscale / 2
    ymax = ymid + vscale / 2
    zmin = zmid - vscale / 2
    zmax = zmid + vscale / 2

    # vectors representing the camera coordinate systems
    v1x = np.array([(vmax - vmin) / 10, 0, 0])
    v1y = np.array([0, (vmax - vmin) / 10, 0])
    v1z = np.array([0, 0, (vmax - vmin) / 10])
    v2x = np.dot(R, v1x) + t
    v2y = np.dot(R, v1y) + t
    v2z = np.dot(R, v1z) + t

    # create the figure
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # 3D plot
    ax1.scatter(X, Y, Z, c=colors, marker='+', zdir='y', alpha=0.3)
    ax1.plot(o1[[0]], o1[[2]], o1[[2]], 'ro', zdir='y')
    ax1.plot(o2[[0]], o2[[2]], o2[[2]], 'bo', zdir='y')
    ax1.plot([o1[0], v1x[0]], [o1[1], v1x[1]], [o1[2], v1x[2]], 'r-', zdir='y')
    ax1.plot([o2[0], v2x[0]], [o2[1], v2x[1]], [o2[2], v2x[2]], 'r-', zdir='y')
    ax1.plot([o1[0], v1y[0]], [o1[1], v1y[1]], [o1[2], v1y[2]], 'b-', zdir='y')
    ax1.plot([o2[0], v2y[0]], [o2[1], v2y[1]], [o2[2], v2y[2]], 'b-', zdir='y')
    ax1.plot([o1[0], v1z[0]], [o1[1], v1z[1]], [o1[2], v1z[2]], 'g-', zdir='y')
    ax1.plot([o2[0], v2z[0]], [o2[1], v2z[1]], [o2[2], v2z[2]], 'g-', zdir='y')
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([zmin, zmax])
    ax1.set_zlim([ymin, ymax])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y")
    ax1.set_aspect("equal")

    # XY plane
    ax2.scatter(X, Y, c=colors, marker='+', alpha=0.3)
    ax2.plot(o1[0], o1[1], 'ro')
    ax2.plot(o2[0], o2[1], 'bo')
    ax2.plot([o1[0], v1x[0]], [o1[1], v1x[1]], 'r-')
    ax2.plot([o2[0], v2x[0]], [o2[1], v2x[1]], 'r-')
    ax2.plot([o1[0], v1y[0]], [o1[1], v1y[1]], 'b-')
    ax2.plot([o2[0], v2y[0]], [o2[1], v2y[1]], 'b-')
    ax2.plot([o1[0], v1z[0]], [o1[1], v1z[1]], 'g-')
    ax2.plot([o2[0], v2z[0]], [o2[1], v2z[1]], 'g-')
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")

    # ZY plane
    ax3.scatter(Z, Y, c=colors, marker='+', alpha=0.3)
    ax3.plot(o1[2], o1[1], 'ro')
    ax3.plot(o2[2], o2[1], 'bo')
    ax3.plot([o1[2], v1x[2]], [o1[1], v1x[1]], 'r-')
    ax3.plot([o2[2], v2x[2]], [o2[1], v2x[1]], 'r-')
    ax3.plot([o1[2], v1y[2]], [o1[1], v1y[1]], 'b-')
    ax3.plot([o2[2], v2y[2]], [o2[1], v2y[1]], 'b-')
    ax3.plot([o1[2], v1z[2]], [o1[1], v1z[1]], 'g-')
    ax3.plot([o2[2], v2z[2]], [o2[1], v2z[1]], 'g-')
    ax3.set_xlim([zmin, zmax])
    ax3.set_ylim([ymin, ymax])
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Y")
    ax3.set_aspect("equal")

    # XZ plane
    ax4.scatter(X, Z, c=colors, marker='+', alpha=0.3)
    ax4.plot(o1[0], o1[2], 'ro')
    ax4.plot(o2[0], o2[2], 'bo')
    ax4.plot([o1[0], v1x[0]], [o1[2], v1x[2]], 'r-')
    ax4.plot([o2[0], v2x[0]], [o2[2], v2x[2]], 'r-')
    ax4.plot([o1[0], v1y[0]], [o1[2], v1y[2]], 'b-')
    ax4.plot([o2[0], v2y[0]], [o2[2], v2y[2]], 'b-')
    ax4.plot([o1[0], v1z[0]], [o1[2], v1z[2]], 'g-')
    ax4.plot([o2[0], v2z[0]], [o2[2], v2z[2]], 'g-')
    ax4.set_xlim([xmin, xmax])
    ax4.set_ylim([zmin, zmax])
    ax4.set_xlabel("X")
    ax4.set_ylabel("Z")
    ax4.set_aspect("equal")

    fig.set_size_inches(12, 12, forward=True)
    plt.tight_layout()


def plot_2d(I1, I2, matches, points_3d, P1, P2):
    points_2d = np.empty(matches.shape)
    for i in range(len(points_3d)):
        r1 = np.dot(P1, np.append(points_3d[i], [1]))
        r2 = np.dot(P2, np.append(points_3d[i], [1]))
        points_2d[i, :2] = r1[:2] / r1[2]
        points_2d[i, 2:] = r2[:2] / r2[2]

    fig, ax = plt.subplots()
    ax.imshow(np.concatenate([I1, I2], axis=1))

    ax.plot(matches[:, 0], matches[:, 1], 'r+', alpha=0.5)
    ax.plot(points_2d[:, 0], points_2d[:, 1], 'b+', alpha=0.5)
    ax.plot(np.array([matches[:, 0], points_2d[:, 0]]), np.array([matches[:, 1], points_2d[:, 1]]), 'k')

    ax.plot(matches[:, 2] + I1.shape[1], matches[:, 3], 'r+', alpha=0.5)
    ax.plot(points_2d[:, 2] + I1.shape[1], points_2d[:, 3], 'b+', alpha=0.5)
    ax.plot(np.array([matches[:, 2] + I1.shape[1], points_2d[:, 2] + I1.shape[1]]), np.array([matches[:, 3], points_2d[:, 3]]), 'k')

    ax.set_xlim([0, I1.shape[1] * 2])
    ax.set_ylim([I1.shape[0], 0])


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
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    ## Load images, K matrices and matches
    data_dir = os.path.join('..', 'data', name)

    # images
    I1 = scipy.misc.imread(os.path.join(data_dir, "{}1.jpg".format(name)))
    I2 = scipy.misc.imread(os.path.join(data_dir, "{}2.jpg".format(name)))

    # K matrices
    K1 = np.array(scipy.io.loadmat(os.path.join(data_dir, "{}1_K.mat".format(name)))["K"], order='C')
    K2 = np.array(scipy.io.loadmat(os.path.join(data_dir, "{}2_K.mat".format(name)))["K"], order='C')
    print("K1 = \n{}".format(K1))
    print("K2 = \n{}".format(K2))

    # corresponding points
    # this is a N x 4 where:
    # matches[i,0:2] is a point in the first image
    # matches[i,2:4] is the corresponding point in the second image
    matches = np.loadtxt(os.path.join(data_dir, "{}_matches.txt".format(name)))

    # visualize matches (disable or enable this whenever you want)
    fig, ax = plt.subplots()
    ax.imshow(np.concatenate([I1, I2], axis=1))
    colors = np.array([
        (matches[:, 0] - matches[:, 0].min()) / (matches[:, 0].max() - matches[:, 0].min()),
        np.zeros(len(matches)),
        (matches[:, 1] - matches[:, 1].min()) / (matches[:, 1].max() - matches[:, 1].min())
    ]).T
    ax.scatter(matches[:, 0], matches[:, 1], c=colors, marker='+')
    ax.scatter(matches[:, 2] + I1.shape[1], matches[:, 3], c=colors, marker='+')
    #ax.plot(np.array([matches[:, 0], matches[:, 2] + I1.shape[1]]), matches[:, [1, 3]].T, 'r', alpha=0.1)

    # compute the fundamental matrix
    (F, res_err) = fundamental_matrix(matches)
    print('F = \n{}'.format(F))
    print('Residual in F = {:.4f}'.format(res_err))

    # compute the essential matrix
    E = np.dot(np.dot(K2.T, F), K1)

    # compute the rotation and translation matrices
    (R, t) = find_rotation_translation(E)
    for R2 in R:
        print("R =\n{}".format(R2))
    for t2 in t:
        print("t = {}".format(t2))

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

            points_3d, errs[ti, ri] = find_3d_points(matches, P1, P2, R2, t2)

            Z1 = points_3d[:, 2]
            Z2 = (np.dot(R2[2], points_3d.T) + t2[2]).T
            num_points[ti, ri] = np.sum((Z1 > 0) & (Z2 > 0))

    j = 0 # pick one out the best combinations
    (ti, ri) = np.nonzero(num_points == np.max(num_points))
    print('Reconstruction error = {:.4f}'.format(errs[ti[j], ri[j]]))

    t2 = t[ti[j]]
    R2 = R[ri[j]]
    P2 = np.dot(K2, np.concatenate([R2, t2[:, None]], axis=1))

    # compute the 3D points with the final P2
    points, err = find_3d_points(matches, P1, P2, R2, t2)

    plot_3d(points, R2, t2, colors)
    plot_2d(I1, I2, matches, points, P1, P2)
