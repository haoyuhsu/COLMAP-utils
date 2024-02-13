from pyntcloud import PyntCloud
import numpy as np
import vedo
import torch
import os
import argparse

from colmap_read_model import *


def main():

    # Argument of database
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", default='')
    args = parser.parse_args()

    if args.ply_path.endswith('.bin'):
        cloud_dict = read_points3d_binary(args.ply_path)
        pcds = []
        for k in cloud_dict:
            cloud = cloud_dict[k]
            pcd = cloud.xyz
            pcd = pcd.reshape(-1, 3)
            pcds.append(pcd)
        pcds = np.vstack(pcds)
    else:
        cloud = PyntCloud.from_file(args.ply_path)
        pcds = cloud.points.to_numpy()[:,(0,1,2)]

    pcds = torch.Tensor(pcds)
    print("Number of points: ", pcds.shape[0])

    points = vedo.Points(pcds, r=1, c=(0.3, 0.3, 0.3), alpha=0.5)
    vedo.show(points)


if __name__ == "__main__":
    main()