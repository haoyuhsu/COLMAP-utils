import pickle
import numpy as np
import os
import sys
import math
import shutil
import json
import time
import argparse
import glob
import vedo
from colmap_read_model import read_images_binary, read_cameras_binary, read_points3D_binary


ROOT_DIR = "/home/max/Documents/maxhsu/"


scene_up_vector_dict = {
    'tnt': {
        'Playground': [-0.00720354, -0.9963133, -0.08548705],
    },
    '360': {
        'bonsai': [ 0.02405242, -0.77633506, -0.6298614 ],
        'counter': [ 0.07449666, -0.80750495, -0.5851376 ],
        'garden': [-0.03292375, -0.8741887, -0.48446894],
    },
    'lerf': {
        'donuts': [0.0, 0.0, 1.0],
        'dozer_nerfgun_waldo': [-0.76060444, 0.00627117, 0.6491853 ],
        'espresso': [0.0, 0.0, 1.0],
        'figurines': [0.0, 0.0, 1.0],
        'ramen': [0.0, 0.0, 1.0],
        'shoe_rack': [0.0, 0.0, 1.0],
        'teatime': [0.0, 0.0, 1.0],
        'waldo_kitchen': [0.0, 0.0, 1.0],
    },
    'colmap_lerf': {
        'donuts': [ 0.07987297, -0.8506788, -0.5195825 ],  
        'dozer_nerfgun_waldo': [ 0.1031235, -0.83134925, -0.5460989 ],
        'espresso': [ 0.0531004, -0.8072565, -0.58780724],
        'figurines': [ 0.16696297, -0.9803059, -0.10546955],
        'ramen': [ 0.02134954, -0.74014527, -0.6721081 ],
        'shoe_rack': [ 0.00508022, -0.8688783, -0.4949998 ],
        'teatime': [ 0.0540938, -0.8366087, -0.54512364],
        'waldo_kitchen': [-0.01319592, -0.9988512, -0.04606834],
    }
}


scene_sampled_camera_dict = {
    'tnt': {
        'Playground': {
            'center_pos': [0.0, 0.0, -0.1],
            'radius': 0.5,
            'theta': 10,
            'num_views': 250,
            'hw': (548, 1008),
        }
    },
    '360': {
        'counter': {
            'center_pos': [0.1, 0.2, -0.5],
            'radius': 0.7,
            'theta': 30,
            'num_views': 150,
            'hw': (738, 994),
        },
        'garden': {
            'center_pos': [0.1, 0.0, -0.5],
            'radius': 1.0,
            'theta': 15,
            'num_views': 150,
            'hw': (738, 994),
        },
    },
    'lerf': {
        'teatime': {
            'center_pos': [-0.5, 0.1, -0.16],
            'radius': 0.5,
            'theta': 45,
            'num_views': 150,
            'hw': (738, 994),
        }
    }
}


def getNerfppNorm(c2w_list):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in c2w_list:
        cam_centers.append(cam[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


# get rotation matrix from aligning one vector to another vector
def get_rotation_matrix_from_vectors(v1, v2):
    # if two numpy array are the same, return identity matrix
    if np.allclose(v1, v2):
        return np.eye(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)
    # print('R:', R)
    return R


def sort_key(x):
    if len(x) > 2 and x[-10] == "_":
        return x[-9:]
    return x


def visualize_camera(poses_array, arrow_len=1, s=1):
    """
    params: poses_array: (N, 4, 4) or (N, 3, 4)
    """
    plt = vedo.Plotter()
    pos = poses_array[:, 0:3, 3]
    x_end = pos + arrow_len * poses_array[:, 0:3, 0]
    y_end = pos + arrow_len * poses_array[:, 0:3, 1]
    z_end = pos + arrow_len * poses_array[:, 0:3, 2]
    x = vedo.Arrows(pos, x_end, c="r", s=s)
    y = vedo.Arrows(pos, y_end, c="g", s=s)
    z = vedo.Arrows(pos, z_end, c="b", s=s)
    plt.show(x, y, z, axes=1, viewup="z")
    return x, y, z


def visualize_camera_and_points3D(poses_array, points3D, point_pos=None, arrow_len=1, s=1):
    """
    params: poses_array: (N, 4, 4) or (N, 3, 4)
    params: points3D: (N_pts, 3)
    """
    plt = vedo.Plotter()
    pos = poses_array[:, 0:3, 3]
    x_end = pos + arrow_len * poses_array[:, 0:3, 0]
    y_end = pos + arrow_len * poses_array[:, 0:3, 1]
    z_end = pos + arrow_len * poses_array[:, 0:3, 2]
    x = vedo.Arrows(pos, x_end, c="r", s=s)
    y = vedo.Arrows(pos, y_end, c="g", s=s)
    z = vedo.Arrows(pos, z_end, c="b", s=s)

    points3D = vedo.Points(points3D, r=1, c=(0.3, 0.3, 0.3), alpha=0.5)

    if point_pos is not None:
        # create a sphere as a reference point
        points = vedo.Sphere(np.array(point_pos), r=0.05, c="r")
        plt.show(x, y, z, points3D, points, axes=1, viewup="z")
    else:
        plt.show(x, y, z, points3D, axes=1, viewup="z")

    return x, y, z


def get_tnt_poses(scene_name='Playground'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/tnt/{}'.format(scene_name))
    pose_path = sorted([path for path in glob.glob(os.path.join(root_dir, 'pose', '*.txt'))], key=sort_key)
    poses = np.stack([np.loadtxt(f_pose).reshape(-1, 4) for f_pose in pose_path], axis=0)
    poses = poses[:, 0:3, :]
    return poses


def get_tnt_intrinsics(scene_name='Playground'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/tnt/{}'.format(scene_name))
    K = np.loadtxt(os.path.join(root_dir, 'intrinsics.txt'), dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy


def get_lerf_poses(scene_name='donuts'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/lerf/{}'.format(scene_name))
    # read poses and intrinsics of each frame from json file
    with open(os.path.join(root_dir, 'transforms.json'), 'r') as f:
        meta = json.load(f)
    # Sort the 'frames' list by 'file_path' to make sure that the order of images is correct
    # https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
    all_file_paths = [frame_info['file_path'] for frame_info in meta['frames']]
    sort_indices = [i[0] for i in sorted(enumerate(all_file_paths), key=lambda x:x[1])]
    meta['frames'] = [meta['frames'][i] for i in sort_indices]
    all_c2w = []
    for frame_info in meta['frames']:
        cam_mtx = np.array(frame_info['transform_matrix'])
        cam_mtx = cam_mtx @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
        all_c2w.append(cam_mtx)  # C2W (4, 4) OpenCV
    c2w_f64 = np.stack(all_c2w)
    c2w_f64 = c2w_f64[:, 0:3, :]
    return c2w_f64


def get_lerf_intrinsics(scene_name='donuts'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/lerf/{}'.format(scene_name))
    with open(os.path.join(root_dir, 'transforms.json'), 'r') as f:
        meta = json.load(f)
    # select intrinsics of the first frame
    for frame_info in meta['frames']:
        if 'fl_x' in meta:
            fx, fy, cx, cy = meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']
        else:
            fx, fy, cx, cy = frame_info['fl_x'], frame_info['fl_y'], frame_info['cx'], frame_info['cy']
        break
    return fx, fy, cx, cy


def get_360_poses(scene_name='bonsai'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/360/{}'.format(scene_name))
    imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))
    img_names = [imdata[k].name for k in imdata]
    perm = np.argsort(img_names)
    w2c_mats = []
    bottom = np.array([[0, 0, 0, 1.]])
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)
    poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices
    return poses


def get_360_intrinsics(scene_name='bonsai'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/360/{}'.format(scene_name))
    camdata = read_cameras_binary(os.path.join(root_dir, 'sparse/0/cameras.bin'))
    fx, fy, cx, cy = camdata[1].params
    return fx, fy, cx, cy


def get_colmap_poses(scene_name='donuts'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/colmap/{}'.format(scene_name))
    imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))
    img_names = [imdata[k].name for k in imdata]
    perm = np.argsort(img_names)
    w2c_mats = []
    bottom = np.array([[0, 0, 0, 1.]])
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)
    poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices
    return poses


def get_colmap_intrinsics(scene_name='bonsai'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/colmap/{}'.format(scene_name))
    camdata = read_cameras_binary(os.path.join(root_dir, 'sparse/0/cameras.bin'))
    fx, fy, cx, cy = camdata[1].params
    return fx, fy, cx, cy


def get_colmap_lerf_poses(scene_name='donuts'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/colmap_lerf/{}'.format(scene_name))
    imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))
    img_names = [imdata[k].name for k in imdata]
    perm = np.argsort(img_names)
    w2c_mats = []
    bottom = np.array([[0, 0, 0, 1.]])
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)
    poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices
    return poses


def get_colmap_lerf_intrinsics(scene_name='bonsai'):
    root_dir = os.path.join(ROOT_DIR, 'datasets/colmap_lerf/{}'.format(scene_name))
    camdata = read_cameras_binary(os.path.join(root_dir, 'sparse/0/cameras.bin'))
    fx, fy, cx, cy = camdata[1].params
    return fx, fy, cx, cy


def load_points3D(dataset='tnt', scene_name='Playground'):
    if dataset == 'tnt':
        root_dir = os.path.join(ROOT_DIR, 'datasets/colmap/playground/')
    elif dataset == 'lerf' or dataset == 'colmap_lerf':
        root_dir = os.path.join(ROOT_DIR, 'datasets/colmap/{}/'.format(scene_name))
    elif dataset == '360':
        root_dir = os.path.join(ROOT_DIR, 'datasets/colmap/{}/'.format(scene_name))
    elif dataset == 'colmap':
        root_dir = os.path.join(ROOT_DIR, 'datasets/colmap/{}/'.format(scene_name))
    else:
        raise NotImplementedError
    points3D_path = os.path.join(root_dir, 'sparse/0/points3D.bin')
    cloud_dict = read_points3D_binary(points3D_path)
    pcds = []
    for k in cloud_dict:
        cloud = cloud_dict[k]
        pcd = cloud.xyz
        pcd = pcd.reshape(-1, 3)
        pcds.append(pcd)
    pcds = np.vstack(pcds)
    return pcds


def normalize_poses(poses):
    '''
    Normalize poses to be within unit cube
    '''
    # compute normalized poses  
    scale = np.linalg.norm(poses[..., 0:3, 3], axis=-1).max()
    poses[:, 0:3, 3] /= scale
    return poses, scale


def get_up_vector(dataset='tnt', scene_name='Playground'):
    '''
    Get up vector of the scene
    '''
    up_vector = scene_up_vector_dict[dataset][scene_name]
    up_vector = up_vector / np.linalg.norm(up_vector)
    return up_vector


def align_poses(poses, up_vector):
    '''
    Align poses by rotation with up vector
    '''
    v1 = up_vector
    v2 = np.array([0., 0., 1.])
    R = get_rotation_matrix_from_vectors(v1, v2)
    # rotate c2w matrix by R
    for i in range(poses.shape[0]):
        poses[i] = R @ poses[i]
    return poses, R


def render_scaled_poses(dataset='tnt', scene_name='Playground', arrow_len=1, s=1):
    '''
    Render normalized camera poses
    '''
    if dataset == 'tnt':
        poses = get_tnt_poses(scene_name)
        # meshes = load_meshes('tnt', scene_name)
    elif dataset == 'lerf':
        poses = get_lerf_poses(scene_name)
    elif dataset == '360':
        poses = get_360_poses(scene_name)
    elif dataset == 'colmap_lerf':
        poses = get_colmap_lerf_poses(scene_name)
    else:
        raise NotImplementedError
    visualize_camera(poses, arrow_len, s)

    # compute normalized poses  
    poses, _ = normalize_poses(poses)
    visualize_camera(poses, arrow_len, s)
    # visualize_camera_and_meshes(poses, meshes, arrow_len, s)


def render_aligned_poses(dataset='tnt', scene_name='Playground', arrow_len=1, s=1):
    '''
    Render normalized and re-aligned camera poses
    '''
    if dataset == 'tnt':
        poses = get_tnt_poses(scene_name)
    elif dataset == 'lerf':
        poses = get_lerf_poses(scene_name)
    elif dataset == '360':
        poses = get_360_poses(scene_name)
    elif dataset == 'colmap_lerf':
        poses = get_colmap_lerf_poses(scene_name)
    elif dataset == 'colmap':
        poses = get_colmap_poses(scene_name)
    else:
        raise NotImplementedError
    
    ##### Visualize original camera poses #####
    # x, y, z = visualize_camera(poses, arrow_len, s)

    # up_vector = get_up_vector(dataset, scene_name)

    ##### Visualize up vector #####
    # center = np.mean(poses[..., 0:3, 3], axis=0)
    # up_arrow = vedo.Arrow(center, center + arrow_len * 5 * up_vector, c="r")
    # plt = vedo.Plotter()
    # plt.show(x, y, z, up_arrow, axes=1, viewup="z")

    ##### Get rotation matrix to align poses #####
    # poses, R = align_poses(poses, up_vector)

    ##### Visualize aligned camera poses #####
    # visualize_camera(poses, arrow_len, s)

    # compute normalized poses  
    # poses, scale = normalize_poses(poses)
    transform = getNerfppNorm(poses)
    t = transform["translate"]

    ##### Visualize aligned and normalized camera poses #####
    # visualize_camera(poses, arrow_len, s)

    points3D = load_points3D(dataset, scene_name)
    # points3D = (R @ points3D.T).T  # rotate and normalize points3D
    # points3D /= scale  # normalize points3D

    points3D = points3D + t
    poses[:, 0:3, 3] = poses[:, 0:3, 3] + t

    # camera_sample_info = scene_sampled_camera_dict[dataset][scene_name]
    # center_pos = np.array(camera_sample_info['center_pos'])
    center_pos = np.mean(poses[..., 0:3, 3], axis=0)
    print(center_pos - t)

    # crop out points3D within [-1, 1]
    points3D = points3D[np.all(np.abs(points3D) < 5, axis=-1)]

    ##### Visualize aligned and normalized camera poses with point clouds #####
    visualize_camera_and_points3D(poses, points3D, center_pos, arrow_len, s)


def normalize(vec):
    eps = np.finfo(float).eps
    normalized_vec = vec / (np.linalg.norm(vec)+eps)
    return normalized_vec


def rotm_from_lookat(lookat, up):
    z_axis = normalize(lookat)
    # x_axis = normalize(np.cross(up, z_axis))
    x_axis = normalize(np.cross(z_axis, up))
    y_axis = normalize(np.cross(z_axis, x_axis))
    R = np.array((x_axis, y_axis, z_axis)).T  # cv2world
    return R


def grid_half_sphere(radius=1.5, num_views=30, theta=None):
    if theta is None:
        theta = np.deg2rad(np.array((0, 15, 30, 45, 60)))
    else:
        theta = np.deg2rad([theta])
    phi = np.deg2rad(np.linspace(0, 360, num_views // len(theta)+1)[:-1])
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()
    x = np.cos(theta) * np.cos(phi) * radius
    y = np.cos(theta) * np.sin(phi) * radius
    z = np.sin(theta) * radius
    t = np.stack((x, y, z), axis=-1)
    return t


def render_sampled_poses(dataset='tnt', scene_name='Playground', arrow_len=1, s=1):
    
    ##### sample poses around the center of the scene #####
    camera_sample_info = scene_sampled_camera_dict[dataset][scene_name]
    center_pos = np.array(camera_sample_info['center_pos'])
    radius = camera_sample_info['radius']
    theta = camera_sample_info['theta']
    num_views = camera_sample_info['num_views']
    cam_pos_list = grid_half_sphere(radius=radius, num_views=num_views, theta=theta) + center_pos

    poses = []
    for t in cam_pos_list:
        lookat = center_pos - t
        R = rotm_from_lookat(lookat, np.array([0, 0, 1]))
        c2w = np.hstack((R, t.reshape(3, 1)))
        poses.append(c2w)
    poses = np.stack(poses, axis=0)


    ##### Get original camera poses #####
    if dataset == 'tnt':
        orig_poses = get_tnt_poses(scene_name)
    elif dataset == 'lerf':
        orig_poses = get_lerf_poses(scene_name)
    elif dataset == '360':
        orig_poses = get_360_poses(scene_name)
    elif dataset == 'colmap_lerf':
        orig_poses = get_colmap_lerf_poses(scene_name)
    else:
        raise NotImplementedError

    # rotate and normalize points3D
    up_vector = get_up_vector(dataset, scene_name)
    orig_poses, R = align_poses(orig_poses, up_vector)
    orig_poses, scale = normalize_poses(orig_poses)
    points3D = load_points3D(dataset, scene_name)
    points3D = (R @ points3D.T).T / scale  # rotate and normalize points3D

    visualize_camera_and_points3D(poses, points3D, center_pos, arrow_len, s)

    ##### get intrinsics #####
    h, w = camera_sample_info['hw']
    if dataset == 'tnt':
        fx, fy, cx, cy = get_tnt_intrinsics(scene_name)
    elif dataset == 'lerf':
        fx, fy, cx, cy = get_lerf_intrinsics(scene_name)
    elif dataset == '360':
        fx, fy, cx, cy = get_360_intrinsics(scene_name)
    elif dataset == 'colmap_lerf':
        fx, fy, cx, cy = get_colmap_lerf_intrinsics(scene_name)
    else:
        raise NotImplementedError

    ##### save poses and intrinsics to .json file #####
    output_folder = os.path.join(ROOT_DIR, 'datasets/{}/{}/custom_camera_path'.format(dataset, scene_name))
    os.makedirs(output_folder, exist_ok=True)
    transforms_dict = {
        "trajectory_name": "trajectory_001",
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
    }
    frames = [{"transform_matrix": c2w.tolist()} for c2w in poses]
    transforms_dict["frames"] = frames

    with open(os.path.join(output_folder, 'transforms_001.json'), 'w') as f:
        json.dump(transforms_dict, f, indent=4)


lerf_scenes = ['donuts', 'dozer_nerfgun_waldo', 'espresso', 'figurines', 'ramen', 'shoe_rack', 'teatime', 'waldo_kitchen']
tnt_scenes = ['Playground']
_360_scenes = ['bonsai', 'counter', 'garden']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='tnt, lerf, 360, colmap, colmap_lerf')
    parser.add_argument('-scene_name', help='scene name')
    parser.add_argument('-arrow_len', type=float, help='arrow length', default=0.1)
    args = parser.parse_args()

    print("dataset: {}".format(args.dataset), "scene_name: {}".format(args.scene_name))

    # render_scaled_poses(args.dataset, args.scene_name, args.arrow_len)

    render_aligned_poses(args.dataset, args.scene_name, args.arrow_len)

    # render_sampled_poses(args.dataset, args.scene_name, args.arrow_len)