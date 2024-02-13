# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import glob
from scipy import misc
import sys
import json
import argparse
from colmap_read_model import *
import shutil

from read_dataset import *
from database import modify_db


def cmd(s):
  print(s)
  exit_code = os.system(s)
  if exit_code != 0:
    print("Error: ", exit_code)
    exit(exit_code)


def write_images_txt(poses, intrinsics, output_path, is_c2w=True):
  '''
  Write images.txt used for COLMAP
  Input:
    poses: dictionary of poses
    output_path: path to save images.txt
  '''
  # COLMAP format
  # 1st line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
  # 2nd line: leave it blank~~
  with open(output_path, 'w') as f:
    for idx, img_name in enumerate(sorted(poses.keys())):
      pose = poses[img_name]
      if is_c2w:
        pose = convert_c2w_to_w2c(pose)  # (3, 4)
      Q = rotmat2qvec(pose[:3, :3])
      T = pose[:3, 3]

      IMAGE_ID = idx+1
      QW, QX, QY, QZ = Q
      TX, TY, TZ = T
      if len(intrinsics) == 1:
        CAMERA_ID = 1
      else:
        CAMERA_ID = idx+1
      NAME = img_name

      f.writelines(f"{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME}\n")
      f.write('\n')


def write_cameras_txt(intrinsics, output_path, HEIGHT, WIDTH):
  '''
  Write cameras.txt used for COLMAP
  Input:
    intrinsics: (list of) 3 x 3 intrinsics matrix
    output_path: path to save cameras.txt
  '''
  # Camera list with one line of data per camera:
  #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
  with open(output_path, 'w') as f:
    for idx, intrinsics in enumerate(intrinsics):
      CAMERA_ID = idx+1
      MODEL = "PINHOLE"
      PARAMS = intrinsics
      f.writelines(f"{CAMERA_ID} {MODEL} {WIDTH} {HEIGHT} {PARAMS[0,0]} {PARAMS[1,1]} {PARAMS[0,2]} {PARAMS[1,2]}\n")
      f.write('\n')


def setup_tnt(dataset_dir, output_dir, HEIGHT, WIDTH):
  '''
    Setup Tanks and Temples dataset
  '''
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(output_dir + "/images", exist_ok=True)
  # copy images
  src_img_dir = os.path.join(dataset_dir, "rgb")
  dst_img_dir = os.path.join(output_dir, "images")
  for img in sorted(glob.glob(src_img_dir + "/*")):
    shutil.copy(img, dst_img_dir)
  # create cameras.txt
  intrinsics = read_tnt_intrinsics(dataset_dir)
  cams_txt_path = os.path.join(output_dir, "cameras.txt")
  write_cameras_txt(intrinsics, cams_txt_path, HEIGHT, WIDTH)
  # create images.txt
  poses = read_tnt_poses(dataset_dir)
  imgs_txt_path = os.path.join(output_dir, "images.txt")
  write_images_txt(poses, intrinsics, imgs_txt_path, is_c2w=True)


def setup_360(dataset_dir, output_dir, HEIGHT, WIDTH):
  '''
    Setup 360 dataset
  '''
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(output_dir + "/images", exist_ok=True)
  # copy images
  src_img_dir = os.path.join(dataset_dir, "images")
  dst_img_dir = os.path.join(output_dir, "images")
  for img in sorted(glob.glob(src_img_dir + "/*")):
    shutil.copy(img, dst_img_dir)
  # create cameras.txt
  intrinsics = read_360_intrinsics(dataset_dir)
  cams_txt_path = os.path.join(output_dir, "cameras.txt")
  write_cameras_txt(intrinsics, cams_txt_path, HEIGHT, WIDTH)
  # create images.txt
  poses = read_360_poses(dataset_dir)
  imgs_txt_path = os.path.join(output_dir, "images.txt")
  write_images_txt(poses, intrinsics, imgs_txt_path, is_c2w=False)


def setup_lerf(dataset_dir, output_dir, HEIGHT, WIDTH):
  '''
    Setup LERF dataset
  '''
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(output_dir + "/images", exist_ok=True)
  # copy images
  src_img_dir = os.path.join(dataset_dir, "images")
  dst_img_dir = os.path.join(output_dir, "images")
  for img in sorted(glob.glob(src_img_dir + "/*")):
    shutil.copy(img, dst_img_dir)
  # create cameras.txt and images.txt
  poses, intrinsics = read_lerf_poses_and_intrinsics(dataset_dir)
  cams_txt_path = os.path.join(output_dir, "cameras.txt")
  write_cameras_txt(intrinsics, cams_txt_path, HEIGHT, WIDTH)
  imgs_txt_path = os.path.join(output_dir, "images.txt")
  write_images_txt(poses, intrinsics, imgs_txt_path, is_c2w=True)

  # output transforms.json (for debugging)
  # transforms = {}
  # transforms["camera_model"] = "OPENCV"
  # transforms["orientation_override"] = "none"
  # frames_info = []
  # for idx, img_name in enumerate(sorted(poses.keys())):
  #   c2w = poses[img_name]
  #   K = intrinsics[idx]
  #   frame_info = {}
  #   frame_info["file_path"] = './images/' + img_name
  #   frame_info["transform_matrix"] = (np.concatenate([c2w @ np.diag([1, -1, -1, 1]), np.array([[0., 0., 0., 1.]])], axis=0)).tolist()  # OpenCV to OpenGL camera
  #   frame_info["fl_x"] = K[0, 0]
  #   frame_info["fl_y"] = K[1, 1]
  #   frame_info["cx"] = K[0, 2]
  #   frame_info["cy"] = K[1, 2]
  #   frame_info["w"] = WIDTH
  #   frame_info["h"] = HEIGHT
  #   frames_info.append(frame_info)
  # transforms["frames"] = frames_info
  # with open(output_dir + "/transforms.json", 'w') as f:
  #   json.dump(transforms, f, indent=4)



def runner(dataset_dir, output_dir):
  '''
    Use colmap with known poses and intrinsics
      1. feature extractor
      2. modify database
      3. exhaustive matcher
      4. point triangulation 
  '''
  # Feature Extraction  
  cmd("colmap feature_extractor \
    --database_path " + output_dir + "/database.db \
    --image_path " + output_dir + "/images \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1 " )
  
  # modify database
  modify_db(
    output_dir + "/database.db", \
    output_dir + "/images.txt", \
    output_dir + "/cameras.txt")

  # Feature Matching
  cmd("colmap exhaustive_matcher \
    --database_path " + output_dir + "/database.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.use_gpu 1")

  os.makedirs(output_dir + "/sparse/0", exist_ok=True)

  cmd("cp " + output_dir + "/images.txt " + output_dir + "/sparse/0/images.txt")
  cmd("cp " + output_dir + "/cameras.txt " + output_dir + "/sparse/0/cameras.txt")
  cmd("touch " + output_dir + "/sparse/0/points3D.txt")

  # Point Triangulation
  cmd("colmap point_triangulator \
    --database_path " + output_dir + "/database.db \
    --image_path " + output_dir + "/images \
    --input_path " + output_dir + "/sparse/0 \
    --output_path " + output_dir + "/sparse/0")

  # Convert .bin to .txt
  cmd("colmap model_converter \
    --input_path " + output_dir + "/sparse/0 \
    --output_path " + output_dir + "/sparse/0 \
    --output_type TXT")
  

def runner_with_mapper(dataset_dir, output_dir):
  '''
    Use colmap without any prior
      1. feature extractor
      2. exhaustive matcher
      3. Mapper
  '''
  # Feature Extraction (use single camera model)
  cmd("colmap feature_extractor \
    --database_path " + output_dir + "/database.db \
    --image_path " + output_dir + "/images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 " )

  # Feature Matching
  cmd("colmap exhaustive_matcher \
    --database_path " + output_dir + "/database.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.use_gpu 1")
  
  os.makedirs(output_dir + "/sparse/0", exist_ok=True)

  # Mapper (bundle adjustment)
  cmd("colmap mapper \
    --database_path " + output_dir + "/database.db \
    --image_path " + output_dir + "/images \
    --output_path " + output_dir + "/sparse \
    --Mapper.ba_global_function_tolerance=0.000001")

  # Convert .bin to .txt
  cmd("colmap model_converter \
    --input_path " + output_dir + "/sparse/0 \
    --output_path " + output_dir + "/sparse/0 \
    --output_type TXT")
  

if __name__ == '__main__':

  # # Playground (Tanks & Temples)
  # setup_tnt("../tnt/Playground", "./output/playground", 548, 1008)
  # runner("../tnt/Playground", "./output/playground")

  # # Bonsai (360)
  # setup_360("../360/bonsai", "./output/bonsai", 2078, 3118)
  # runner("../360/bonsai", "./output/bonsai")

  # # Counter (360)
  # setup_360("../360/counter", "./output/counter", 2076, 3115)
  # runner("../360/counter", "./output/counter")

  # # Garden (360)
  # setup_360("../360/garden", "./output/garden", 3361, 5187)
  # runner("../360/garden", "./output/garden")

  # Donuts (LERF)
  # setup_lerf("../lerf/donuts", "./output/donuts", 738, 994)
  # runner_with_mapper("../lerf/donuts", "./output/donuts")
  setup_360("../colmap_lerf/donuts", "../colmap/donuts", 738, 994)
  runner("../colmap_lerf/donuts", "../colmap/donuts")

  # # Dozer Nerfgun Waldo (LERF)
  # # setup_lerf("../lerf/dozer_nerfgun_waldo", "./output/dozer_nerfgun_waldo", 768, 1024)
  # # runner_with_mapper("../lerf/dozer_nerfgun_waldo", "./output/dozer_nerfgun_waldo")
  setup_360("../colmap_lerf/dozer_nerfgun_waldo", "../colmap/dozer_nerfgun_waldo", 768, 1024)
  runner("../colmap_lerf/dozer_nerfgun_waldo", "../colmap/dozer_nerfgun_waldo")

  # # Espresso (LERF)
  # # setup_lerf("../lerf/espresso", "./output/espresso", 738, 994)
  # # runner_with_mapper("../lerf/espresso", "./output/espresso")
  setup_360("../colmap_lerf/espresso", "../colmap/espresso", 738, 994)
  runner("../colmap_lerf/espresso", "../colmap/espresso")

  # # Figurines (LERF)
  # # setup_lerf("../lerf/figurines", "./output/figurines", 738, 994)
  # # runner_with_mapper("../lerf/figurines", "./output/figurines")
  setup_360("../colmap_lerf/figurines", "../colmap/figurines", 738, 994)
  runner("../colmap_lerf/figurines", "../colmap/figurines")

  # # Ramen (LERF)
  # # setup_lerf("../lerf/ramen", "./output/ramen", 738, 994)
  # # runner_with_mapper("../lerf/ramen", "./output/ramen")
  setup_360("../colmap_lerf/ramen", "../colmap/ramen", 738, 994)
  runner("../colmap_lerf/ramen", "../colmap/ramen")

  # # Shoe Rack (LERF)
  # # setup_lerf("../lerf/shoe_rack", "./output/shoe_rack", 738, 994)
  # # runner_with_mapper("../lerf/shoe_rack", "./output/shoe_rack")
  setup_360("../colmap_lerf/shoe_rack", "../colmap/shoe_rack", 738, 994)
  runner("../colmap_lerf/shoe_rack", "../colmap/shoe_rack")

  # # Teatime (LERF)
  # # setup_lerf("../lerf/teatime", "./output/teatime", 738, 994)
  # # runner_with_mapper("../lerf/teatime", "./output/teatime")
  setup_360("../colmap_lerf/teatime", "../colmap/teatime", 738, 994)
  runner("../colmap_lerf/teatime", "../colmap/teatime")

  # # Waldo Kitchen (LERF)
  # # setup_lerf("../lerf/waldo_kitchen", "./output/waldo_kitchen", 738, 994)
  # # runner_with_mapper("../lerf/waldo_kitchen", "./output/waldo_kitchen")
  setup_360("../colmap_lerf/waldo_kitchen", "../colmap/waldo_kitchen", 738, 994)
  runner("../colmap_lerf/waldo_kitchen", "../colmap/waldo_kitchen")
