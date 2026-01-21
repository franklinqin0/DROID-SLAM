import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def load_associations(assoc_file):
    """
    Load associations from TUM RGB-D associate output.
    Format per line: <rgb_ts> <rgb_rel_path> <depth_ts> <depth_rel_path>
    Returns: list of (rgb_ts, rgb_path, depth_ts, depth_path)
    """
    associations = []
    with open(assoc_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                # rgb_ts = float(parts[0])
                rgb_ts = parts[0]
                rgb_path = parts[1]
                # depth_ts = float(parts[2])
                depth_ts = parts[2]
                depth_path = parts[3]
                associations.append((rgb_ts, rgb_path, depth_ts, depth_path))
    return associations


def image_stream(datapath, rgb_path, depth_path, calib, stride, associations=None):
    """ Combined RGB / RGB-D image generator """
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # If associations file is provided, drive the pairing/order from it
    assoc_list = None
    if associations is not None:
        assoc_file = associations if os.path.isabs(associations) else os.path.join(datapath, associations)
        assoc_list = load_associations(assoc_file)

    if assoc_list is None:
        image_list = sorted(glob.glob(os.path.join(datapath, rgb_path, '*.png')))[::stride]
        depth_list = sorted(glob.glob(os.path.join(datapath, depth_path, '*.png')))[::stride] if depth_path is not None else None

        for t, imfile in enumerate(image_list):
            image = cv2.imread(imfile)

            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            if depth_path is not None:
                dfile = depth_list[t]
                depth = cv2.imread(dfile, cv2.IMREAD_ANYDEPTH) / 1000.0
                depth = torch.as_tensor(depth)
                depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
                depth = depth[:h1-h1%8, :w1-w1%8]
                yield t, image[None], depth, intrinsics
            else:
                yield t, image[None], intrinsics
    else:
        # Associations mode
        for (rgb_ts, rgb_rel, depth_ts, depth_rel) in assoc_list[::stride]:
            rgb_file = rgb_rel if os.path.isabs(rgb_rel) else os.path.join(datapath, rgb_rel)
            image = cv2.imread(rgb_file)

            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            if depth_path is not None:
                dfile = depth_rel if os.path.isabs(depth_rel) else os.path.join(datapath, depth_rel)
                depth = cv2.imread(dfile, cv2.IMREAD_ANYDEPTH) / 1000.0
                depth = torch.as_tensor(depth)
                depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
                depth = depth[:h1-h1%8, :w1-w1%8]
                yield rgb_ts, image[None], depth, intrinsics
            else:
                yield rgb_ts, image[None], intrinsics


def save_reconstruction(droid, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, help="path to RGB and depth parent directory")
    parser.add_argument("--rgb_dir", type=str, default="color_frames", help="path to RGB image directory")
    parser.add_argument("--depth_dir", type=str, default="depth_frames", help="path to depth image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--save_traj", type=str, default=None,
                        help="path to save dense trajectory in ORB-SLAM3 format (CameraTrajectory.txt)")
    parser.add_argument("--use_depth", action="store_true", help="enable RGB-D mode (requires depth_frames)")
    parser.add_argument("--associations", type=str, default=None,
                        help="TUM RGB-D association file (relative to base_dir or absolute)")
    args = parser.parse_args()

    if not args.use_depth:
        args.depth_dir = None

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    rgbd_stream = image_stream(args.base_dir, args.rgb_dir, args.depth_dir if args.use_depth else None,
                               args.calib, args.stride, associations=args.associations)
    for item in tqdm(rgbd_stream):
        if args.depth_dir is not None:
            t, image, depth, intrinsics = item
        else:
            t, image, intrinsics = item
            depth = None

        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
        
        droid.track(t, image, depth, intrinsics=intrinsics)

    # Fill poses for every frame (dense trajectory) - always use RGB stream
    rgb_stream = image_stream(args.base_dir, args.rgb_dir, None,
                              args.calib, args.stride, associations=args.associations)
    traj_est = droid.terminate(rgb_stream)

    # Optionally save dense trajectory in ORB-SLAM3 format: timestamp tx ty tz qx qy qz qw
    if args.save_traj is not None:
        import numpy as np
        import torch
        # If associations provided, use their timestamps as the first column; else use filename (no extension)
        ts_list = None
        if args.associations is not None:
            assoc_file = args.associations if os.path.isabs(args.associations) else os.path.join(args.base_dir, args.associations)
            assoc_list = load_associations(assoc_file)
            ts_list = [rgb_ts for (rgb_ts, _, _, _) in assoc_list]
        else:
            image_dir = os.path.join(args.base_dir, args.rgb_dir)
            full_image_list = sorted(os.listdir(image_dir))
            ts_list = [os.path.splitext(fn)[0] for fn in full_image_list]

        # Expect traj shape (N, 7): [tx, ty, tz, qx, qy, qz, qw]
        if isinstance(traj_est, dict) and "poses" in traj_est:
            traj = traj_est["poses"]
        else:
            traj = traj_est
        if torch.is_tensor(traj):
            traj = traj.cpu().numpy()
        traj = np.asarray(traj)
        assert traj.ndim == 2 and traj.shape[1] == 7, f"Unexpected traj_est shape {traj.shape}"

        with open(args.save_traj, "w") as f:
            for i, p in enumerate(traj):
                tx, ty, tz = p[:3]
                qx, qy, qz, qw = p[3:]
                tstr = ts_list[i] if i < len(ts_list) else f"{i:.6f}"
                f.write(f"{tstr} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        print(f"Camera trajectory saved to {args.save_traj}")

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
        print(f"Reconstruction saved to {args.reconstruction_path}")
