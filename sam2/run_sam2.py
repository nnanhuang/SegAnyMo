import torch
from sam2.build_sam import build_sam2_video_predictor
import os
import shutil
import argparse
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from glob import glob
import json
import imageio
import pickle
from imageio import get_writer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def process_invisible_traj(traj, visible_mask, confidences, state, predictor, dilation_size=8, max_iterations=10, obj_id=0, timestep = 100, downscale_factor=None):
    iteration = 0  
    memory_dict = {} 
    take_all = False
    if traj.shape[1] <= 5:
        take_all = True

    while iteration < max_iterations:
        # 1. find max_visible_frame
        visible_points_per_frame = visible_mask.sum(axis=0)
        top_k = int(timestep*0.2)
        top_k_indices = np.argsort(visible_points_per_frame)[-top_k:][::-1]
        top_k_indices = sorted(top_k_indices)
        middle_index = len(top_k_indices) // 2
        t = top_k_indices[middle_index]
        if visible_points_per_frame[t] == 0:
            t = visible_points_per_frame.argmax()

        # 2. use process_points_with_memory , get new_memory_dict
        mem_pts,mem_confi,mem_vis, new_memory_dict, obj_id = process_points_with_memory(state, predictor, t, obj_id, 
                                    traj.transpose(1, 0, 2), confidences, visible_mask, dilation_size, take_all,
                                    downscale_factor)

        # 3. update memory_dict
        memory_dict.update(new_memory_dict)

        # 4. update traj and visible_mask 
        traj = mem_pts.transpose(1, 0, 2)  
        visible_mask = mem_vis  
        confidences = mem_confi 

        iteration += 1

        if traj.shape[1] < 6:
            break

    return memory_dict

def find_dense_pts(points):
    nbrs = NearestNeighbors(n_neighbors=points.shape[0]).fit(points)
    distances, indices = nbrs.kneighbors(points)
    density = np.sum(distances, axis=1)

    max_density_index = np.argmin(density) 
    max_density_point = points[max_density_index]
    max_density_point = np.expand_dims(max_density_point, axis=0)
    
    return max_density_point

def process_points_with_memory(state, predictor, t_cano, initial_obj_id, traj, confi_t, mask_t, dilation_size=3, take_all=False, downscale_factor=None):
    """
    Iteratively remove points that meet the criteria using a loop, 
    while storing object information using a memory mechanism.
    
    """
    obj_id_valid = initial_obj_id
    mem_pts = traj.copy()
    mem_vis = mask_t.copy()
    mem_confi = confi_t.copy()
    memory_dict = {}

    points_cano =  mem_pts[:,:,t_cano]
    vis_cano = mem_vis[:,t_cano]
    vis_point_cano = points_cano[vis_cano]
    remain_pts = vis_point_cano

    # 1. Find the centroid points
    nearest_point = find_dense_pts(remain_pts)

    predictor.reset_state(state)
    obj_id = 1
    labels = np.array([1], np.int32)

    # 2. Use predictor to add new points and obtain masks
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=t_cano,
        obj_id=obj_id,
        points=nearest_point,
        labels=labels,
    )

    # 3. Obtain the mask and perform dilation [1,480,854]
    mask_sam = (out_mask_logits[obj_id-1] > 0.0).cpu().numpy()
    
    dilated_mask = dilate_mask(mask_sam, dilation_size=dilation_size)
    
    in_mask = np.zeros(mem_pts.shape[0], dtype=bool)
    in_mask_all = find_pts_in_mask(dilated_mask, points_cano)
    in_mask[vis_cano] = in_mask_all[vis_cano]

    # The prompt mask needs to be visible + not dilated
    prompt_mask = np.zeros(mem_pts.shape[0], dtype=bool)
    prompt_mask_all = find_pts_in_mask(mask_sam, points_cano)
    prompt_mask[vis_cano] = prompt_mask_all[vis_cano]
    
    if args.vis:
        save_mask_with_points(mask_sam, nearest_point, "prompt.png")
        save_mask_with_points(dilated_mask, points_cano, "dilated_mask.png")
        save_mask_with_points(dilated_mask, points_cano[in_mask], "in_mask.png")

    # 5. If there are 5 or more points within the mask, store the object information
    include = judge(in_mask, mem_pts, mem_vis)
    if (np.sum(prompt_mask) > 0 and include) or take_all:
        obj_id_valid += 1
        memory_dict[obj_id_valid] = {
            'time': t_cano,
            'pts_trajs': mem_pts[prompt_mask],
            'confi_trajs': mem_confi[prompt_mask],
            'vis_trajs': mem_vis[prompt_mask],
            'num': int(in_mask.sum())
        }

    # 6. Retain points outside the mask (visible points outside mask) + invisible points

    mem_pts = mem_pts[~in_mask]
    mem_confi = mem_confi[~in_mask]
    mem_vis = mem_vis[~in_mask]

    return mem_pts,mem_confi,mem_vis, memory_dict, obj_id_valid

def judge(in_mask, mem_pts, mem_vis, thre=6):
    include = False
    if np.sum(in_mask) >= thre:
        include = True
        return include
    
    T = mem_pts.shape[-1]
    for t in range(T):
        points = mem_pts[:, :, t]
        v_mask = mem_vis[:, t]
        mask = v_mask & in_mask
        
        points_valid = points[mask]
        num = points_valid.shape[0]
        if num > thre:
            include = True
            return include
    return include

def dilate_mask(mask, dilation_size=5):
    if mask.shape[0] == 1:
        mask = mask[0]

    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
    dilated_mask = np.expand_dims(dilated_mask, axis=0)

    return dilated_mask

def cls_iou(pred, label, thres = 0.7):
    mask = (pred>thres).float()
    b, c, n = label.shape 
    # IOU
    intersect = (mask * label).sum(-1)
    union = mask.sum(-1) + label.sum(-1) - intersect
    iou = intersect / (union + 1e-12)
    return iou.mean()

def save_mask_with_points(mask, points, save_path, point_color='red', point_size=20):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
    fig, ax = plt.subplots()

    ax.imshow(mask_image)

    x_coords, y_coords = points[:, 0], points[:, 1]

    ax.scatter(x_coords, y_coords, color=point_color, s=point_size)

    ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def save_multi_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    for object_id, object_mask in per_obj_output_mask.items():
        object_name = f"{object_id:03d}"
        os.makedirs(
            os.path.join(output_mask_dir, video_name, object_name),
            exist_ok=True,
        )
        output_mask = object_mask.reshape(height, width).astype(np.uint8)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, object_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)

def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask

def load_data(dynamic_dir):
    '''
    return : [2, N, T] and [N, T], [N, T]
    '''
    track_path = os.path.join(dynamic_dir, "dynamic_traj.npy")
    visible_path = os.path.join(dynamic_dir, "dynamic_visibility.npy")
    confi_path = os.path.join(dynamic_dir, "dynamic_confidences.npy")
    traj = np.load(track_path)
    visible_mask = np.load(visible_path).astype(bool)
    confi = np.load(confi_path)
    
    return traj, visible_mask, confi

def apply_mask_to_rgb(rgb_image, mask_image):
    masked_rgb = np.zeros_like(rgb_image)
    
    masked_rgb[mask_image > 0] = rgb_image[mask_image > 0]
    
    return masked_rgb

def save_video_from_images3(rgb_images, mask_images, video_dir, fps=30):
    assert len(rgb_images) > 0 and len(mask_images) > 0, "图像列表不能为空"
    
    height, width, _ = rgb_images[0].shape
    os.makedirs(video_dir, exist_ok=True)
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")
    mask_rgb_color_video_path = os.path.join(video_dir, "mask_rgb_color.mp4")

    rgb_writer = get_writer(rgb_video_path, fps=fps)
    mask_writer = get_writer(mask_video_path, fps=fps)
    mask_rgb_writer = get_writer(mask_rgb_video_path, fps=fps)
    mask_rgb_color_writer = get_writer(mask_rgb_color_video_path, fps=fps)

    for rgb_img, mask_img in zip(rgb_images, mask_images):
        # Prepare mask image with white background
        mask_img_white_bg = np.ones_like(rgb_img) * 255  # Set background to white
        mask_img_white_bg[mask_img > 0] = rgb_img[mask_img > 0]  # Apply mask part only

        # Colored transparent overlay for mask_rgb_color
        colored_mask = rgb_img.copy()
        overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Example green color for mask
        alpha = 0.5  # Transparency level

        colored_mask[mask_img > 0] = (
            alpha * overlay_color + (1 - alpha) * rgb_img[mask_img > 0]
        ).astype(np.uint8)

        # Writing each frame to the video files
        rgb_writer.append_data(rgb_img)
        mask_writer.append_data(mask_img_white_bg)
        mask_rgb_writer.append_data(mask_img_white_bg)  # For mask_rgb with white background
        mask_rgb_color_writer.append_data(colored_mask)  # For mask_rgb_color with transparent overlay

    print(f'Videos saved to {video_dir}!')
    rgb_writer.close()
    mask_writer.close()
    mask_rgb_writer.close()
    mask_rgb_color_writer.close()

def save_video_from_images2(rgb_images, mask_images, video_dir, fps=30):
    assert len(rgb_images) > 0 and len(mask_images) > 0, "图像列表不能为空"
    
    height, width, _ = rgb_images[0].shape
    os.makedirs(video_dir, exist_ok=True)
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")

    rgb_writer = get_writer(rgb_video_path, fps=fps)
    mask_writer = get_writer(mask_video_path, fps=fps)
    mask_rgb_writer = get_writer(mask_rgb_video_path, fps=fps)

    for rgb_img, mask_img in zip(rgb_images, mask_images):
        mask_img = (mask_img.astype(np.uint8) * 255)
        mask_img_colored = np.stack([mask_img, mask_img, mask_img], axis=-1)  # Convert grayscale to RGB
        
        masked_img = apply_mask_to_rgb(rgb_img, mask_img)  # Assumed that this function already exists
        
        rgb_writer.append_data(rgb_img)
        mask_writer.append_data(mask_img_colored)
        mask_rgb_writer.append_data(masked_img)

    print(f'video saved to {mask_rgb_video_path}!')
    rgb_writer.close()
    mask_writer.close()
    mask_rgb_writer.close()

def save_video_from_images(rgb_images, mask_images, video_dir, fps=30):
    assert len(rgb_images) > 0 and len(mask_images) > 0, "image list cannot be empty"
    
    height, width, _ = rgb_images[0].shape
    os.makedirs(video_dir, exist_ok=True)
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")

    rgb_out = cv2.VideoWriter(rgb_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    mask_out = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    mask_rgb_out = cv2.VideoWriter(mask_rgb_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for rgb_img, mask_img in zip(rgb_images, mask_images):
        mask_img = mask_img.astype(np.uint8) * 255
        mask_img_colored = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        
        masked_img = apply_mask_to_rgb(rgb_img, mask_img)
        
        rgb_out.write(rgb_img)
        mask_out.write(mask_img_colored)
        mask_rgb_out.write(masked_img)

    rgb_out.release()
    mask_out.release()
    mask_rgb_out.release

def save_obj_video(args, obj_id, video_name, frame_names):
    obj_mask_dir = os.path.join(args.output_mask_dir, video_name,f'{obj_id:03d}')
    rgb_p_example = os.listdir(args.video_dir)[0] 
    if os.path.splitext(rgb_p_example)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]:
        suffix = os.path.splitext(rgb_p_example)[-1]
    else:
        suffix = os.path.splitext(os.listdir(args.video_dir)[1])[-1]
    obj_masks = []
    obj_rgbs = []
    for frame_name in frame_names:
        mask_path = os.path.join(obj_mask_dir, f"{frame_name}.png")
        input_mask, _ = load_ann_png(mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
        if len(per_obj_input_mask) ==0:
            obj_mask = np.zeros((input_mask.shape[0], input_mask.shape[1]), dtype=bool)
        else:
            obj_mask = per_obj_input_mask[1] # [540,960]
        obj_masks.append(obj_mask)
        rgb_path = os.path.join(args.video_dir, f'{frame_name}{suffix}')
        rgb_image = cv2.imread(rgb_path) # [540,960,3]
        obj_rgbs.append(rgb_image)
    
    video_dir = obj_mask_dir.replace("sam2", "sam2_video")
    save_video_from_images(obj_rgbs, obj_masks, video_dir)

def downsample_mask(mask_2d, scale_x, scale_y):
    # [H, W]
    new_width = int(mask_2d.shape[1] * scale_x)
    new_height = int(mask_2d.shape[0] * scale_y)
    downsampled_mask = cv2.resize(mask_2d, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return downsampled_mask

def find_pts_in_mask(mask, pts, downscale_factor=None):
    mask_2d = mask[0]
    if downscale_factor is not None:
        scale_x, scale_y = downscale_factor
        downsampled_mask = downsample_mask(mask_2d, scale_x, scale_y)
        
        scaled_x_coords = (pts[:, 0] * scale_x).astype(int)
        scaled_y_coords = (pts[:, 1] * scale_y).astype(int)
        
        scaled_x_coords = np.clip(scaled_x_coords, 0, downsampled_mask.shape[1] - 1)
        scaled_y_coords = np.clip(scaled_y_coords, 0, downsampled_mask.shape[0] - 1)

        in_mask = downsampled_mask[scaled_y_coords, scaled_x_coords]
    else:
        x_coords, y_coords = pts[:, 0].astype(int), pts[:, 1].astype(int)
        
        x_coords = np.clip(x_coords, 0, mask_2d.shape[1] - 1)
        y_coords = np.clip(y_coords, 0, mask_2d.shape[0] - 1)
        in_mask = mask_2d[y_coords, x_coords]

    in_mask = in_mask.astype(bool)
    
    return in_mask

def find_centroid_and_nearest_farthest(points):
    """
    First calculate the centroid of a given 2D point set, then find the point closest to the centroid and the one or two points farthest from the centroid.

    Parameters:
    points (numpy.ndarray): A 2D point set of shape (N, 2), where N is the number of points.

    Returns:
    nearest_point (numpy.ndarray): A point of shape (1, 2) that is closest to the centroid.
    nearest_index (int): The index of the point closest to the centroid.
    farthest_points (numpy.ndarray): A point or set of points of shape (n, 2) that is farthest from the centroid.
    farthest_indices (list): The indices of the point or set of points farthest from the centroid.
    centroid (numpy.ndarray): The coordinates of the centroid.
"""
    centroid = np.mean(points, axis=0)
    
    distances = np.linalg.norm(points - centroid, axis=1)
    
    nearest_index = np.argmin(distances)
    nearest_point = np.expand_dims(points[nearest_index], axis=0)  # (1, 2)
    
    num_points = len(points)
    
    if num_points == 1:
        farthest_points = nearest_point
        farthest_indices = [nearest_index]
    elif num_points == 2:
        farthest_indices = [i for i in range(num_points) if i != nearest_index]
        farthest_points = np.expand_dims(points[farthest_indices[0]], axis=0)  #  (1, 2)
    else:
        farthest_indices = np.argsort(distances)[-2:]  
        farthest_points = points[farthest_indices]  # (2, 2)
    
    return nearest_point, nearest_index, farthest_points, farthest_indices, centroid

def is_subset(mask1, mask2, coverage_threshold=0.9):
    mask1_area = (mask1 > 0).sum()
    intersection_area = np.logical_and(mask1 > 0, mask2 > 0).sum()
    
    if mask1_area == 0:
        return False  
    coverage_ratio = intersection_area / mask1_area
    return coverage_ratio >= coverage_threshold

def record_potential_merge(potential_merges, obj_id1, obj_id2):
    if obj_id1 not in potential_merges:
        potential_merges[obj_id1] = defaultdict(int)
    potential_merges[obj_id1][obj_id2] += 1

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def analyze_frame_merges(video_segments, iou_threshold=0.5):
    # potential_merges 
    potential_merges = {}
    frame_count = len(video_segments)
    
    for per_obj_output_mask in video_segments.values():
        visited = set()
        
        for obj_id1, mask1 in per_obj_output_mask.items():
            for obj_id2, mask2 in per_obj_output_mask.items():
                if obj_id1 != obj_id2 and (obj_id1, obj_id2) not in visited and (obj_id2, obj_id1) not in visited:
                    iou = compute_iou(mask1, mask2)
                    
                    if iou > iou_threshold or is_subset(mask1, mask2) or is_subset(mask2, mask1):
                        record_potential_merge(potential_merges, obj_id1, obj_id2)
                    visited.add((obj_id1, obj_id2))
    
    final_merges = defaultdict(list)
    for obj_id1, merge_counts in potential_merges.items():
        for obj_id2, count in merge_counts.items():
            if count / frame_count >= 0.3:  # If they overlap in 80% of the frames, they are considered the same object
                if obj_id2 not in final_merges[obj_id1]:
                    final_merges[obj_id1].append(obj_id2)
    
    # Group object IDs into unique sets representing the same object
    groups = []
    visited = set()
    
    for obj_id1, merged_ids in final_merges.items():
        if obj_id1 not in visited:
            current_group = set([obj_id1] + merged_ids)
            for obj_id2 in merged_ids:
                current_group.update(final_merges.get(obj_id2, []))
            groups.append(sorted(current_group))  # Add each unique group
            visited.update(current_group)
    
    # Include unmerged objects as individual groups
    all_obj_ids = {obj_id for frame in video_segments.values() for obj_id in frame.keys()}
    merged_obj_ids = {obj_id for group in groups for obj_id in group}
    unmerged_obj_ids = all_obj_ids - merged_obj_ids
    
    for obj_id in unmerged_obj_ids:
        groups.append([obj_id])
    
    # Map each group to a new unique object ID starting from 1
    result = {i + 1: group for i, group in enumerate(groups)}
    
    # Return result with unique groups
    return result
    
def merge_masks(video_segments, merge_groups):
    # Initialize merged_video_segments to have the same structure as video_segments
    merged_video_segments = {}

    # Iterate through each frame in the video_segments
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        merged_masks = {}

        # Iterate through each merge group to combine masks
        for new_obj_id, obj_ids in merge_groups.items():
            combined_mask = None
            
            for obj_id in obj_ids:
                # Retrieve the mask for each object ID in the group, if it exists in this frame
                mask = per_obj_output_mask.get(obj_id)
                if mask is not None:
                    # Initialize combined_mask if it's the first mask in this group
                    if combined_mask is None:
                        combined_mask = np.copy(mask)
                    else:
                        # Use element-wise maximum to merge masks
                        combined_mask = np.maximum(combined_mask, mask)
            
            # Assign the merged mask to the new object ID
            if combined_mask is not None:
                merged_masks[new_obj_id] = combined_mask
        
        # Save the merged masks for this frame in the same structure as video_segments
        merged_video_segments[out_frame_idx] = merged_masks

    return merged_video_segments

def main(args):
    video_name = os.path.basename(args.dynamic_dir)
    if "baseline" in args.output_mask_dir:
        output_mask_dir = (args.output_mask_dir)
        # output_mask_dir = os.path.dirname(args.output_mask_dir)
    else:
        output_mask_dir = os.path.join(args.output_mask_dir, "initial_preds")
    
    if not args.cal_only:
        checkpoint = "sam2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        frame_names = sorted([
            os.path.splitext(p)[0]
            for p in os.listdir(args.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ])
        # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        img_ext = os.listdir(args.video_dir)[0]
        img_path = os.path.join(args.video_dir, img_ext)
        with Image.open(img_path) as img:
            width, height = img.size  # 获取宽度和高度

        # load data
        traj, visible_mask, confidences = load_data(args.dynamic_dir)
        _, N, T = traj.shape
                
        q_ts = list(range(0, T, 8*2))
        max_iterations = max(len(q_ts), 5)
        max_iterations = min(max_iterations, 10)
        
        downscale_factor = None
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(args.video_dir)
            os.makedirs(args.output_mask_dir, exist_ok=True)
            memory_dict = process_invisible_traj(traj, visible_mask, confidences, state, predictor, dilation_size=6, max_iterations=max_iterations, timestep=T, downscale_factor=downscale_factor)
            
        video_segments = {}
        basename = os.path.basename(args.video_dir)            
                        
        if len(memory_dict)==0:
            print("[WARN]: {seq_name} don\'t have dynamic objects!")
            exit()

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for obj_id_valid, pkg in memory_dict.items():
            q_ts = list(range(0, T, 16))
            predictor.reset_state(state)
            time = pkg['time']
            if time in q_ts:
                q_ts.remove(time)  
            q_ts.insert(0, time) 
            pts_trajs = pkg['pts_trajs']
            confi_trajs = pkg['confi_trajs']
            vis_trajs = pkg['vis_trajs']
            
            require_reverse = True
            for t in q_ts:
                # [2,T], [T]
                pts = pts_trajs[:, :, t]
                vis = vis_trajs[:, t]
                
                visible_points = pts[vis]
                if visible_points.shape[0] == 0:
                    print(f'propogate in time {t} for object {obj_id_valid} \n')
                    continue
                if t == 0:
                    require_reverse = False
                # nearest_point, _, farthest_points, _, _ = find_centroid_and_nearest_farthest(visible_points)
                # prompt_points = np.concatenate((nearest_point, farthest_points), axis=0)
                prompt_points = find_dense_pts(visible_points)

                num = prompt_points.shape[0]
                labels = np.ones(num, dtype=np.int32)
                _, _, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=state,
                                                    frame_idx=t,
                                                    obj_id=obj_id_valid,
                                                    points=prompt_points,
                                                    labels=labels,
                                                )
                                
                prompt_mask = out_mask_logits[0] > 0.0
                # save_mask_with_points((prompt_mask).cpu().numpy(), visible_points, "test.png")

                in_mask = find_pts_in_mask(prompt_mask.cpu().numpy(), visible_points)
                if in_mask.sum() < (visible_points.shape[0] * 0.7):
                    nearest_point, _, farthest_points, _, _ = find_centroid_and_nearest_farthest(visible_points)
                    prompt_points = np.concatenate((nearest_point, farthest_points), axis=0)
                    # prompt_points = find_dense_pts(visible_points)

                    num = prompt_points.shape[0]
                    labels = np.ones(num, dtype=np.int32)
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=state,
                                                        frame_idx=t,
                                                        obj_id=obj_id_valid,
                                                        points=prompt_points,
                                                        labels=labels,
                                                    )

                    # save_mask_with_points((out_mask_logits[0] > 0.0).cpu().numpy(), prompt_points, "test2.png")
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    # 如果对象 ID 已经存在，我们更新它的 mask；如果不存在，则添加
                    if out_obj_id not in video_segments[out_frame_idx]:
                        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                    else:
                        # 如果需要，可以合并新的 mask 信息，而不是替换
                        video_segments[out_frame_idx][out_obj_id] = video_segments[out_frame_idx][out_obj_id] | (out_mask_logits[i] > 0.0).cpu().numpy()
            
            if require_reverse:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, reverse=True):
                    if out_frame_idx not in video_segments:
                        video_segments[out_frame_idx] = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        if out_obj_id not in video_segments[out_frame_idx]:
                            video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                        else:
                            video_segments[out_frame_idx][out_obj_id] = video_segments[out_frame_idx][out_obj_id] | (out_mask_logits[i] > 0.0).cpu().numpy()

        video_segments = dict(sorted(video_segments.items()))
        # write the output masks as palette PNG files to output_mask_dir
        save_dirname = os.path.join(output_mask_dir, basename)
        if os.path.exists(save_dirname):
            # If it exists, remove it
            shutil.rmtree(save_dirname)

        final_merges = analyze_frame_merges(video_segments, iou_threshold=0.9)
        merged_video_segments = merge_masks(video_segments, final_merges)

        for out_frame_idx, per_obj_output_mask in merged_video_segments.items():
            # after merge per_obj_output_mask
            save_multi_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
                output_palette=DAVIS_PALETTE,
            )
                    
    # load predictions
    frame_names = sorted([
        os.path.splitext(p)[0]
        for p in os.listdir(args.video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ])
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    sam_dir = os.path.join(output_mask_dir, video_name)
    # obj_ids =  sorted(os.listdir(sam_dir))
    if "baseline" in args.output_mask_dir:
        dynamic_path = os.path.join(os.path.dirname(args.output_mask_dir),"final_res",video_name)
    else:
        dynamic_path = os.path.join(args.output_mask_dir,"final_res",video_name)
    os.makedirs(dynamic_path, exist_ok=True)
    
    if "baseline" in args.output_mask_dir:
        seq_mask_dir = os.path.join(args.output_mask_dir, video_name)
    else:
        seq_mask_dir = os.path.join(args.output_mask_dir, "initial_preds",video_name)

    mask_paths = sorted(glob(os.path.join(seq_mask_dir, "*.png"))) + sorted(
            glob(os.path.join(seq_mask_dir, "*.jpg")) + sorted(glob(os.path.join(seq_mask_dir, "*.jpeg")))
    )
    
    all_mask = []
    for d_path in mask_paths:
        # d_path = os.path.join(sam_dir, f'{frame_name}.png')
        if not os.path.exists(d_path):
            mask_img = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_img, p = load_ann_png(d_path)
        mask_img = (mask_img > 0).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_img)
        all_mask.append(mask_tensor)
    predict_mask = torch.stack(all_mask, dim=0)

    # save video
    rgb_p_example = os.listdir(args.video_dir)[0] 
    if os.path.splitext(rgb_p_example)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]:
        suffix = os.path.splitext(rgb_p_example)[-1]
    else:
        suffix = os.path.splitext(os.listdir(args.video_dir)[1])[-1]

    rgbs = []
    for frame_name in frame_names:
        rgb_path = os.path.join(args.video_dir, f'{frame_name}{suffix}')
        rgb_image = cv2.imread(rgb_path) # [540,960,3]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgbs.append(rgb_image)

    save_dir = os.path.join(dynamic_path, "video")
    np_masks = [mask.numpy() for mask in all_mask]
    
    save_video_from_images3(rgbs, np_masks, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_mask_dir', type=str,default="exp_res/sam_res/main/FBMS-moving")
    parser.add_argument('--video_dir', type=str,default="current-data-dir/baseline/FBMS-moving/Testset/farm01")
    parser.add_argument('--dynamic_dir', type=str,default="exp_res/tracks_seg/main/FBMS-moving/farm01", help="motion segment result")
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--gt_dir', type=str,default="current-data-dir/davis/DAVIS/Annotations/480p/a")  
    parser.add_argument('--cal_only', action="store_true")
    
    args = parser.parse_args()
        
    main(args)