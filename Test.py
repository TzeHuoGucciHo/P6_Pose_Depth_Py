import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from easy_dwpose import DWposeDetector
from depth_anything_v2.dpt import DepthAnythingV2
import os

def draw_keypoint_labels(image, keypoints):
    draw = ImageDraw.Draw(image)
    keypoint_dict = {}
    normalized_keypoint_dict = {}

    for i, (x, y) in enumerate(keypoints):
        normalized_keypoint_dict[KEYPOINT_LABELS[i]] = (x, y)
        x_pixel, y_pixel = int(x * image.width), int(y * image.height)
        keypoint_dict[KEYPOINT_LABELS[i]] = (x_pixel, y_pixel)

        draw.ellipse((x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3), fill="red")
        draw.text((x_pixel + 5, y_pixel - 5), KEYPOINT_LABELS[i], fill="white")

    return image, keypoint_dict, normalized_keypoint_dict

def load_model(device, encoder='vits'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_model = DepthAnythingV2(**model_configs[encoder])
    model_path = os.path.join(os.path.expanduser("~"), "Depth-Anything-V2", "checkpoints",
                              f"depth_anything_v2_{encoder}.pth")
    depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return depth_model.to(device).eval()

def process_depth_image(raw_img, depth_model):
    depth = depth_model.infer_image(raw_img)
    depth_resized = cv2.resize(depth, (raw_img.shape[1], raw_img.shape[0]))
    depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
    return depth_normalized.astype('uint8')

def calculate_orientation(keypoint_dict):
    r_shoulder = keypoint_dict.get("RShoulder")
    l_shoulder = keypoint_dict.get("LShoulder")
    if r_shoulder and l_shoulder:
        return "front" if r_shoulder[0] < l_shoulder[0] else "back"
    return "undetermined"

def extract_keypoint_colors(image, keypoint_dict):
    keypoint_colors = {}
    for key, (x, y) in keypoint_dict.items():
        r, g, b = image.getpixel((x, y))
        keypoint_colors[key] = (r, g, b, f"#{r:02x}{g:02x}{b:02x}")
    return keypoint_colors

def save_results(images, output_folder, input_filename, log_data):
    os.makedirs(output_folder, exist_ok=True)
    for name, img in images.items():
        img.save(os.path.join(output_folder, f"{input_filename}_{name}.png"))
    with open(os.path.join(output_folder, f"{input_filename}_log.txt"), "w") as f:
        f.write(log_data)

def process_images_in_folder(input_folder):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    depth_model = load_model(device)
    dwpose = DWposeDetector(device=device)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            input_path = os.path.join(input_folder, filename)
            input_image = Image.open(input_path)
            raw_img = cv2.imread(input_path)

            keypoints_output = dwpose(input_image, return_keypoints=True)
            bodies = keypoints_output['bodies']
            pose_image = dwpose(input_image, output_type="pil")
            pose_image, keypoint_dict, normalized_keypoint_dict = draw_keypoint_labels(pose_image, bodies)
            depth_image = process_depth_image(raw_img, depth_model)
            depth_image_pil = Image.fromarray(depth_image)
            depth_image_pil, _, _ = draw_keypoint_labels(depth_image_pil, bodies)
            orientation = calculate_orientation(keypoint_dict)
            keypoint_colors = extract_keypoint_colors(input_image, keypoint_dict)

            log_data = "".join(
                f"{key} - Pixel: ({x}, {y}), RGB: {rgb[:3]}, HEX: {rgb[3]}\n" for key, (x, y), rgb in
                zip(keypoint_dict.keys(), keypoint_dict.values(), keypoint_colors.values())
            )
            log_data += "\n" + "\n".join(
                f"{key} - Normalized: ({x:.3f}, {y:.3f}), Pixel: ({x_pixel}, {y_pixel}), Depth: {depth_image[y_pixel, x_pixel]}"
                for key, (x, y) in normalized_keypoint_dict.items()
                for x_pixel, y_pixel in [keypoint_dict[key]]
            )
            log_data += f"\n\nEstimated Orientation: {orientation}\n"

            input_filename = os.path.splitext(filename)[0]
            output_folder = os.path.join(os.getcwd(), "1_Output_Images", input_filename)
            save_results({"overlay_image": input_image, "pose_image": pose_image, "depth_image": depth_image_pil}, output_folder, input_filename, log_data)
            print(f"Processed {filename}")

KEYPOINT_LABELS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
]

input_folder = os.path.join(os.getcwd(), "1_Input_Images")
process_images_in_folder(input_folder)