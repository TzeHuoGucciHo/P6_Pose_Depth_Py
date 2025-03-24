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

def process_depth_image(raw_img, depth_model, device, input_width, input_height):
    depth = depth_model.infer_image(raw_img)

    input_height, input_width = raw_img.shape[:2]
    depth_height, depth_width = depth.shape[:2]
    scale_factor = min(input_width / depth_width, input_height / depth_height)

    new_depth_width = int(depth_width * scale_factor)
    new_depth_height = int(depth_height * scale_factor)

    depth_resized = cv2.resize(depth, (new_depth_width, new_depth_height))

    top_padding = (input_height - new_depth_height) // 2
    bottom_padding = input_height - new_depth_height - top_padding
    left_padding = (input_width - new_depth_width) // 2
    right_padding = input_width - new_depth_width - left_padding

    depth_padded = cv2.copyMakeBorder(
        depth_resized, top_padding, bottom_padding, left_padding, right_padding,
        cv2.BORDER_CONSTANT, value=0
    )

    depth_normalized = cv2.normalize(depth_padded, None, 0, 255, cv2.NORM_MINMAX)
    return depth_normalized.astype('uint8')

def calculate_orientation(keypoint_dict):
    # Get the coordinates of the shoulders
    r_shoulder = keypoint_dict.get("RShoulder")
    l_shoulder = keypoint_dict.get("LShoulder")

    if r_shoulder and l_shoulder:
        r_shoulder_x, _ = r_shoulder
        l_shoulder_x, _ = l_shoulder

        # If the right shoulder is to the left of the left shoulder in the image, the person is facing front
        return "front" if r_shoulder_x < l_shoulder_x else "back"

    print("Could not detect shoulder keypoints.")
    return "undetermined"

def extract_keypoint_colors(image, keypoint_dict):
    keypoint_colors = {}
    for key, (x_pixel, y_pixel) in keypoint_dict.items():
        r, g, b = image.getpixel((x_pixel, y_pixel))
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        keypoint_colors[key] = (r, g, b, hex_color)
        print(f"{key} - Pixel: ({x_pixel}, {y_pixel}), RGB: ({r}, {g}, {b}), HEX: {hex_color}")
    return keypoint_colors

def save_results(input_image, pose_image, result_image, depth_image_pil, output_folder, input_filename):
    os.makedirs(output_folder, exist_ok=True)

    input_image.save(os.path.join(output_folder, f"{input_filename}_overlay_image.png"))
    pose_image.save(os.path.join(output_folder, f"{input_filename}_pose_image.png"))
    result_image.save(os.path.join(output_folder, f"{input_filename}_final_result.png"))
    depth_image_pil.save(os.path.join(output_folder, f"{input_filename}_depth_image.png"))

def main(input_image_path):
    # Load image
    input_image = Image.open(input_image_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dwpose = DWposeDetector(device=device)
    keypoints_output = dwpose(input_image, return_keypoints=True)
    bodies = keypoints_output['bodies']
    pose_image = dwpose(input_image, output_type="pil")

    # Draw keypoints
    pose_image, keypoint_dict, normalized_keypoint_dict = draw_keypoint_labels(pose_image, bodies)

    # Load Depth model
    depth_model = load_model(device)

    # Read raw image
    raw_img = cv2.imread(input_image_path)

    # Process depth image
    depth_image = process_depth_image(raw_img, depth_model, device, raw_img.shape[1], raw_img.shape[0])

    # Create depth image for drawing
    depth_image_pil = Image.fromarray(depth_image)

    # Draw keypoints on depth image
    depth_image_pil, _, _ = draw_keypoint_labels(depth_image_pil, bodies)

    # Print keypoint info
    for key, (x, y) in normalized_keypoint_dict.items():
        x_pixel, y_pixel = int(x * raw_img.shape[1]), int(y * raw_img.shape[0])
        depth_value = depth_image[y_pixel, x_pixel]
        print(f"{key} - Normalized: ({x:.3f}, {y:.3f}), Pixel: ({x_pixel}, {y_pixel}), Depth: {depth_value}")

    orientation = calculate_orientation(keypoint_dict)

    # Color information for keypoints
    keypoint_colors = extract_keypoint_colors(input_image, keypoint_dict)

    # Prepare images for saving
    overlay_image, _, _ = draw_keypoint_labels(input_image, bodies)
    pose_image_resized = pose_image.resize(overlay_image.size)
    overlay_image_rgba = overlay_image.convert("RGBA")
    pose_image_rgba = pose_image_resized.convert("RGBA")
    result = cv2.addWeighted(np.array(overlay_image_rgba), 1, np.array(pose_image_rgba), 1, 0)
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # Save results
    input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
    output_folder = os.path.join(os.getcwd(), "1_Output_Images", input_filename)

    save_results(input_image, pose_image, result_pil, depth_image_pil, output_folder, input_filename)

    print("\nEstimated Orientation:", orientation)

KEYPOINT_LABELS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
]

# Run main function with input image path
input_image_path = os.path.join(os.getcwd(), "1_Input_Images", "person_1.jpg")

main(input_image_path)
