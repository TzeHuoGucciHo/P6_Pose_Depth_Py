import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from easy_dwpose import DWposeDetector
from depth_anything_v2.dpt import DepthAnythingV2
import os

# --- Script 1: Pose Detection ---

input_image_path = os.path.join(os.getcwd(), "1_Input_Images", "person_0.jpg")
input_image = Image.open(input_image_path)

# Keypoint Labels
KEYPOINT_LABELS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
]

def draw_keypoint_labels(image, keypoints):
    draw = ImageDraw.Draw(image)  # Initialize drawing object
    keypoint_dict = {}  # Dictionary for denormalized (pixel) keypoints
    normalized_keypoint_dict = {}  # Dictionary for normalized keypoints


    # Loop through for each keypoint coordinate
    for i, (x, y) in enumerate(keypoints):
        # Store normalized keypoint values
        normalized_keypoint_dict[KEYPOINT_LABELS[i]] = (x, y)

        # Convert normalized (0-1) to denormalized (pixel) coordinates
        x_pixel, y_pixel = int(x * image.width), int(y * image.height)

        # Store denormalized (pixel) keypoints
        keypoint_dict[KEYPOINT_LABELS[i]] = (x_pixel, y_pixel)

        # Draw keypoints and labels using the denormalized coordinates
        draw.ellipse((x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3), fill="red")
        # draw.text((x_pixel + 5, y_pixel - 5), KEYPOINT_LABELS[i], fill="white")

    return image, keypoint_dict, normalized_keypoint_dict

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dwpose = DWposeDetector(device=device)

keypoints_output = dwpose(input_image, return_keypoints=True)
bodies = keypoints_output['bodies'] # Extract body keypoints

# Get pose
pose_image = dwpose(input_image, output_type="pil")

# Annotate pose image with keypoint labels
pose_image, keypoint_dict, normalized_keypoint_dict = draw_keypoint_labels(pose_image, bodies)

# --- Script 2: Depth Detection ---

# Each model variant has different encoder sizes, feature dimensions, and output channels
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Selects the model variant to use (smallest is 'vits', largest is 'vitg')
encoder = 'vits'

# Initializes the depth estimation model with the selected encoder configuration
depth_model = DepthAnythingV2(**model_configs[encoder])

# Loads pre-trained model weights from the specified file. Change the path accordingly.
model_path = os.path.join(os.path.expanduser("~"), "Depth-Anything-V2", "checkpoints", "depth_anything_v2_vits.pth")
depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))

# Moves the model to the selected device (CPU/GPU) so that it infer the depth map
depth_model = depth_model.to(device).eval()

raw_img = cv2.imread(input_image_path)

# This function outputs a 2D depth map as a NumPy array
depth = depth_model.infer_image(raw_img)

# Extracts the original height and width of the input image
input_height, input_width = raw_img.shape[:2]

# Extracts the height and width of the depth map produced by the model
depth_height, depth_width = depth.shape[:2]

# Computes the scaling factor to resize the depth map to match the input image size
scale_factor = min(input_width / depth_width, input_height / depth_height)

# Computes the new dimensions of the depth map after scaling
new_depth_width = int(depth_width * scale_factor)
new_depth_height = int(depth_height * scale_factor)

# Resizes the depth map to the computed dimensions
depth_resized = cv2.resize(depth, (new_depth_width, new_depth_height))

# Calculates the padding needed to match the input image size, if necessary
top_padding = (input_height - new_depth_height) // 2
bottom_padding = input_height - new_depth_height - top_padding
left_padding = (input_width - new_depth_width) // 2
right_padding = input_width - new_depth_width - left_padding

# Adds black border padding which is done to avoid distortion or cropping while keeping the depth information aligned
depth_padded = cv2.copyMakeBorder(
    depth_resized, top_padding, bottom_padding, left_padding, right_padding,
    cv2.BORDER_CONSTANT, value=0
)

# Scales depth values to the range [0, 255] for better visualization
depth_normalized = cv2.normalize(depth_padded, None, 0, 255, cv2.NORM_MINMAX)

# Converts the depth map to an 8-bit for compatibility with other image processing functions
depth_normalized = depth_normalized.astype('uint8')

# --- Combine Depth and Pose Information ---

# Convert depth image to a PIL Image to overlay the keypoints
depth_image = Image.fromarray(depth_normalized)

# Overlay pose on the depth map
pose_image_pil, keypoint_dict, normalized_keypoint_dict = draw_keypoint_labels(depth_image, bodies)

# Print depth information for each keypoint using normalized coordinates
for key, (x, y) in normalized_keypoint_dict.items():
    x_pixel, y_pixel = int(x * input_width), int(y * input_height) # Convert normalized to pixel coordinates
    depth_value = depth_padded[y_pixel, x_pixel]  # Get depth using pixel coordinates

    # Print both normalized, pixel, and depth values for each keypoint
    print(f"{key} - Normalized: ({x:.3f}, {y:.3f}), Pixel: ({x_pixel}, {y_pixel}), Depth: {depth_value}")

# --- Script 3: Orientation Detection ---

# Hardcoded approach to find the orientation of the person based on the position of the arms color-coded in output image
pose_image_np = np.array(pose_image) # Convert the annotated image to a NumPy array for image processing

# Define color thresholds for green (left arm) and orange (right arm), fairly arbitrary trial and error values
# In BGR format (OpenCV convention)
lower_green = np.array([0, 200, 0])
upper_green = np.array([50, 255, 50])

lower_orange = np.array([200, 100, 0])
upper_orange = np.array([255, 200, 50])

# Create masks for green and orange pixels for the arms
mask_green = cv2.inRange(pose_image_np, lower_green, upper_green)
mask_orange = cv2.inRange(pose_image_np, lower_orange, upper_orange)

# Stack the coordinates of green and orange pixels detected in the masks
green_coords = np.column_stack(np.where(mask_green > 0))
orange_coords = np.column_stack(np.where(mask_orange > 0))

# Set the initial orientation to undetermined
orientation = "undetermined"

# Sometimes both arms can't be detected, so we need to check if they exist before comparing their positions
if green_coords.size and orange_coords.size:

    # Compute the average x position of green and orange pixels, respectively
    avg_green_x = np.mean(green_coords[:, 1])
    avg_orange_x = np.mean(orange_coords[:, 1])

    # If the average x position of green pixels is greater than that of orange pixels, the orientation is front
    orientation = "front" if avg_green_x > avg_orange_x else "back"
else:
    print("Could not detect colored segments for arms. Check color thresholds.")

# --- Script 4: Keypoint Color Extraction ---

# Dictionary to store keypoint colors
keypoint_colors = {}

for key, (x_pixel, y_pixel) in keypoint_dict.items():
    # Get the RGB color at the keypoint's pixel coordinates
    r, g, b = input_image.getpixel((x_pixel, y_pixel))

    # Convert to HEX format
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)

    # Store the color information
    keypoint_colors[key] = (r, g, b, hex_color)

    print(f"{key} - Pixel: ({x_pixel}, {y_pixel}), RGB: ({r}, {g}, {b}), HEX: {hex_color}")

# --- Output ---

# Show the keypoints on the input image
overlay_image, keypoint_dict, normalized_keypoint_dict = draw_keypoint_labels(input_image, bodies)
#overlay_image.show()

# Show the pose image with keypoints labeled
#pose_image.show()

# Show the final image with keypoints overlaid on the depth map
#pose_image_pil.show()

# Resize pose_image to match overlay_image size (if necessary)
pose_image_resized = pose_image.resize(overlay_image.size)

# Convert both images to the same mode (RGBA)
overlay_image_rgba = overlay_image.convert("RGBA")
pose_image_rgba = pose_image_resized.convert("RGBA")

# Now you can overlay the images
result = cv2.addWeighted(np.array(overlay_image_rgba), 1, np.array(pose_image_rgba), 1, 0)

# Convert the numpy ndarray result back to a PIL image
result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

# Show the result
#result_pil.show()

# Extract the filename from the input image path and remove the extension
input_filename = os.path.splitext(os.path.basename(input_image_path))[0]

# Define the output folder path based on the input image path
output_folder = os.path.join(os.getcwd(), "1_Output_Images", input_filename)

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the overlay image
overlay_image.save(os.path.join(output_folder, f"{input_filename}_overlay_image.png"))

# Save the pose image
pose_image.save(os.path.join(output_folder, f"{input_filename}_pose_image.png"))

# Save the pose image with keypoints on the depth map
pose_image_pil.save(os.path.join(output_folder, f"{input_filename}_pose_image_depth_map.png"))

# Save the result image (overlayed image)
result_pil.save(os.path.join(output_folder, f"{input_filename}_final_result.png"))

# Print the estimated orientation of the person
print("\n", f"Estimated Orientation: {orientation}")

# For getting specific keypoint positions. See KEYPOINT_LABELS for the labels.
# print("\n", "Nose", keypoint_dict["Nose"])
