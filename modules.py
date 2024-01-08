
import numpy as np

######################################################################################

from PIL import Image

def read_image_as_grayscale_then_MinMax_normalize(image_path):

	image = Image.open(image_path).convert('L')
	img_array = np.asarray(image, dtype=np.float32)
	
	min_val = img_array.min()
	max_val = img_array.max()
	
	if max_val - min_val > 0:
		normalized_img = (img_array - min_val) / (max_val - min_val)
	else:
		raise ValueError('Image has no variance, it might be empty or corrupted')
	
	return normalized_img

######################################################################################

import matplotlib
import colorsys

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):

	h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
	cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)], axis=0)
	cols[0] = 0

	random_label_cmap = matplotlib.colors.ListedColormap(cols)

	return random_label_cmap

######################################################################################

import cv2

def smooth_segmented_labels(image):
	"""
	Smooths the segmented labels in an image using convex hull and replaces each label with its smoothed version.

	Parameters:
		image (numpy.ndarray): The input segmented image array (grayscale).

	Returns:
		numpy.ndarray: The image array with smoothed labels, as uint16.
	"""
	# Ensure the image is in the correct format
	if len(image.shape) != 2:
		raise ValueError("Input image must be a grayscale image")

	# Initialize the output image
	smoothed_image = np.zeros_like(image)

	# Unique labels in the image (excluding background)
	unique_labels = np.unique(image)
	unique_labels = unique_labels[unique_labels != 0]

	for label in unique_labels:
		# Create a binary mask for the current label
		label_mask = (image == label).astype(np.uint8)

		# Find contours for this label
		contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Calculate convex hull for each contour and draw it in the output image
		for contour in contours:
			hull = cv2.convexHull(contour)
			cv2.drawContours(smoothed_image, [hull], -1, int(label), thickness=cv2.FILLED)

	return smoothed_image.astype(np.uint16)  # Ensure output is uint16

######################################################################################

from skimage import measure

def compile_label_info(predicted_labels, window_coords, min_area_threshold = 20):
	all_labels_info = []
	new_label_id = 1  # Initialize label ID counter

	for patch_labels, (x_offset, y_offset, _, _) in zip(predicted_labels, window_coords):
		for region in measure.regionprops(patch_labels):
			area = region.area

			if area > min_area_threshold:
				# Calculate global centroid
				local_centroid_y, local_centroid_x = region.centroid
				global_center_x = local_centroid_x + x_offset
				global_center_y = local_centroid_y + y_offset

				# Calculate global bounding box
				minr, minc, maxr, maxc = region.bbox
				global_bbox = (minc + x_offset, minr + y_offset, maxc + x_offset, maxr + y_offset)

				# Extract the binary image of the label within its local bounding box
				label_image = new_label_id * region.image.astype(int)

				label_info = {
					'label_id': new_label_id,
					'global_centroid': (global_center_x, global_center_y),
					'global_bbox': global_bbox,
					'label_image': label_image
				}
				all_labels_info.append(label_info)

				new_label_id += 1  # Increment label ID for the next entry

	return all_labels_info

######################################################################################

def patchify(image, window_size, overlap):
	height, width = image.shape
	stride = window_size - overlap

	x_coords = list(range(0, width - window_size + 1, stride))
	y_coords = list(range(0, height - window_size + 1, stride))

	if x_coords[-1] != width - window_size:
		x_coords.append(width - window_size)
	if y_coords[-1] != height - window_size:
		y_coords.append(height - window_size)

	windows = []
	window_coords = []

	for y in y_coords:
		for x in x_coords:
			window = image[y:y+window_size, x:x+window_size]
			windows.append(window)
			window_coords.append((x, y, x+window_size, y+window_size))

	windows = np.asarray(windows)

	return windows, window_coords

######################################################################################

def remove_border_labels(label_image, patch_coords, original_image, neutral_value=0):
	"""
	Remove labels at the borders of the patch, unless the border is shared with the original image.

	Parameters:
	- label_image: An array where each connected region is assigned a unique integer label.
	- patch_coords: A tuple (x1, y1, x2, y2) indicating the coordinates of the patch within the original image.
	- image_shape: The shape (height, width) of the original image.
	- neutral_value: The value to assign to removed labels.

	Returns:
	- An array of the same shape as `label_image` with the appropriate border labels removed.
	"""
	x1, y1, x2, y2 = patch_coords
	height, width = original_image.shape
	border_labels = set()

	# Check each border and add labels to remove if not at the edge of the original image
	if y1 != 0:
		border_labels.update(label_image[0, :])
	if y2 != height:
		border_labels.update(label_image[-1, :])
	if x1 != 0:
		border_labels.update(label_image[:, 0])
	if x2 != width:
		border_labels.update(label_image[:, -1])

	# Create a mask of regions to remove
	border_mask = np.isin(label_image, list(border_labels))

	# Set the border labels to neutral_value (background)
	cleaned_label_image = label_image.copy()
	cleaned_label_image[border_mask] = neutral_value

	return cleaned_label_image

######################################################################################

def non_maximum_suppression(boxes, overlapThresh):
	if len(boxes) == 0:
		return []

	# Initialize the list of selected indices
	selected_indices = []

	# Sort the boxes by the bottom-right y-coordinate (y2)
	sorted_indices = np.argsort([box[3] for box in boxes])

	while len(sorted_indices) > 0:
		# Select the box with the largest y2 and remove it from sorted_indices
		current_index = sorted_indices[-1]
		selected_indices.append(current_index)
		sorted_indices = sorted_indices[:-1]

		for other_index in sorted_indices.copy():
			if does_overlap(boxes[current_index], boxes[other_index], overlapThresh):
				# Remove indices that overlap too much with the current box
				sorted_indices = sorted_indices[sorted_indices != other_index]

	return selected_indices

######################################################################################

def does_overlap(box1, box2, overlapThresh):
	# Calculate the intersection of two boxes
	x_min = max(box1[0], box2[0])
	y_min = max(box1[1], box2[1])
	x_max = min(box1[2], box2[2])
	y_max = min(box1[3], box2[3])

	# Calculate area of intersection
	intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

	# Calculate area of both boxes
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

	# Check if the intersection is greater than the threshold
	return intersection_area > overlapThresh * min(box1_area, box2_area)

######################################################################################

def place_labels_on_canvas(normalized_img, nms_region_info_list):
	"""
	Places labels on a canvas with random decisions for overlapping areas.

	Parameters:
	normalized_img (numpy.ndarray): The base image used to determine the canvas size.
	nms_region_info_list (list): List of dictionaries containing label info and bounding box.

	Returns:
	numpy.ndarray: The canvas with labels placed on it.
	"""
	canvas_height, canvas_width = normalized_img.shape
	canvas = np.zeros((canvas_height, canvas_width), dtype=np.int16)

	for label_info in nms_region_info_list:
		bbox = label_info['global_bbox']
		label_img = label_info['label_image']

		start_x, start_y, end_x, end_y = bbox
		height, width = label_img.shape[:2]

		temp_canvas = np.zeros_like(canvas)
		temp_canvas[start_y:start_y + height, start_x:start_x + width] = label_img

		overlap_area = (canvas > 0) & (temp_canvas > 0)

		for y in range(start_y, start_y + height):
			for x in range(start_x, start_x + width):
				if overlap_area[y, x]:
					if np.random.rand() < 0.5:  # 50% chance
						temp_canvas[y, x] = canvas[y, x]

		canvas[temp_canvas > 0] = temp_canvas[temp_canvas > 0]

	return canvas
	
######################################################################################
