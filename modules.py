
import os
import numpy as np
from tqdm.auto import tqdm
import cv2

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

def compute_iou(boxA, boxB):
	# Determine the coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# Compute the area of intersection
	interArea = max(0, xB - xA) * max(0, yB - yA)

	# Compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	# Compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

######################################################################################

def nms_without_scores(regions, iou_threshold=0.5):
	# Sort the regions by area in descending order
	regions = sorted(regions, key=lambda x: x['area'], reverse=True)

	# List to hold regions that survive NMS
	nms_regions = []

	while regions:
		# Select the region with the largest area and remove from list
		chosen_region = regions.pop(0)
		nms_regions.append(chosen_region)

		# Compare this region with all others, suppress if necessary
		regions = [region for region in regions if compute_iou(chosen_region['bbox'], region['bbox']) < iou_threshold]

		# If there are no regions left to compare and we have not added any regions, break the loop to ensure we return at least one
		if not regions and not nms_regions:
			nms_regions.append(chosen_region)
			break

	# If NMS resulted in no regions due to suppression, add back the largest one
	if not nms_regions:
		nms_regions.append(sorted(regions, key=lambda x: x['area'], reverse=True)[0])

	return nms_regions

######################################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter

def visualize_bboxes(normalized_image, region_info_list):
	display_image = (normalized_image * 255).astype(np.uint8)

	# Create a blank canvas
	canvas = np.zeros(normalized_image.shape[:2], dtype=np.uint16)

	# Draw the bounding boxes and centers on the canvas
	for region_info in region_info_list:
		# Draw the bbox on the canvas
		rr, cc = rectangle_perimeter(start=(region_info['bbox'][0], region_info['bbox'][1]), end=(region_info['bbox'][2]-1, region_info['bbox'][3]-1), shape=canvas.shape)
		canvas[rr, cc] = 255  # Draw white rectangle
		
		# Mark the center on the canvas
		canvas[int(region_info['global_center'][1]), int(region_info['global_center'][0])] = 255

	# Overlay the canvas on the normalized image
	# Create an RGB version of the normalized image if it's not already in that format
	if len(display_image.shape) == 2:
		normalized_image_color = np.dstack((display_image, display_image, display_image))
	else:
		normalized_image_color = display_image.copy()

	# Overlay the bounding boxes in red
	normalized_image_color[canvas == 255] = [255, 0, 0]  # Red color

	# Instead of showing the image, return it as an image object
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.imshow(normalized_image_color)
	ax.set_xticks([])
	ax.set_yticks([])

	# Remove the axes
	ax.axis('off')

	# Return the figure object
	return fig

######################################################################################

from skimage import measure

def generate_bbox_list(window_coords, border_cleaned_predicted_labels, min_area_threshold = 20):

	# Initialize list to store the results
	region_info_list = []

	labels_array = border_cleaned_predicted_labels

	# Analyze each patch
	for i in range(len(labels_array)):
		# Get the properties of the labeled regions in the patch
		regions = measure.regionprops(labels_array[i])
		patch_coord = window_coords[i]  # (x_min, y_min, x_max, y_max)

		# Iterate through each labeled region and extract properties
		for region in regions:
			# Area of the region
			area = region.area
			
			# Check if the area is greater than or equal to the threshold
			if area > min_area_threshold:
				# Local center of mass (centroid) of the region
				local_center = region.centroid  
				# Local bounding box of the region
				local_bbox = region.bbox  
				# Label of the region
				label = region.label

				# Create a mask for the region
				minr, minc, maxr, maxc = local_bbox
				region_mask = labels_array[i][minr:maxr, minc:maxc] == label

				# Apply the mask to the image array to isolate the label
				image_array = region_mask * label

				# Convert local centroid coordinates to global coordinates
				global_center_x = round(local_center[1] + patch_coord[0], 4)  # x-coordinate
				global_center_y = round(local_center[0] + patch_coord[1], 4)  # y-coordinate

				# Convert local bbox coordinates to global coordinates
				global_bbox = (
					local_bbox[0] + patch_coord[1],
					local_bbox[1] + patch_coord[0],
					local_bbox[2] + patch_coord[1],
					local_bbox[3] + patch_coord[0]
				)

				# Add the area, global center, bbox, and image array to the list
				region_info = {
					'area': area,
					'global_center': (global_center_x, global_center_y),
					'bbox': global_bbox,
					'image_array': image_array
				}
				region_info_list.append(region_info)


	return region_info_list

######################################################################################

def reconstruct_patches(region_info_list, original_image):

	# Determine the dimensions of the original image
	original_image_shape = original_image.shape

	# Initialize an empty array for the reconstructed image
	reconstructed_image = np.zeros(original_image_shape, dtype=np.int16)

	# Dictionary to keep track of the regions already placed in the reconstructed image
	region_placements = {}
	current_label = 1  # Initialize the label counter

	# Loop over each region to place it back into the reconstructed image
	for info in region_info_list:
		minr, minc, maxr, maxc = info['bbox']
		binary_image_array = info['image_array']

		# Check if there is an overlap and resolve it
		overlap = False
		for i in range(minr, maxr):
			for j in range(minc, maxc):
				if binary_image_array[i - minr, j - minc] != 0:
					if reconstructed_image[i, j] != 0:
						overlap = True
						break  # No need to check further if we found an overlap
			if overlap:
				break

		# Decide which label to keep if there's an overlap
		if overlap:
			# Overwrite the entire area of the overlap with the new label
			for i in range(minr, maxr):
				for j in range(minc, maxc):
					if binary_image_array[i - minr, j - minc] != 0:
						label = region_placements.get((i, j), current_label)
						reconstructed_image[i, j] = label
						region_placements[(i, j)] = label
		else:
			# If there's no overlap, place the label and increment the label counter
			label = current_label
			current_label += 1
			for i in range(minr, maxr):
				for j in range(minc, maxc):
					if binary_image_array[i - minr, j - minc] != 0:
						reconstructed_image[i, j] = label
						region_placements[(i, j)] = label

	# reconstructed_image_resized = cv2.resize(reconstructed_image, (original_image_shape[1], original_image_shape[0]), interpolation=cv2.INTER_NEAREST)

	return reconstructed_image

######################################################################################
