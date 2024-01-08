# PyBlendPatches - an Image Segmentation and Reconstruction Toolkit

This toolkit is designed to facilitate semantic segmentation of large images by splitting them into smaller patches, performing segmentation on each individual patch, and then reconstructing these segmented patches into a single, large semantic segmentation map.

## Overview

Handling high-resolution images for semantic segmentation can be computationally intensive and may not fit into the memory constraints of many systems. This toolkit addresses this challenge by dividing an image into manageable patches, running segmentation predictions on each patch independently, and stitching them back together to form a cohesive segmentation map.

Figure 1: Original Image
- The original image before processing. It serves as the input for the segmentation toolkit.
![](https://github.com/ajinkya-kulkarni/PyBlendPatches/blob/main/image.png)

Figure 2: Image Decomposition
- The image is decomposed into overlapping patches. This crucial step then enables the processing of smaller, manageable regions of large images.

Figure 3: Segmentation Predictions
- Predicted segmentation masks for each patch with different colors representing distinct blobs identified by the model.

![](https://github.com/ajinkya-kulkarni/PyBlendPatches/blob/main/patches.png)
![](https://github.com/ajinkya-kulkarni/PyBlendPatches/blob/main/predictions.png)

Figure 4: Reconstructed predictions
- The final reconstructed image after segmentation. It shows the aggregated predictions from all patches, seamlessly stitched back together.
![](https://github.com/ajinkya-kulkarni/PyBlendPatches/blob/main/result.png)

## Key Features

- **Patchify**: Decomposes an image into overlapping patches, ready for segmentation.
- **Segmentation Prediction**: Applies a segmentation model to each patch.
- **Border Cleaning**: Adjusts segmentation labels at patch borders to maintain consistency across patch edges.
- **Reconstruction**: Reassembles the segmented patches into a full-size segmentation image.
- **Visualization**: Offers tools to visualize the segmentation process and the end results for quality assurance and analysis.

## Dependencies

The toolkit requires the following Python packages:

- `numpy`: For numerical operations on arrays.
- `matplotlib`: For plotting and visualizing images.
- `tqdm`: For progress bars during operations.
- `scikit-image`: For image processing tasks in some of the toolkit functions.

Install these dependencies using pip:

```bash
pip install numpy matplotlib tqdm scikit-image
```
