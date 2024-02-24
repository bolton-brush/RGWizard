# RGView
RGView is a cross-platform graphical user interface (GUI) application developed for the viewing and analysis of radiographic images, specifically tailored for the radiographs from the Bolton Brush Legacy Growth Study Collection. This collection predominantly features radiographs collected during the 1930s, although it spans until the 1980s.  
RGView aims to facilitate academic and research-oriented examination of radiographic imagery by employing various computational enhancement techniques. It serves as a specialized tool for researchers and academics working with this historically significant collection.  

![image](https://github.com/frenchfaso/RGV/assets/1913203/7b1cc527-364d-4852-8827-eee61a209d55)

## Overview

RGView integrates a suite of image processing algorithms to support radiographic research and analysis, particularly focusing on the unique needs and challenges associated with the Bolton & Brush Collection. It allows for the direct comparison between original and computationally enhanced radiogrpahs images, aiding in the interpretation and study of radiographic data from this collection.

## Features

- **Original radiogrpah Image Viewing**: Enables the examination of radiogrpah images in their unaltered state, providing a baseline for comparison with enhanced outcomes.

- **Image Enhancement Techniques**:
  - **Auto Contrast Adjustment**: Implements automatic contrast adjustment to improve the visibility of radiogrpah image features.
  - **Histogram Equalization**: Applies histogram equalization to redistribute the image's intensity, enhancing overall detail.
  - **Adaptive Histogram Equalization (CLAHE)**: Employs CLAHE for localized contrast enhancement, particularly beneficial for highlighting subtle nuances in radiogrpah images.

- **Real-time Histogram Feedback**: Incorporates a histogram display that offers immediate feedback on the intensity distribution changes resulting from the applied image enhancements, facilitating a deeper understanding of these processes.

- **Automated Image Preprocessing**:
  - **Auto Crop**: Automatically identifies and removes extraneous white borders, focusing on the pertinent radiographic content.
  - **Auto Rotation and Flipping via AI**: Utilizes a neural network model trained specifically for Lateral Cephalometric (Lateral Ceph) radiogrpahs, automating the correction of image orientation through rotation and/or flipping. RGView is designed to be extensible with additional neural network models, and research is underway to develop models for automating other radiographic tasks such as the classification of different types of radiogrpahs, rotation/flipping for other types of radiogrpahs, and automatic segmentation of scans containing more than one radiogrpah.

## Usage

### Installation
RGView is provided as a standalone executable file, eliminating the need for a traditional installation process. Users can directly download the appropriate version for Linux or Windows from the GitHub Releases section of the project repository. This approach simplifies the deployment and use of RGView, making it immediately accessible for research and analysis purposes.

### Building from sources
Guidelines on building bundles from sources.
