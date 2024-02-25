# RGView
RGView is a cross-platform graphical user interface (GUI) application developed for the viewing and analysis of radiographic images, specifically tailored for the radiographs from the Bolton Brush Legacy Growth Study Collection. This collection predominantly features radiographs collected during the 1930s, though it extends until the 1980s.  
The name RGView draws inspiration from the DICOM Modality (a key attribute of the DICOM standard), with "RG" representing Radiographic imaging (conventional film/screen), highlighting the application's focus on facilitating academic and research-oriented examination of radiographic scanned imagery, which could be decayed or otherwise in a poorer quality state compared to digitally acquired radiographs.  
By employing various computational enhancement techniques, RGView serves as a specialized tool for researchers and academics working with this historically significant collection.

![image](https://github.com/frenchfaso/RGV/assets/1913203/7b1cc527-364d-4852-8827-eee61a209d55)

## Overview

RGView integrates a suite of image processing algorithms to support radiographic research and analysis, particularly focusing on the unique needs and challenges associated with the Bolton Brush Legacy Growth Study Collection. It allows for the direct comparison between original and computationally enhanced radiogrpahs images, aiding in the interpretation and study of radiographic data from this collection.

## Features

- **Original radiogrpah Image Viewing**: Enables the examination of radiogrpah images in their unaltered state, providing a baseline for comparison with enhanced outcomes.

- **Image Enhancement Techniques**:
  - **Auto Contrast Adjustment**: Implements automatic contrast adjustment to improve the visibility of radiogrpah image features.
  - **Histogram Equalization**: Applies histogram equalization to redistribute the image's intensity, enhancing overall detail.
  - **Adaptive Histogram Equalization (CLAHE)**: Employs CLAHE for localized contrast enhancement, particularly beneficial for highlighting subtle nuances in radiogrpah images.

- **Real-time Histogram Feedback**: Incorporates a histogram display that offers immediate feedback on the intensity distribution changes resulting from the applied image enhancements, facilitating a deeper understanding of these processes.

- **Automated Image Preprocessing**:
  - **Auto Crop**: Automatically identifies and removes extraneous white borders, focusing on the pertinent radiographic content.
  - **Auto Rotation and Flipping via AI**: Utilizes a neural network model trained specifically for Lateral Cephalometric (Lateral Ceph) radiogrpahs, automating the correction of image orientation through rotation and/or flipping. RGView is designed to be extensible with additional neural network models, and research is underway to develop models for automating other radiographic tasks such as the classification of different types of radiogrpahs, rotation/flipping for other types of radiographs, and automatic segmentation of scans containing more than one radiograph.

- **Batch Enhancement Application**: Allows for the chosen enhancement technique, including automatic enhancements like auto crop and neural network model enhancements, to be applied in batch to all loaded radiographs. This feature streamlines the process of applying consistent enhancements across a collection of images, ensuring uniformity and saving significant time for researchers.

## Usage
### Installation
RGView is available as a standalone executable, streamlining deployment and use without the need for a traditional installation process.  
For Linux, Windows and Mac versions, download directly from the project's [Releases section on GitHub](https://github.com/frenchfaso/RGV/releases).

Windows users might require the Microsoft Visual C++ Redistributable to ensure RGView runs properly. This essential component provides runtime support for the application. It can be downloaded from [Microsoft's official site](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2022).


### Building from Source
To compile RGView from the source code, one should first obtain the repository's contents. This can be achieved by cloning the repository or downloading the ZIP file containing the source code.  
Following this, establishing a Python 3.11 environment is necessary, which can be done using either Conda or venv, as detailed below.

#### Using Conda
To create a Python environment with Conda, execute the following command in your terminal:
```bash
conda create -n rgview python=3.11
conda activate rgview
```
This creates a new Conda environment named rgview and activates it.

#### Using Venv
For creating a Python environment with venv, the commands are as follows:
```bash
python3 -m venv rgview-env
source rgview-env/bin/activate  # On Windows, use `rgview-env\Scripts\activate`
```
This sequence creates and activates a new virtual environment named rgview-env.

#### Installing Dependencies
Once the environment is set up, install the required dependencies with:
```bash
pip install -r requirements.txt
```
#### Building the Executable
Finalize the build process by generating a single-file executable bundle for your platform with PyInstaller:
```bash
pyinstaller RGView.spec
```
This command compiles the application. After this step, you should find a single executable file under the `dist` folder, ready for deployment.
