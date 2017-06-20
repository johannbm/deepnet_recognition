# Intelligent Mirror Face Recognition

This repository contains code for integrating face recognition with the [MagicMirror<sup>2</sup> framework](https://github.com/MichMich/MagicMirror).
It supports running both locally on the RPI or it can read a video feed from an IP camera remotely and send results to the RPI.

## Getting Started

### Prerequisites

This module depends on quite a few python packages. Installing the following modules on a
linux or Mac environment should be straight forward. However installing all packages for
running the system locally on the RPI will take a lot of time and effort. I suggest googling 
how to install each individual package if you aim to run this on the RPI.

- Python 2.7
- macOS or Linux
- [Face Recognition](https://github.com/ageitgey/face_recognition)
- [OpenCV](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)
- [Numpy](http://www.numpy.org/)
- Potentially more ...

### Usage

This software requires interaction between various libraries for the system to work as intended.

- On the RPI:
    - [MagicMirror<sup>2</sup>](https://github.com/MichMich/MagicMirror)
    - this
    - [RPi-Cam-Web-Interface](http://elinux.org/RPi-Cam-Web-Interface)
- On remote machine:
    - This repository
    
### Setup

The goal is to modify the MagicMirror<sup>2</sup> interface based on the recognized user.
For this to work the system needs a coherent list of users both locally on the RPI and on the remote machine.

#### Adding users to the system
1. Create a folder in the */images* directory for each person you want to recognize. The name is not important
    - Put as many pictures as you want of each person. **At least more than 1 is suggested**. The additional recognition
    time when using more images are near negligible. So feel free.
        - I suggest you precede each foldername by the index intended for that person. The directory is read in sorted order.
    - Set **num_faces** in *conf.json* to the number of images per person. Higher value = better accuracy
        - It cannot be lower than the number of images the folder with the least images has.
2. In the MagicMirror<sup>2</sup> configuration file ```config.js``` set the *MMM-Facial-Recognition-2* module's *users*
property to a list of the users name. Index 0 is reserved for unknown users. You can call the users whatever you want as long 
as the indices align with the order the */images* folder is read. The strings entered in the list are used by *MMM-facial-recognition-2*
to swap modules based on current user.

**Example:**
- The */images* directory contains the following folders:
    - 001_Ingunn/   **1**
    - 002_Jarle/    **2**
    - 003_Emil/     **3**
    - 004_Eirik/    **4**
- When read by the user recognition system the folders will be given indexes as shown in bold.
- The *MMM-Facial-Recognition-2* module's *users* property will then be set to the following list:
["stranger", "ingunn", "jarle", "emil", "eirik"]
        

#### Running the System on a Remote Architecture

1. Run [MagicMirror<sup>2</sup>](https://github.com/MichMich/MagicMirror) on the RPI
    - Make sure THIS MagicMirror<sup>2</sup> module is installed
2. Host the camera as a IP camera using [RPi-Cam-Web-Interface](http://elinux.org/RPi-Cam-Web-Interface) software
    - Set **rpi_IP** in *conf.json* to the RPI's IP adress
    - Set **run_on_rpi** in *conf.json* to ```false```
3. Start the recognition by running ```python user_recognition_main.py```
