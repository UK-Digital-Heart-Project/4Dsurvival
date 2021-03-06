# Deep learning cardiac motion analysis for human survival prediction (4D*survival*) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1451540.svg)](https://doi.org/10.5281/zenodo.1451540)
![4Dsurvival Network Architecture](data/DAE3.png)

The code in this repository implements 4D*survival*, a deep neural network for carrying out classification/prediction using 3D motion input data. The present implementation was trained using MRI-derived heart motion data and survival outcomes on pulmonary hypertension patients. 

# Overview
The files in this repository are organized into 3 directories:
* [code](survival4D) : contains base functions for fitting the 2 types of statistical models used in our paper: 4D*survival* (supervised denoising autoencoder for survival outcomes) and a penalized Cox Proportional Hazards regression model.
* [demo](demo)
    * [demo/scripts](demo/scripts): contains functions for the statistical analyses carried out in our paper:
        * Training of DL model - [demo/scripts/demo_hypersearchDL.py](demo/scripts/demo_hypersearchDL.py)
        * Generation of Kaplan-Meier plots - [demo/scripts/demo_KMplot.py](demo/scripts/demo_KMplot.py)
        * statistical comparison of model performance - [demo/scripts/demo_modelcomp_pvalue.py](demo/scripts/demo_modelcomp_pvalue.py)
        * Bootstrap internal validation - [demo/scripts/demo_validate.py](demo/scripts/demo_validate.py)
    * [demo/notebooks](demo/notebooks): contains ipython notebooks for demoing the above scripts.
* [data](data) : contains simulated data on which functions from the `demo` directory can be run.

To run the code in the [demo](demo) directory, we provide a [Binder](https://mybinder.org/) interface (for the Jupyter notebooks) and a Docker container (for the corresponding Python scripts). Below are usage instructions:

## 1. Running Jupyter notebooks via Binder or Code Ocean

The Jupyter notebooks in the [demo](demo) directory are hosted on [Binder](https://mybinder.org/), which provides an interactive user interface for executing Jupyter notebooks. Click the link provided below for access:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/UK-Digital-Heart-Project/4Dsurvival/master)

You can also edit code, change parameters, and re-run the analysis interactively in Code Ocean.

[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://doi.org/10.24433/CO.8519672.v1)


## 2. Installation/Usage instructions for Docker Image

A Docker image is available for running the code available in the [demo](demo) directory. This image contains a base Ubuntu Linux operating system image set up with all the libraries required to run the code (e.g. *Tensorflow*, *Keras*, *Optunity*, etc.). The image contains all the code, as well as simulated cardiac data on which the code can be run. 

### Install Docker
Running our 4D*survival* Docker image requires installation of the Docker software, instructions are available at https://docs.docker.com/install/ 

### Download 4D*survival* Docker image (CPU)
Once the Docker software has been installed, our 4D*survival* Docker image can be pulled from the Docker hub using the following command:
    
    docker pull ghalibbello/4dsurvival:latest

Once the image download is complete, open up a command-line terminal. On Windows operating systems, this would be the *Command Prompt* (cmd.exe), accessible by opening the [Run Command utility](https://en.wikipedia.org/wiki/Run_command) using the shortcut key `Win`+`R` and then typing `cmd`. On Mac OS, the terminal is accessible via (Finder > Applications > Utilities > Terminal). On Linux systems, any terminal can be used.
Once a terminal is open, running the following command:

    docker images

should show `ghalibbello/4dsurvival` on the list of Docker images on your local system

### Run 4D*survival* Docker image
    
    docker run -it ghalibbello/4dsurvival:latest /bin/bash

launches an interactive linux shell terminal that gives users access to the image's internal file system. This file system contains all the code in this repository, along with the simulated data on which the code can be run.
Typing 
```
ls -l
```
will list all the folders in the working directory of the Docker image (/4DSurv). You should see the 3 main folders `code`, `data` and `demo`, which contain the same files as the corresponding folders with the same name in this github repository.

Below we will demonstrate how to perform (within the Docker image) the following analyses:
- [x] Train deep learning network
- [x] Train and validate conventional parameter model


### 4D*survival* GPU Docker image
To be able to utilise GPU in docker container, download the GPU docker image:

    docker pull lisurui6/4dsurvival-gpu:1.0
    
Run the GPU docker image 

    nvidia-docker run -ti lisurui6/4dsurvival:1.0

Dockerfile for build the GPU image is described in [docker/gpu/1.0/Dockerfile](docker/gpu/1.0/Dockerfile).

To use the docker image with the latest lifelines, pull from `lisurui6/4dsurvival-gpu:1.1`.

Dockerfile is described in [docker/gpu/1.1/Dockerfile](docker/gpu/1.1/Dockerfile).

#### Train deep learning network
In the docker image, `survival4D` has already installed, so that you can run the following python command anywhere. 
If you are running outside of docker, and want to install the package, from the 4Dsurvival directory, do:

    python setup.py develop

`develop` command allows you to makes changes to the code and do not need to reinstall for the changes to be applied. 


From the 4Dsurvival directory, navigate to the `demo/scripts` directory by typing:
```
cd demo/scripts
ls -l
```
The `demo_hypersearchDL.py` file should be visible. This executes a hyperparameter search (see Methods section in paper) for training of the `4Dsurvival` deep learning network. A demo of this code (which uses simulated input data) can now be run (WARNING: on most machines, this will take several hours to complete):
```
python3 demo_hypersearchDL.py
```

Also under the `demo/scripts` folder, the `demo_validate.py` file should be visible. This executes the bootstrap-based approach for training and internal validation of the Cox Proportional Hazards model for conventional (volumetric) parameters. A demo of this code (which uses simulated input data) can now be run :
```
python3 demo_validate.py
```

By default, all code in `demo/scripts` will use data from [data](data) directory. To point to a different data directory, 
run the above commands with an additional option: `-d`, such as

    python3 demo_hypersearchDL.py -d /path-to-data-dir

You could also specify the data file name with option `-f`, such as

    python3 demo_hypersearchDL.py -d /path-to-data-dir -f data.pkl
    

## Citations
Bello GA, Dawes TJW, Duan J, Biffi C, de Marvao A, Howard LSGE, Gibbs JSR, Wilkins MR, Cook SA, Rueckert D, O'Regan DP. Deep-learning cardiac motion analysis for human survival prediction. *[Nature Machine Intelligence](https://doi.org/10.1038/s42256-019-0019-2)* 1, 
95–104 (2019).

Duan J, Bello G, Schlemper J, Bai W, Dawes TJ, Biffi C, de Marvao A, Doumou G, O’Regan DP, Rueckert D. Automatic 3D bi-ventricular segmentation of cardiac images by a shape-refined multi-task deep learning approach. *[IEEE Transactions on Medical Imaging](https://doi.org/10.1109/TMI.2019.2894322)* 38(9), 2151-2164 (2019).


 
