# Custom OpenCV

This is my personal implementation of some functions seen during the course of artificial vision. 
Basically, in this code are been re-implemented some of the very basic functions of OpenCV.

The main.cpp file is a simple example that explains how tu use the "library".

## Requirements
- It's required an installation of OpenCV with extra modules and the non free algorithm. So, first dowload both opencv and contrib from GIT, then use this command to build correctly:
```
mkdir opencv_build
cd opencv_build

cmake -DOPENCV_ENABLE_NONFREE:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv/
make -j4
sudo make install -j4
```