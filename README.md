# Monocular Visual Odometry
Python3 implementation for 2D-2D monocular visual odometry. [Work in Progress]

## Preparation
Python Packages Required - 
1. OpenCV
2. NumPy

## Usage
```
usage: run.py [-h] -k K -i I [-scale] [--feature_type FEATURE_TYPE]
              [--image_format IMAGE_FORMAT] [--result_file RESULT_FILE]
              [--poses_file POSES_FILE] [--video VIDEO]

Arguments:

  -h, --help                      show this help message and exit

  -k K                            Camera Intrinsic Matrix .txt file (REQUIRED)

  -i I                            Images directory for VO Estimation (REQUIRED)

  -scale                          Enable Scale Consistent Odometry (Need to Provide GT Poses)
                        
  --feature_type FEATURE_TYPE     Feature descriptor format (FAST, ORB, SIFT, SURF)

  --image_format IMAGE_FORMAT     Images format (png, jpg, jpeg, etc)
                        
  --result_file RESULT_FILE       Estimated poses txt file name
                        
  --poses_file POSES_FILE         Ground Truth Poses txt (for scale and visualization)
                        
  --video VIDEO                   Video file to parse for VO images

```
## Results

## References
1. VO Tutorial Part 1 - https://ieeexplore.ieee.org/document/6096039
2. VO Tutorial Part 2 - https://ieeexplore.ieee.org/abstract/document/6153423/references#references
3. D. Nister, O. Naroditsky and J. Bergen, "Visual odometry," Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., Washington, DC, USA, 2004, pp. I-I, doi: 10.1109/CVPR.2004.1315094.
4. Avi Singh's Blog on Visual Odometry - https://avisingh599.github.io/vision/monocular-vo/
5. Fall 2020 - Vision Algorithms for Mobile Robots, ETH Zurich - http://rpg.ifi.uzh.ch/teaching.html