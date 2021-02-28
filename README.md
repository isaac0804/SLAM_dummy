# SLAM from Dummy
This is a dummy trying to do something related to SLAM (Simultaneous Localization and Mapping).
## Library 
- cv2 for feature detection
## Demo Images
### Harris Corner Algorithm
![Result using Harris Corner](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5b67cd59-b74a-4a87-94d8-16d9eb2ac106/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210228%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210228T134845Z&X-Amz-Expires=86400&X-Amz-Signature=f75ec145159c9740bb047e5ef4c9ec2698207b844081d21f370ca340f04d6869&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
### Shi-Tomasi Corner Algorithm
![Result using Shi-Tomasi Corner](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/487291d2-ca26-455b-84cd-a0b66cfddc45/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210228%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210228T134714Z&X-Amz-Expires=86400&X-Amz-Signature=01d05cbf3703f17bcfb74115204ec20ad151135b4d4430604fb7f0d0d74457fb&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
### ORB (Oriented FAST and Rotated BRIEF)
![Result using ORB](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/459ace02-bb89-476c-881d-25c86cb5d55b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210227%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210227T145608Z&X-Amz-Expires=86400&X-Amz-Signature=3fa5430a9c8bb6eeab9c8929a55d55e9889c47c2bda6fae177f55c44e15d4c48&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
### Feature Matching Frame by Frame
![Result of feature matching](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/dbbf7eca-58a5-4913-bb4b-f76f2f68df21/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210228%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210228T135316Z&X-Amz-Expires=86400&X-Amz-Signature=0058fd8c8df9c08233713aab6d24bdacf5fdaabe3a2bb7f77ed7db644fde026e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
The green circles indicate the features in the current frame, while red circles indicate the features in the last frame. The blue line between the green and red circles is the matching of features in consecutive frames.
## Done
- [x]  Prepare few videos for testing
- [x]  Environment setup 
- [x]  Get familiar with Pycharm
- [x]  Create directory
- [x]  Create github repository
- [x]  Load video using opencv
- [x]  Try implement feature detection (Harris)
- [x]  Try implement feature detection (Shi-Tomasi)
- [x]  Try implement feature detection (ORB)
- [x]  Feature matching

## To Do
- [ ]  Pangolin Library
- [ ]  3d mapping