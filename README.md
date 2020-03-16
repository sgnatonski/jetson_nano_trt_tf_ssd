# [How to run SSD Mobilenet V2 object detection on Jetson Nano at 20+ FPS ](https://www.dlology.com/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/)| DLology
First, make sure you have flashed the latest JetPack 4.3 on your Jetson Nano development SD card.


This is slightly modified version of https://github.com/Tony607/jetson_nano_trt_tf_ssd implementation with added support for analysing RTSP video streams and detection visualization.

# Build the docker image
```shell
docker build .
```
# Run the docker image
{rtsp_camera_url} = RTSP url of camera video stream

{should_display_detections} = either 0 or 1 (1 for visual representation of detections)

{docker_image_id} = ID of Docker image built
```shell
docker run --runtime nvidia --network host --privileged -it  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e RTSP_CAM={rtsp_camera_url} -e RTSP_CAM_UI={should_display_detections} {docker_image_id}
```
It might be necessary to run
```shell
xhost +
```
before running docker image, if {should_display_detections} parameter is set to 1
