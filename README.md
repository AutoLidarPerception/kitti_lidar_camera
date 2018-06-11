# lidar_camera
　　LiDAR-Camera fusion based on [kitti_ros](https://github.com/Durant35/kitti_ros).


## TODO list
- [ ] Project Point Cloud into RGB Image.
- [ ] Color Point Cloud with RGB from Image.


## How to use
　We name your ros workspace as `CATKIN_WS` and `git clone` [kitti_ros](https://github.com/Durant35/kitti_ros) as a ros package.
```sh
# clone source code
$ cd $(CATKIN_WS)/src
$ git clone https://github.com/Durant35/kitti_ros

# build your ros workspace
$ cd $(CATKIN_WS)
$ catkin build -DCMAKE_BUILD_TYPE=Release

# launch kitti_ros's kitti_player
$ source devel/setup.bash
$ roslaunch kitti_ros kitti_player.launch
```


## [Parameters](./launch/kitti_player.launch)
+ `keyboard_file`: Keyboard listener is based on Linux input subsystem.
+ `fps`: default `10`Hz, the same as LiDAR frequency.
+ `kitti_data_path`: KiTTI raw data directory, like `.../2011_09_26_drive_0005_sync`
```yaml
.
├── 2011_09_26_drive_0005_sync
│   ├── image_00
│   │   ├── data
│   │   │   ├── 0000000xxx.png
│   │   │   ├── ...
│   │   └── timestamps.txt
│   ├── image_01
│   │   ├── data
│   │   │   ├── 0000000xxx.png
│   │   │   └── ...
│   │   └── timestamps.txt
│   ├── image_02
│   │   ├── data
│   │   │   ├── 0000000xxx.png
│   │   │   └── ...
│   │   └── timestamps.txt
│   ├── image_03
│   │   ├── data
│   │   │   ├── 0000000xxx.png
│   │   │   └── ...
│   │   └── timestamps.txt
│   ├── oxts
│   │   ├── data
│   │   │   ├── 0000000xxx.txt
│   │   │   └── ...
│   │   ├── dataformat.txt
│   │   └── timestamps.txt
│   ├── tracklet_labels.xml
│   └── velodyne_points
│       ├── data
│       │   ├── 0000000xxx.bin
│       │   └── xxx
│       ├── timestamps_end.txt
│       ├── timestamps_start.txt
│       └── timestamps.txt
├── 201?_??_??_drive_0???_sync
│   ├── ...
│   └── ...
├── calib_cam_to_cam.txt
├── calib_imu_to_velo.txt
└── calib_velo_to_cam.txt
```
+ `filter_by_camera_angle`: Only care about Camera's angle of view, default `true`.
+ `care_objects`: default `['Car','Van','Truck','Pedestrian','Sitter','Cyclist','Tram','Misc']`, `[]` means no forground objects.


## Thanks
+ [**utiasSTARS/pykitti**](https://github.com/utiasSTARS/pykitti)


