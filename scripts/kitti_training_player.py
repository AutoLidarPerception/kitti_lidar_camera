#!/usr/bin/env python

import thread
import os
import sys

import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
# ground truth bounding box
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import cv2
import datetime as dt
import time

from kitti import read_label_from_xml
from kitti import load_pc_from_bin
from kitti import get_boxcorners
from kitti import filter_by_camera_angle
from kitti import publish_raw_clouds
from kitti import publish_ground_truth_boxes
from kitti import publish_ground_truth_markers
from kitti import publish_raw_image

'''
    sudo apt-get install python-evdev
'''
from evdev import InputDevice
from select import select

KEY_IDLE=0
KEY_SPACE=57
KEY_LEFT=105
KEY_RIGHT=106
NEXT_FRAME=KEY_RIGHT
LAST_FRAME=KEY_LEFT
KEY_VAL=KEY_IDLE
def on_keyboard(name_dev):
    global KEY_VAL
    dev = InputDevice(name_dev)
    while True:
        select([dev], [], [])
        for event in dev.read():
            if (event.value!=0) and (event.code!=0):
                # KEY_VAL will keep until next pressed
                KEY_VAL = event.code

if __name__ == "__main__":
    # ROS parameters
    keyboard_file = rospy.get_param("/kitti_player/keyboard_file", "/dev/input/event3")
    mode = rospy.get_param("/kitti_player/mode", "observation")
    fps = rospy.get_param("/kitti_player/fps", 10)
    dataset_file = rospy.get_param("/kitti_player/dataset_file", "")
    care_objects = rospy.get_param("/kitti_player/care_objects", "")
    filter_by_camera_angle_ = rospy.get_param("/kitti_player/filter_by_camera_angle", True)

    playing = False
    # open a keyboard listen thread on play mode
    if mode == "play":
        try:
            thread.start_new_thread(on_keyboard, (keyboard_file,))
        except Exception, e:
            print str(e)
            print "Error: unable to start keyboard listen thread."

    rospy.init_node("kitti_training_player")
    # Publisher of Kitti raw data: point cloud & image & ground truth
    pub_points = rospy.Publisher("/kitti/points_raw", PointCloud2, queue_size=1000000)
    pub_img = rospy.Publisher("/kitti/img_raw", Image, queue_size=1000000)
    ground_truth_pub_ = rospy.Publisher("/kitti/bb_raw", PoseArray, queue_size=1000000)

    object_marker_pub_ = rospy.Publisher("/kitti/bb_marker", MarkerArray, queue_size=1000000)

    # Publisher of bounding box corner vertex
    pub_vertex = rospy.Publisher("/kitti/points_corners", PointCloud2, queue_size=1000000)
    # pub_img_bb = rospy.Publisher("/kitti/objects_bb", PointCloud2, queue_size=1000000)
    # Publisher of bounding box
    pub_clusters = rospy.Publisher("/kitti/points_clusters", PointCloud2, queue_size=1000000)

    # Shared header for synchronization
    header_ = std_msgs.msg.Header()
    header_.frame_id = "velodyne"

    fps = rospy.Rate(fps)

    datasets = open(dataset_file, 'r')

    for dataset_path in datasets.readlines():
        # filter out "#" as comment
        dataset_path = dataset_path.strip().lstrip()
        if len(dataset_path) <= 0:
            continue
        if dataset_path[0] != '/':
            continue

        bin_path = dataset_path + "/" + "velodyne_points/data"
        timestamp_file = dataset_path + "/" + "velodyne_points/timestamps.txt"

        xml_path = dataset_path + "/" + "tracklet_labels.xml"
        # img_path = path + "/" + "image_0[0-3]/data/"
        img_path = dataset_path + "/" + "image_02/data"

        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                # t = dt.datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)

        datas = []
        if os.path.isdir(bin_path):
            for lists in os.listdir(bin_path):
                if os.path.isdir(lists):
                    continue
                else:
                    datas.append(lists)
        datas.sort()

        # bounding_boxes[frame index]
        bounding_boxes, tracklet_counter = read_label_from_xml(xml_path, care_objects)

        idx = 0
        # support circular access ...-2,-1,0,1,2...
        while idx < len(datas):
            # CTRL+C exit
            if rospy.is_shutdown():
                print ""
                print "###########"
                print "[INFO] ros node had shutdown..."
                sys.exit(0)

            pc = load_pc_from_bin(bin_path + "/" + datas[idx])
            print "\n[",timestamps[idx],"]","# of Point Clouds:", pc.size

            ##TODO timestamp
            #header_.stamp = rospy.Time.from_sec(timestamps[idx].total_seconds())
            # print (timestamps[idx] - dt.datetime(1970,1,1)).total_seconds()
            header_.stamp = rospy.Time.from_sec((timestamps[idx] - dt.datetime(1970,1,1)).total_seconds())

            img_name = os.path.splitext(datas[idx])[0]+".png"
            img_file = cv2.imread(img_path + "/" + img_name)

            # Camera angle filters
            if filter_by_camera_angle_:
                pc = filter_by_camera_angle(pc)

            corners = None
            if idx in bounding_boxes.keys():
                places = bounding_boxes[idx]["place"]
                # avoid IndexError: too many indices for array
                if bounding_boxes[idx]["rotate"].ndim > 1:
                    rotates_z = bounding_boxes[idx]["rotate"][:, 2]
                else:
                    rotates_z = bounding_boxes[idx]["rotate"][2]
                size = bounding_boxes[idx]["size"]

                # Create 8 corners of bounding box
                corners = get_boxcorners(places, rotates_z, size)

            publish_raw_clouds(pub_points, header_, pc)

            if corners is not None:
                publish_ground_truth_boxes(ground_truth_pub_, header_, places, rotates_z, size)
                publish_ground_truth_markers(object_marker_pub_, header_, corners.reshape(-1, 3))
            else:
                print "no care object in current frame: " + datas[idx]
                # publish empty message
                publish_ground_truth_boxes(ground_truth_pub_, header_, None, None, None)
                publish_ground_truth_markers(object_marker_pub_, header_, None)

            publish_raw_image(pub_img, header_, img_file)
            print "###########"
            print "[INFO] Show image: ",img_name

            fps.sleep()
            # Keyboard control logic
            if playing:
                if KEY_VAL == KEY_SPACE:
                    playing = False
                    KEY_VAL = KEY_IDLE
                else:
                    idx += 1
            while not playing:
                if KEY_VAL == NEXT_FRAME:
                    idx += 1
                    if idx >= len(datas):
                        idx = 0
                    KEY_VAL = KEY_IDLE
                    break
                elif KEY_VAL == LAST_FRAME:
                    idx -= 1
                    if idx < 0:
                        idx = 0
                    KEY_VAL = KEY_IDLE
                    break
                elif KEY_VAL == KEY_SPACE:
                    playing = True
                    idx += 1
                    KEY_VAL = KEY_IDLE
                    break
                else:
                    # CTRL+C exit
                    if rospy.is_shutdown():
                        print ""
                        print "###########"
                        print "[INFO] ros node had shutdown..."
                        sys.exit(-1)

        print "###########"
        print "[INFO] played dataset: ",dataset_path

    print "###########"
    print "[INFO] All datasets played..."