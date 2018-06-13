#!/usr/bin/env python

### Point cloud
from sensor_msgs.msg import PointField
import sensor_msgs.point_cloud2 as pc2
### Ground truth's bounding box
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
### RGB image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2

class Publisher(object):

    @staticmethod
    def publish_raw_clouds(publisher, header, points):
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        cloud = pc2.create_cloud(header, fields, points)
        publisher.publish(cloud)


    @staticmethod
    def publish_rgb_clouds(publisher, header, points):
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]
        cloud = pc2.create_cloud(header, fields, points)
        publisher.publish(cloud)


    @staticmethod
    def publish_raw_image(publisher, header, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        msg_img = Image()
        try:
            msg_img = CvBridge().cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        msg_img.header = header

        publisher.publish(msg_img)

    @staticmethod
    def publish_bounding_vertex(publisher, header, corners):
        msg_boxes = pc2.create_cloud_xyz32(header, corners)
        publisher.publish(msg_boxes)


    @staticmethod
    def publish_img_bb(publisher, header, corners):
        # one bounding box
        corner = corners[0:8]
        # image bounding box (min_x,min_y,min_z)-->(max_x,max_y,max_z)
        min_x = min(corner[:,0])
        max_x = max(corner[:,0])
        min_y = min(corner[:,1])
        max_y = max(corner[:,1])
        min_z = min(corner[:,2])
        max_z = max(corner[:,2])

        img_bb = np.array([
            [min_x, min_y, min_z],
            [max_x, max_y, max_z]
        ])

        msg_img_bb = pc2.create_cloud_xyz32(header, img_bb)
        publisher.publish(msg_img_bb)


    @staticmethod
    def publish_ground_truth_markers(publisher, header, corners):
        # clear previous bounding boxes to avoid drift bounding boxes
        msg_boxes = MarkerArray()
        marker = Marker()
        marker.header = header
        marker.ns = "kitti_ros"
        marker.action = Marker.DELETEALL
        marker.id = 0
        msg_boxes.markers.append(marker)
        publisher.publish(msg_boxes)

        if corners is None:
            return None

        msg_boxes = MarkerArray()

        num_boxes = len(corners)/8
        marker_id = 0
        for i in range(num_boxes):
            corner = corners[i*8:(i+1)*8]

            marker = Marker()
            marker.header = header
            marker.ns = "kitti_publisher"
            # marker only identify by id
            marker.id = marker_id; marker_id += 1;
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            p = [Point() for _ in range(24)]
            p[0].x =corner[0,0]; p[0].y =corner[0,1]; p[0].z =corner[0,2];
            p[1].x =corner[1,0]; p[1].y =corner[1,1]; p[1].z =corner[1,2];
            p[2].x =corner[1,0]; p[2].y =corner[1,1]; p[2].z =corner[1,2];
            p[3].x =corner[2,0]; p[3].y =corner[2,1]; p[3].z =corner[2,2];
            p[4].x =corner[2,0]; p[4].y =corner[2,1]; p[4].z =corner[2,2];
            p[5].x =corner[3,0]; p[5].y =corner[3,1]; p[5].z =corner[3,2];
            p[6].x =corner[3,0]; p[6].y =corner[3,1]; p[6].z =corner[3,2];
            p[7].x =corner[0,0]; p[7].y =corner[0,1]; p[7].z =corner[0,2];

            p[8].x =corner[4,0]; p[8].y =corner[4,1]; p[8].z =corner[4,2];
            p[9].x =corner[5,0]; p[9].y =corner[5,1]; p[9].z =corner[5,2];
            p[10].x=corner[5,0]; p[10].y=corner[5,1]; p[10].z=corner[5,2];
            p[11].x=corner[6,0]; p[11].y=corner[6,1]; p[11].z=corner[6,2];
            p[12].x=corner[6,0]; p[12].y=corner[6,1]; p[12].z=corner[6,2];
            p[13].x=corner[7,0]; p[13].y=corner[7,1]; p[13].z=corner[7,2];
            p[14].x=corner[7,0]; p[14].y=corner[7,1]; p[14].z=corner[7,2];
            p[15].x=corner[4,0]; p[15].y=corner[4,1]; p[15].z=corner[4,2];

            p[16].x=corner[0,0]; p[16].y=corner[0,1]; p[16].z=corner[0,2];
            p[17].x=corner[4,0]; p[17].y=corner[4,1]; p[17].z=corner[4,2];
            p[18].x=corner[1,0]; p[18].y=corner[1,1]; p[18].z=corner[1,2];
            p[19].x=corner[5,0]; p[19].y=corner[5,1]; p[19].z=corner[5,2];
            p[20].x=corner[2,0]; p[20].y=corner[2,1]; p[20].z=corner[2,2];
            p[21].x=corner[6,0]; p[21].y=corner[6,1]; p[21].z=corner[6,2];
            p[22].x=corner[3,0]; p[22].y=corner[3,1]; p[22].z=corner[3,2];
            p[23].x=corner[7,0]; p[23].y=corner[7,1]; p[23].z=corner[7,2];

            for i in range(24):
                marker.points.append(p[i])

            # box line width
            marker.scale.x = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            # A suitable lifetime to avoid be flash
            # marker.lifetime = rospy.Duration(2);
            msg_boxes.markers.append(marker)

        publisher.publish(msg_boxes)


    @staticmethod
    def publish_clusters(publisher, header, points, corners):
        clusters = []
        num_boxes = len(corners)/8
        init = False
        for i in range(num_boxes):
            corner = corners[i*8:(i+1)*8]
            # print corner

            clusters_idx = np.logical_and(
                (points[:,0]>min(corner[:,0])),
                (points[:,0]<max(corner[:,0])))
            clusters_tmp = points[clusters_idx]
            clusters_idx = np.logical_and(
                (clusters_tmp[:,1]>min(corner[:,1])),
                (clusters_tmp[:,1]<max(corner[:,1])))
            clusters_tmp = clusters_tmp[clusters_idx]
            clusters_idx = np.logical_and(
                (clusters_tmp[:,2]>min(corner[:,2])),
                (clusters_tmp[:,2]<max(corner[:,2])))
            clusters_tmp = clusters_tmp[clusters_idx]
            if not init:
                clusters = clusters_tmp
                init = True
            else:
                # np.append ==> 1D
                clusters = np.append(clusters, clusters_tmp).reshape(-1, 4)
                # print clusters

        msg_clusters = pc2.create_cloud_xyz32(header, clusters[:,:3])
        publisher.publish(msg_clusters)


    @staticmethod
    ##TODO publish ground truth 3D bounding box one by one
    def publish_ground_truth_boxes(publisher, header, places, rotate_zs, sizes):
        msg_ground_truthes = PoseArray()
        msg_ground_truthes.header = header

        if rotate_zs is None:
            # publish an empty one
            publisher.publish(msg_ground_truthes)
            return None
        elif rotate_zs.size == 1:
            x, y, z = places
            h, w, l = sizes

            p = Pose()
            p.position.x = np.float64(x); p.position.y = np.float64(y); p.position.z = np.float64(z);
            p.orientation.x = np.float64(l); p.orientation.y = np.float64(w); p.orientation.z = np.float64(h);
            p.orientation.w = np.float64(rotate_zs)
            msg_ground_truthes.poses.append(p)

        else:
            for place, rotate_z, size in zip(places, rotate_zs, sizes):
                x, y, z = place
                h, w, l = size

                p = Pose()
                p.position.x = np.float64(x); p.position.y = np.float64(y); p.position.z = np.float64(z);
                p.orientation.x = np.float64(l); p.orientation.y = np.float64(w); p.orientation.z = np.float64(h);
                p.orientation.w = np.float64(rotate_z)
                msg_ground_truthes.poses.append(p)

        publisher.publish(msg_ground_truthes)
