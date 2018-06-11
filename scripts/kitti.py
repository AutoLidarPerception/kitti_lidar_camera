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

# self-implemented XML parser
from parse_xml import parseXML

# Load PointCloud data from pcd file
def load_pc_from_pcd(pcd_path):
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

# Load PointCloud data from bin file [X, Y, Z, I]
def load_pc_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

# Detection datasets
def read_label_from_txt(label_path):
    # Read label from txt file
    text = np.fromfile(label_path)
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if (label[0] == "DontCare"):
                continue

            if label[0] == ("Car" or "Van"): #  or "Truck"
                bounding_box.append(label[8:15])

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6]
    else:
        return None, None, None

"""
  Read label of care objects from xml file.

  Returns:
    label_dic (dictionary): labels for one sequence.
        size (list): Bounding Box Size. [h, w, l]
        place (list): Bounding Box Position. [tx, ty, tz]
        rotate (list): Bounding Box Rotation. [rx, ry, rz]
    tracklet_counter: number of label(trajectory) for one sequence
"""
def read_label_from_xml(label_path, care_types):
    labels = parseXML(label_path)
    label_dic = {}
    tracklet_counter = 0
    for label in labels:
        obj_type = label.objectType
        care = False
        for care_type in care_types:
            if obj_type == care_type:
                care = True
                break
        if care:
            tracklet_counter += 1
            first_frame = label.firstFrame
            nframes = label.nFrames
            size = label.size
            for index, place, rotate in zip(range(first_frame, first_frame+nframes), label.trans, label.rots):
                if index in label_dic.keys():
                    # array merged using vertical stack
                    label_dic[index]["place"] = np.vstack((label_dic[index]["place"], place))
                    label_dic[index]["size"] = np.vstack((label_dic[index]["size"], np.array(size)))
                    label_dic[index]["rotate"] = np.vstack((label_dic[index]["rotate"], rotate))
                else:
                    # inited as array
                    label_dic[index] = {}
                    label_dic[index]["place"] = place
                    label_dic[index]["rotate"] = rotate
                    label_dic[index]["size"] = np.array(size)

    return label_dic, tracklet_counter

def read_calib_file(calib_path):
    # Read a calibration file
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def proj_to_velo(calib_data):
    # Projection matrix to 3D axis for 3D Label
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)

# Filter camera angles for KiTTI Datasets
def filter_by_camera_angle(pc):
    bool_in = np.logical_and((pc[:, 1] < pc[:, 0] - 0.27), (-pc[:, 1] < pc[:, 0] - 0.27))
    """
    /*
     * @brief KiTTI Velodyne Coordinate
     *          |x(forward)
     *      C   |   D
     *          |
     *  y---------------
     *          |
     *      B   |   A
     */
    """
    # bool_in = np.where(pc[:, 0] > 0)
    return pc[bool_in]

def create_publish_obj(obj, places, rotates, size):
    """Create object of correct data for publisher"""
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        obj.append((x, y, z))
        h, w, l = sz
        if l > 10:
            continue
        for hei in range(0, int(h*100)):
            for wid in range(0, int(w*100)):
                for le in range(0, int(l*100)):
                    a = (x - l / 2.) + le / 100.
                    b = (y - w / 2.) + wid / 100.
                    c = (z) + hei / 100.
                    obj.append((a, b, c))
    return obj

def get_boxcorners(places, rotates_z, size):
    # Create 8 corners of bounding box from ground center
    if rotates_z.size <= 0:
        return None
    elif rotates_z.size == 1:
        x, y, z = places
        h, w, l = size
        rotate_z = rotates_z
        if l > 10:
            return None

        corner = np.array([
            [x - l / 2., y - w / 2., z],        #
            [x + l / 2., y - w / 2., z],        #
            [x + l / 2., y + w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate_z), -np.sin(rotate_z), 0],
            [np.sin(rotate_z), np.cos(rotate_z), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])

        return np.array(a)
    # rotates_z may be only one dimension
    else:
        corners = []
        for place, rotate_z, sz in zip(places, rotates_z, size):
            x, y, z = place
            h, w, l = sz
            if l > 10:
                continue

            corner = np.array([
                [x - l / 2., y - w / 2., z],        #
                [x + l / 2., y - w / 2., z],        #
                [x + l / 2., y + w / 2., z],
                [x - l / 2., y + w / 2., z],
                [x - l / 2., y - w / 2., z + h],
                [x + l / 2., y - w / 2., z + h],
                [x + l / 2., y + w / 2., z + h],
                [x - l / 2., y + w / 2., z + h],
            ])

            corner -= np.array([x, y, z])

            rotate_matrix = np.array([
                [np.cos(rotate_z), -np.sin(rotate_z), 0],
                [np.sin(rotate_z), np.cos(rotate_z), 0],
                [0, 0, 1]
            ])

            a = np.dot(corner, rotate_matrix.transpose())
            a += np.array([x, y, z])
            corners.append(a)
        # all corners
        return np.array(corners)

# Publish point clouds
def publish_raw_clouds(publisher, header, points):
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)]
    cloud = pc2.create_cloud(header, fields, points)
    publisher.publish(cloud)

def publish_raw_image(publisher, header, img):
    msg_img = Image()
    try:
        msg_img = CvBridge().cv2_to_imgmsg(img, "bgr8")
    except CvBridgeError as e:
        print(e)
        return
    msg_img.header = header

    publisher.publish(msg_img)

# Publish bounding boxes
def publish_bounding_vertex(publisher, header, corners):
    msg_boxes = pc2.create_cloud_xyz32(header, corners)
    publisher.publish(msg_boxes)

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

# Publish bounding boxes
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

def raw_to_voxel(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert PointCloud2 to Voxel"""
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
    pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pc =((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    voxel = np.zeros((int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution))))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    return voxel

def center_to_sphere(places, size, resolution=0.50, min_value=np.array([0., -50., -4.5]), scale=4, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert object label to Training label for objectness loss"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] < z[1]), (places[:, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    return sphere_center

def sphere_to_center(p_sphere, resolution=0.5, scale=4, min_value=np.array([0., -50., -4.5])):
    """from sphere center to label center"""
    center = p_sphere * (resolution*scale) + min_value
    return center

def voxel_to_corner(corner_vox, resolution, center):#TODO
    """Create 3D corner from voxel and the diff to corner"""
    corners = center + corner_vox
    return corners

def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
    # Read labels from xml or txt file.
    if label_type == "txt": #TODO
        """
          Original Label value is shifted about 0.27m from object center.
          So need to revise the position of objects.
        """
        places, size, rotates = read_label_from_txt(label_path)
        if places is None:
            return None, None, None
        rotates = np.pi / 2 - rotates
        dummy = np.zeros_like(places)
        dummy = places.copy()
        if calib_path:
            places = np.dot(dummy, proj_velo.transpose())[:, :3]
        else:
            places = dummy
        if is_velo_cam:
            places[:, 0] += 0.27

    elif label_type == "xml":
        # bounding_boxes[frame index]
        bounding_boxes, frame_counter = read_label_from_xml(label_path)
        #TODO dynamic index according to velodyne points file index
        # need to check boundary
        places = bounding_boxes[107]["place"]
        rotates = bounding_boxes[107]["rotate"][:, 2]
        size = bounding_boxes[107]["size"]

    return places, rotates, size

def create_label(places, size, corners, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=np.array([0., -50., -4.5])):
    """Create training Labels which satisfy the range of experiment"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] + size[:, 0]/2. < z[1]), (places[:, 2] + size[:, 0]/2. >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))

    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2. # Move bottom to center
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)

    train_corners = corners[xyz_logical].copy()
    anchor_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale, min_value=min_value) #sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], anchor_center)):
        train_corners[index] = corner - center
    return sphere_center, train_corners

def corner_to_train(corners, sphere_center, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=np.array([0., -50., -4.5])):
    """Convert corner to Training label for regression loss"""
    x_logical = np.logical_and((corners[:, :, 0] < x[1]), (corners[:, :, 0] >= x[0]))
    y_logical = np.logical_and((corners[:, :, 1] < y[1]), (corners[:, :, 1] >= y[0]))
    z_logical = np.logical_and((corners[:, :, 2] < z[1]), (corners[:, :, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical)).all(axis=1)
    train_corners = corners[xyz_logical].copy()
    sphere_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale, min_value=min_value) #sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], sphere_center)):
        train_corners[index] = corner - center
    return train_corners

def corner_to_voxel(voxel_shape, corners, sphere_center, scale=4):
    """Create final regression label from corner"""
    corner_voxel = np.zeros((voxel_shape[0] / scale, voxel_shape[1] / scale, voxel_shape[2] / scale, 24))
    corner_voxel[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = corners
    return corner_voxel

def create_objectness_label(sphere_center, resolution=0.5, x=90, y=100, z=10, scale=4):
    """Create Objectness label"""
    obj_maps = np.zeros((int(x / (resolution * scale)), int(y / (resolution * scale)), int(round(z / (resolution * scale)))))
    obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = 1
    return obj_maps

def process(velodyne_path, label_path=None, calib_path=None, dataformat="pcd", label_type="txt", is_velo_cam=False):
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None

    if dataformat == "bin":
        pc = load_pc_from_bin(velodyne_path)
    elif dataformat == "pcd":
        pc = load_pc_from_pcd(velodyne_path)

    if calib_path:
        calib = read_calib_file(calib_path)
        proj_velo = proj_to_velo(calib)[:, :3]

    if label_path:
        '''
         read_label_from_xml()
           +size (list): Bounding Box Size. [h, w, l]
           +place (list): Bounding Box bottom center's Position. [tx, ty, tz]
           +rotate (list): Bounding Box bottom center's Rotation. [rx, ry, rz]
        '''
        places, rotates, size = read_labels(label_path, label_type, calib_path=calib_path, is_velo_cam=is_velo_cam, proj_velo=proj_velo)
    # Create 8 corners of bounding box from ground center
    corners = get_boxcorners(places, rotates, size)
    print("# of Point Clouds", len(pc))

    # Camera angle filters
    pc = filter_camera_angle(pc)
    # obj = []
    # obj = create_publish_obj(obj, places, rotates, size)

    p.append((0, 0, 0))
    p.append((0, 0, -1))
    print pc.shape
    print 1
    # publish_pc2(pc, obj)
    a = center_to_sphere(places, size, resolution=0.25)
    print places
    print a
    print sphere_to_center(a, resolution=0.25)
    bbox = sphere_to_center(a, resolution=0.25)
    print corners.shape
    # publish_pc2(pc, bbox.reshape(-1, 3))

    # publish point clouds & publish boxes 8 corners
    # One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions
    # reshape to 3 columns
    publish_pc2(pc, corners.reshape(-1, 3))