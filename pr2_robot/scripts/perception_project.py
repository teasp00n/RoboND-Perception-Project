#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def apply_voxel_grid_filter(cloud, leaf_size):
    """
    Creates a voxel filter based on the grid size. Basically changes the
    resolution of the point cloud in 3 space similar to down sampling a video
    in 2D.

    :cloud: the point cloud to apply the filter to.
    :leaf_size: the 'resolution' of the new point cloud.
    :returns: the new sampled point cloud.
    """
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return vox.filter()


def apply_passthrough_filter(cloud, axis, axis_min, axis_max):
    """
    Creates a passthrough filter on the given axis for the min and max bounds
    and returns the resultant point cloud.

    :cloud: the point cloud to apply the filter to.
    :axis: the axis the filter is applied to.
    :axis_min: the minimum value allowed on the given axis.
    :axis_max: the maximum value allowed on the given axis.
    :returns: the new point cloud containing only those values in the passthrough filter.
    """
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()


def apply_outlier_filter(cloud, mean_k, threshold):
    """
    Applies a filter to remove all the outliers in the point cloud.

    :cloud: the point cloud to apply the filter to.
    :mean_k: the number of neighboring points to analyze for any given point.
    :threshold: the value to add to the standard deviation to determine if a
    :point is considered an outlier.
    :returns: the new point cloud with outliers removed.
    """
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(mean_k)
    outlier_filter.set_std_dev_mul_thresh(threshold)
    return outlier_filter.filter()


def segment_point_cloud(cloud, max_distance):
    """
    Applies segmentation to the point cloud using the SACMODEL_PLANE model and
    SAC_RANSAC method.

    :cloud: the point cloud to apply the filter to.
    :max_distance: the max distance for a point to be considered fitting the
    model.
    :returns: the pair of inliers and their coefficients
    """
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(max_distance)
    return seg.segment()


def apply_euclidean_clustering(cloud, tolerance, min_cluster_size, max_cluster_size):
    """
    Applies euclidean clustering which is just a fancy way of clustering based on
    proximity.

    :tolerance: distance between a member of this cluster and the potential member.
    :min_cluster_size: minimum members in a cluster.
    :max_cluster_size: maximum members in a cluster.
    :returns: cluster indices for the point cloud.
    """
    tree = cloud.make_kdtree()

    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_cluster_size)
    ec.set_MaxClusterSize(max_cluster_size)
    ec.set_SearchMethod(tree)
    return ec.Extract()


def color_clusters(cloud, cluster_indices):
    """
    Colors the point cloud data in cloud with the color corresponding to the
    cluster index from cluster indices. The colors are provided by the
    get_color_list function provided by one of the helper methods.

    :cloud: the XYZ point cloud to color.
    :cluster_indices: the list of color indices.
    :returns: the color cluster point list which can be used to create a pcl PointCloud2.
    """

    color_cluster_point_list = []
    cluster_color = get_color_list(len(cluster_indices))

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([
                cloud[indice][0], cloud[indice][1],
                cloud[indice][2],
                rgb_to_float(cluster_color[j])
            ])

    return color_cluster_point_list


def recognise_objects(cluster_cloud, cluster_indices, white_cloud):
    """
    Iterates over the clusters attempting to recognise each as an object we
    know about.

    :cluster_cloud: The point cloud containing our clusters.
    :cluster_indices: Indexes for the points mapping them to clusters.
    :white_cloud: The point_cloud with the RGB information removed.
    :returns: The detected objects and their associated labels.
    """

    detected_objects = []
    detected_object_labels = []

    for index, pts_list in enumerate(cluster_indices):
        pcl_cluster = cluster_cloud.extract(pts_list)
        sample_cloud = pcl_to_ros(pcl_cluster)

        detected_object, label = recognise_object(sample_cloud)

        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        detected_object_labels.append(label)
        object_markers_pub.publish(make_label(label, label_pos, index))

        detected_objects.append(detected_object)

    return detected_objects, detected_object_labels


def recognise_object(sample_cloud):
    """
    Attempts to recognise the given sample cloud as an object.

    :sample_cloud: The sample cloud we are trying to recognise.
    :returns: The detected object and label.
    """

    chists = compute_color_histograms(sample_cloud, using_hsv=True)
    normals = get_normals(sample_cloud)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))

    # Make the prediction, retrieve the label for the result
    # and add it to detected_objects_labels list
    prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
    label = encoder.inverse_transform(prediction)[0]

    do = DetectedObject()
    do.label = label
    do.cloud = sample_cloud

    return do, label


def publish_table(cloud):
    """
    Just publishes the given cloud to the table topic.
    """
    ros_cloud_table = pcl_to_ros(cloud)
    table_pub.publish(ros_cloud_table)


def publish_objects(cloud):
    """
    Just publishes the given cloud to the objects topic.
    """
    ros_cloud_objects = pcl_to_ros(cloud)
    objects_pub.publish(ros_cloud_objects)


def publish_clusters(cloud):
    """
    Just publishes the given cloud to the clusters topic.
    """
    ros_cloud_clusters = pcl_to_ros(cloud)
    clusters_pub.publish(ros_cloud_clusters)


def publish_debug(cloud):
    """
    Just publishes the given cloud to the debug topic.
    """
    ros_cloud_debug = pcl_to_ros(cloud)
    debug_pub.publish(ros_cloud_debug)
    print("published to debug")


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    print("pcl_callback called")

    cloud = ros_to_pcl(pcl_msg)

    mean_k = 10
    threshold = 0
    cloud = apply_outlier_filter(cloud, mean_k, threshold)

    leaf_size = .008
    cloud = apply_voxel_grid_filter(cloud, leaf_size)

    filter_axis = 'z'
    axis_min = 0.6
    axis_max = 1
    cloud = apply_passthrough_filter(cloud, filter_axis, axis_min, axis_max)

    filter_axis = 'y'
    axis_min = -0.5
    axis_max = 0.5
    cloud = apply_passthrough_filter(cloud, filter_axis, axis_min, axis_max)

    max_distance = 0.01
    inliers, coefficients = segment_point_cloud(cloud, max_distance)

    table_cloud = cloud.extract(inliers, negative=False)
    publish_table(table_cloud)

    objects_cloud = cloud.extract(inliers, negative=True)
    publish_objects(objects_cloud)

    white_cloud = XYZRGB_to_XYZ(objects_cloud)  # Remove color from the objects cloud

    tolerance = 0.05
    min_cluster_size = 90
    max_cluster_size = 9999
    cluster_indices = apply_euclidean_clustering(white_cloud, tolerance, min_cluster_size, max_cluster_size)

    color_cluster_point_list = color_clusters(white_cloud, cluster_indices)

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    publish_clusters(cluster_cloud)

    detected_objects, detected_object_labels = recognise_objects(objects_cloud, cluster_indices, white_cloud)

    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def find_object(object_name, object_list):
    print("looking for {}".format(object_name))
    found = None
    for obj in object_list:
        if obj.label == object_name:
            found = obj
            break
    return found


def get_place_pose(arm, drop_list_param):
    dest = None
    for dropbox in drop_list_param:
        if dropbox['name'] == arm:
            dest = dropbox
            break
    position = dest['position']
    quaternion = tf.transformations.quaternion_from_euler(position[0], position[1], position[2])
    place_pose = Pose()
    place_pose.position.x = position[0]
    place_pose.position.y = position[1]
    place_pose.position.z = position[2]
    place_pose.orientation.x = np.asscalar(quaternion[0])
    place_pose.orientation.y = np.asscalar(quaternion[1])
    place_pose.orientation.z = np.asscalar(quaternion[2])
    place_pose.orientation.w = np.asscalar(quaternion[3])
    return place_pose


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    yaml_list = []

    labels = []
    centroids = []

    object_name = ""
    object_group = ""

    object_list_param = rospy.get_param('/object_list')

    for i in range(0, len(object_list_param)):
        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        obj = find_object(object_name, object_list)

        # if we can't find the object in the list we go to the next one
        if obj is None:
            print("couldn't find object, moving on")
            continue

        labels.append(obj.label)
        points_arr = ros_to_pcl(obj.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        centroid = (np.asscalar(centroid[0]), np.asscalar(centroid[1]), np.asscalar(centroid[2]))
        centroids.append(centroid)

        quaternion = tf.transformations.quaternion_from_euler(centroid[0], centroid[1], centroid[2])

        pick_pose = Pose()
        pick_pose.position.x = centroid[0]
        pick_pose.position.y = centroid[1]
        pick_pose.position.z = centroid[2]
        pick_pose.orientation.x = np.asscalar(quaternion[0])
        pick_pose.orientation.y = np.asscalar(quaternion[1])
        pick_pose.orientation.z = np.asscalar(quaternion[2])
        pick_pose.orientation.w = np.asscalar(quaternion[3])

        arm = "right"
        if object_group == "red":
            arm = "left"

        drop_list_param = rospy.get_param('/dropbox')
        place_pose = get_place_pose(arm, drop_list_param)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        world = Int32()
        world.data = 3

        obj_name = String()
        obj_name.data = object_name

        which_arm = String()
        which_arm.data = arm

        yaml_dict = make_yaml_dict(world, which_arm, obj_name, pick_pose, place_pose)
        yaml_list.append(yaml_dict)

        # Rotates the robot but not how i would like
        # rotate_msg = Float64()
        # if arm == "left":
            # rotate_msg.data = np.pi / 4
        # else:
            # rotate_msg.data = -np.pi / 4

        # rotate_base_pub.publish(rotate_msg)

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # resp = pick_place_routine(world, obj_name, which_arm, pick_pose, place_pose)

            # print("Response: ", resp.success)
            print("Response: ")

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    send_to_yaml("output" + str(world.data) + ".yaml", yaml_list)


if __name__ == '__main__':

    rospy.init_node('perception_project', anonymous=True)

    pr2_sub = rospy.Subscriber("/pr2/world/points", PointCloud2, pcl_callback, queue_size=1)

    objects_pub = rospy.Publisher("/pr2/world/objects", PointCloud2, queue_size=1)
    table_pub = rospy.Publisher("/pr2/world/table", PointCloud2, queue_size=1)
    clusters_pub = rospy.Publisher("/pr2/world/clusters", PointCloud2, queue_size=1)

    debug_pub = rospy.Publisher("/pr2/world/debug", PointCloud2, queue_size=1)

    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    rotate_base_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=1)

    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
