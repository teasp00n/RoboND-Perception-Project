# 3D Perception

## Filtering

The filtering component of the perception pipeline is concerned with
preprocessing our raw point cloud data to remove noise or superfluous data.
This is important for a couple of reasons; firstly, we make it easier for the
clustering and classification steps to identify features that represent our
objects without having to discern objects from noise. We can also improve
performance of the rest of the pipeline as we reduce the number of points in
our dataset that require processing.

Initially our perception looks like:

[raw]: ./raw.png
![raw][raw]

The first thing we do is remove statistical outliers (noise) from the point
cloud data.

I have implemented it like so:

```python
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
```

And call it in the `pcl_callback`:

```python
mean_k = 10
threshold = 0
cloud = apply_outlier_filter(cloud, mean_k, threshold)
```

My `threshold` value is set to `0` as I'm happy to consider everything beyond one
standard deviation as noise - it is relatively aggressive filtering but seems
to have done a good job tidying things up in my point cloud without removing my
ability to cluster and detect objects

[outlier_filter]: ./outlier_filter.png
![outlier filter][outlier_filter]

Next, I apply voxel grid filtering. If I understand this correctly we are effectively down sampling our point cloud in a similar way that one might scale down an image or video, except we are doing so in 3-space.

My voxel grid filtering is implemented like so:

```python
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
```

And again, called from the `pcl_callback`:

```python
leaf_size = .008
cloud = apply_voxel_grid_filter(cloud, leaf_size)
```

The `leaf_size` sets the new 'resolution' of our point cloud.

[downsampled]: ./downsampled.png
![voxel grid filtering][downsampled]

The final filtering step is applying a couple of passthrough filters to the point cloud. My passthrough filter is implemented as:

```python
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
```

The function is called twice from the `pcl_callback`:

```python
filter_axis = 'z'
axis_min = 0.6
axis_max = 1
cloud = apply_passthrough_filter(cloud, filter_axis, axis_min, axis_max)

filter_axis = 'y'
axis_min = -0.5
axis_max = 0.5
cloud = apply_passthrough_filter(cloud, filter_axis, axis_min, axis_max)
```

The first invocation filters in the `z` (height) axis such that we are only
left with our area of interest in the `z` direction. This was taken directly
from the notes. The second invocation I added to prevent the perception
pipeline from recognising the corners of the table/ dropboxes as items
incorrectly, so we filter in the `y` direction. In a real-world situation we
would want to solve this by modifying our clustering/ classification steps but
this worked well in this case.

It may actually be preferable to perform the pass through filters first as
slicing data off is a constant time operation rather than the other methods of
filtering that have a dependency on `n`.

[passthrough]: ./passthrough.png
![passthrough][passthrough]

## Clustering

The clustering step is concerned with performing euclidean clustering. Euclidean clustering considers each point and if it is within a certain distance of another member of the cluster, is added to the cluster. Remaining points are then considered resulting in our clusters.

My code for the clustering looks like:

```python
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
```

I call the function in the `pcl_callback`:

```python
tolerance = 0.05
min_cluster_size = 90
max_cluster_size = 9999
cluster_indices = apply_euclidean_clustering(white_cloud, tolerance, min_cluster_size, max_cluster_size)
```

Getting good values for `tolerance`, `min_cluster_size`, and `max_cluster_size`
required a little bit of trial and error (and still aren't perfect). The
`tolerance` is the max distance between a member of the cluster and a potential
member. The `min_cluster_size` sets a lower bound on the number of points we
need before we consider it a cluster. Similarly the `max_cluster_size` defines
the upper bound of points allowed in a single cluster.

My clustering isn't ideal for world 3 as I incorrectly cluster the glue and
soap together. I didn't spend longer trying to separate them as the glue is the
last item in the pick list so when it is time to pick it up, we correctly
identify it. (I actually wouldn't identify it until the book in front of it is
removed anyway, more training samples are probably required to fix this).

Now we can color the clusters:

```python
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
```

[clusters]: ./clusters.png
![clusters][clusters]

## Classification

The classification step attempts to classify each of the clusters as an object
but before we do that we have to train the model.

To do this we launch the simple stick training environment and run a modified
version of the `capture_features.py` to capture features for each object at
different angles. I have it set to do `100` iterations per object.

Once we have captured the training data we can use it to train our model with
`train_svm.py` which results in the following confusion matrix.

[training_svm]: ./training_svm.png
![training svm][training_svm]

To help make things a little neater in the `pcl_callback` I have moved the
classification code out into a couple of functions:

```python
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
```

The `recognise_objects` function iterates over the clusters extracting the
desired point cloud and calling `recognise_object` which handles actually
invoking the model. `recognise_objects` also has a side effect of creating and
publishing the labels for rviz.

```python
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
```

The `recognise_object` function builds out the `color` and `normal` histograms
for the given point cloud and concatenates them into a single `feature`
histogram.  This is then handed off to the classifier to get back our predicted
object.

I have included my code for computing the histograms for completeness. I have
kept the default number of bins and range.

```python
def compute_color_histograms(cloud, using_hsv=True):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    r_hist = np.histogram(channel_1_vals, bins=32, range=(0, 256))
    g_hist = np.histogram(channel_2_vals, bins=32, range=(0, 256))
    b_hist = np.histogram(channel_3_vals, bins=32, range=(0, 256))

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((r_hist[0], g_hist[0], b_hist[0])).astype(np.float64)

    # Generate random features for demo mode.
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features)
    return normed_features
```

```python
def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(
            normal_cloud,
            field_names=('normal_x', 'normal_y', 'normal_z'),
            skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute histograms of normal values (just like with color)
    x_hist = np.histogram(norm_x_vals, bins=32, range=(0, 256))
    y_hist = np.histogram(norm_y_vals, bins=32, range=(0, 256))
    z_hist = np.histogram(norm_z_vals, bins=32, range=(0, 256))

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((x_hist[0], y_hist[0], z_hist[0])).astype(np.float64)

    # Generate random features for demo mode.
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features)
    return normed_features
```

Given all the above, our `pcl_callback` code is simplified to:

```python
detected_objects, detected_object_labels = recognise_objects(objects_cloud, cluster_indices, white_cloud)
```

[objects]: ./objects.png
![objects][objects]

## Movement

The `pr2_mover` function is concerned with defining the movement of the pr2 based on the objects we have found and our pick list defined on the rosparam server.

We load the pick list from the server: `object_list_param = rospy.get_param('/object_list')`

Next we iterate through this list and attempt to find rthe corresponding item in the list of detected objects:

```python
for i in range(0, len(object_list_param)):
    object_name = object_list_param[i]['name']
    object_group = object_list_param[i]['group']
    obj = find_object(object_name, object_list)
```

The `find_object` function is a convenience function to iterate over the
detected objects to find the one we are looking for. It returns `None` if it
cannot find it.

```python
def find_object(object_name, object_list):
    print("looking for {}".format(object_name))
    found = None
    for obj in object_list:
        if obj.label == object_name:
            found = obj
            break
    return found
```

If we cannot find it we move on to the next item in the list:

```python
if obj is None:
    print("couldn't find object, moving on")
    continue
```

Assuming we have found our object we can now calculate its centroid:

```python
points_arr = ros_to_pcl(obj.cloud).to_array()
centroid = np.mean(points_arr, axis=0)[:3]
```

We also need to convert the data types back to pythons built in float before appending it to the list:

```python
centroid = (np.asscalar(centroid[0]), np.asscalar(centroid[1]), np.asscalar(centroid[2]))
centroids.append(centroid)
```

Next we use the centroid to create an orientation for the end-effector adjusting for data types as before:

```python
quaternion = tf.transformations.quaternion_from_euler(centroid[0], centroid[1], centroid[2])

pick_pose = Pose()
pick_pose.position.x = centroid[0]
pick_pose.position.y = centroid[1]
pick_pose.position.z = centroid[2]
pick_pose.orientation.x = np.asscalar(quaternion[0])
pick_pose.orientation.y = np.asscalar(quaternion[1])
pick_pose.orientation.z = np.asscalar(quaternion[2])
pick_pose.orientation.w = np.asscalar(quaternion[3])
```

And we inspect the `object_group` to determine what arm to use to grab the item:

```python
arm = "right"
if object_group == "red":
    arm = "left"
```

Knowing what arm we are using we can decide what dropbox makes more sense, and
build out a placement pose.

```python
drop_list_param = rospy.get_param('/dropbox')
place_pose = get_place_pose(arm, drop_list_param)
```

The `get_place_pose` inspects the arm we have chosen and reads the centroid of
the dropbox, finally building out a pose and returns it.

``` python
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
```

Before we send the request to the server we must convert the value types to messages:

```python
world = Int32()
world.data = 3

obj_name = String()
obj_name.data = object_name

which_arm = String()
which_arm.data = arm
```

The `Pose` objects are already ros messages so no conversion is necessary.

The final thing left to do is build out of the yaml for our request and add it our list:

```python
yaml_dict = make_yaml_dict(world, which_arm, obj_name, pick_pose, place_pose)
yaml_list.append(yaml_dict)
```

We can now write our yaml file out to disk:

```python
send_to_yaml("output" + str(world.data) + ".yaml", yaml_list)
```

I have included images of the worlds and associated yaml contents below:

### World 1

[world1]: ./world1.png
![world1][world1]

```yaml
object_list:
- arm_name: right
  object_name: biscuits
  pick_pose:
    orientation:
      w: 0.8866051661688132
      x: 0.28943415047376775
      y: -0.01735793878457402
      z: 0.3603579523521724
    position:
      x: 0.5418375134468079
      y: -0.24172627925872803
      z: 0.7046612501144409
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 1
- arm_name: right
  object_name: soap
  pick_pose:
    orientation:
      w: 0.9078765431532501
      x: 0.25652762503840926
      y: 0.08134658335593455
      z: 0.3214599405222591
    position:
      x: 0.544706404209137
      y: -0.017222251743078232
      z: 0.6757977604866028
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 1
- arm_name: left
  object_name: soap2
  pick_pose:
    orientation:
      w: 0.9222471776426189
      x: 0.17126045856656097
      y: 0.1747088723156448
      z: 0.29934396368724486
    position:
      x: 0.44562023878097534
      y: 0.22152535617351532
      z: 0.678092896938324
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: -0.10354981910140117
      y: 0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 1
```

### World 2

[world2]: ./world2.png
![world2][world2]

```yaml
object_list:
- arm_name: right
  object_name: biscuits
  pick_pose:
    orientation:
      w: 0.8816923847875086
      x: 0.30332915941006866
      y: -0.01397086643366037
      z: 0.36112985829253974
    position:
      x: 0.5717589855194092
      y: -0.24619807302951813
      z: 0.7047865390777588
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 2
- arm_name: right
  object_name: soap
  pick_pose:
    orientation:
      w: 0.9066633168713958
      x: 0.2607632828365981
      y: 0.09319684375488742
      z: 0.3182428137101709
    position:
      x: 0.561161458492279
      y: 0.0030242418870329857
      z: 0.6760101914405823
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 2
- arm_name: left
  object_name: book
  pick_pose:
    orientation:
      w: 0.9020374414522884
      x: 0.2172886131083926
      y: 0.225178761490549
      z: 0.2973359013056161
    position:
      x: 0.5791746377944946
      y: 0.28069543838500977
      z: 0.7209631204605103
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: -0.10354981910140117
      y: 0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 2
- arm_name: left
  object_name: soap2
  pick_pose:
    orientation:
      w: 0.9222098494841366
      x: 0.1704891428881065
      y: 0.177082121285657
      z: 0.2985035477053776
    position:
      x: 0.44583651423454285
      y: 0.2267691045999527
      z: 0.6776955723762512
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: -0.10354981910140117
      y: 0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 2
- arm_name: left
  object_name: glue
  pick_pose:
    orientation:
      w: 0.9009485187680031
      x: 0.27155999738108433
      y: 0.16182273642039785
      z: 0.2972546657826266
    position:
      x: 0.6317411661148071
      y: 0.13051316142082214
      z: 0.6800858378410339
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: -0.10354981910140117
      y: 0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 2
```

### World 3

[world3]: ./world3.png
![world3][world3]

```yaml
object_list:
- arm_name: left
  object_name: sticky_notes
  pick_pose:
    orientation:
      w: 0.9043313664872894
      x: 0.20554850620183435
      y: 0.1185197007172705
      z: 0.3548065271768757
    position:
      x: 0.47455698251724243
      y: 0.06855595111846924
      z: 0.7643585801124573
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: -0.10354981910140117
      y: 0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 3
- arm_name: right
  object_name: snacks
  pick_pose:
    orientation:
      w: 0.8797957016440942
      x: 0.25113939200768914
      y: -0.05953720636238466
      z: 0.3991789701489136
    position:
      x: 0.42699551582336426
      y: -0.31021174788475037
      z: 0.7841095924377441
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 3
- arm_name: right
  object_name: biscuits
  pick_pose:
    orientation:
      w: 0.8806417474362255
      x: 0.29680359974695364
      y: 0.015226459398447828
      z: 0.3689795262394768
    position:
      x: 0.5753613114356995
      y: -0.19341444969177246
      z: 0.7361447215080261
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 3
- arm_name: left
  object_name: eraser
  pick_pose:
    orientation:
      w: 0.9080364440605478
      x: 0.23779195614977486
      y: 0.22353303792241724
      z: 0.2625981393793837
    position:
      x: 0.6093286275863647
      y: 0.28490349650382996
      z: 0.6531654596328735
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: -0.10354981910140117
      y: 0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 3
- arm_name: right
  object_name: soap
  pick_pose:
    orientation:
      w: 0.8880046269360018
      x: 0.30641633864866474
      y: 0.1179375153245556
      z: 0.3219433994174586
    position:
      x: 0.6689753532409668
      y: 0.012160982936620712
      z: 0.6998435854911804
  place_pose:
    orientation:
      w: 0.8950723724076043
      x: 0.10354981910140117
      y: -0.33180792179936425
      z: 0.2793320356634324
    position:
      x: 0
      y: -0.71
      z: 0.605
  test_scene_num: 3
```
