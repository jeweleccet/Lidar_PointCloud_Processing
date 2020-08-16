'''
@article{Zhou2018,
	author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
	title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
	journal   = {arXiv:1801.09847},
	year      = {2018},
}
'''
## IMPORT LIBRARIES
import numpy as np
import time
import open3d
import pandas as pd
import matplotlib.pyplot as plt

## USE http://www.open3d.org/docs/release/tutorial/Basic/

## CHALLENGE 1 - OPEN A FILE OF YOUR CHOICE AND VISUALIZE THE POINT CLOUD
# The supported extension names are: pcd, ply, xyz, xyzrgb, xyzn, pts.
pcd = open3d.io.read_point_cloud("sdc.pcd")

#Visualization
#open3d.visualization.draw_geometries([pcd])

print(f"Points before downsampling: {len(pcd.points)} ")
pcd =pcd.voxel_down_sample(voxel_size=0.2)
print(f"Points after downsampling: {len(pcd.points)}")# DOWNSAMPLING
#open3d.visualization.draw_geometries([pcd])

#Segmentation using RANSAC
t1 = time.time()
plane_model,inliers = pcd.segment_plane(distance_threshold=0.3,ransac_n=3,num_iterations = 150)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers,invert = True)
outlier_cloud.paint_uniform_color([0.5,0.5,0.5])
inlier_cloud.paint_uniform_color([0,1,0])
t2 = time.time()
print(f"Time to segment points using RANSAC {t2-t1}")
#open3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])

## CHALLENGE 4 - CLUSTERING USING DBSCAN
#labels =np.array(outlier_cloud.cluster_dbscan(eps=0.45, min_points=7, print_progress=False))
#max_label= labels.max()
#print(f"point cloud has {max_label + 1} clusters")
#colors = plt.get_cmap("tab20")(labels/(max_label if max_label >0 else 1))
#colors[labels < 0 ]=0
#outlier_cloud.colors=open3d.utility.Vector3dVector(colors[:,:3])
#t3=time.time()
#print(f"Time to cluster outliers using DBSCAN {t3 - t2}")
#open3d.visualization.draw_geometries([outlier_cloud])

## BONUS CHALLENGE - CLUSTERING USING KDTREE AND KNN INSTEAD
pcd_tree = open3d.geometry.KDTreeFlann(outlier_cloud)
print("Paint the 1500th point red.")
outlier_cloud.colors[1500] = [1, 0, 0]
print("Find its 200 nearest neighbors, paint blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(outlier_cloud.points[1500], 200)
np.asarray(outlier_cloud.colors)[idx[1:], :] = [0, 0, 1]
print("Find its neighbors with distance less than 0.2, paint green.")
[k, idx, _] = pcd_tree.search_radius_vector_3d(outlier_cloud.points[1500], 0.2)
np.asarray(outlier_cloud.colors)[idx[1:], :] = [0, 1, 0]
print("Visualize the point cloud.")
open3d.visualization.draw_geometries([outlier_cloud])
## IF YOU HAVE PPTK INSTALLED, VISUALIZE USING PPTK
#import pptk
#v = pptk.viewer(pcd.points)

## CHALLENGE 2 - VOXEL GRID DOWNSAMPLING
