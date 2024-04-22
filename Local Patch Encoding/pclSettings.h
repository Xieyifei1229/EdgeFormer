#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

//filter
#include<pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>  

//io
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/vtk_lib_io.h>

//kdtree
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/flann.h>
#include <pcl/kdtree/kdtree_flann.h>

//surface
#include<pcl/surface/gp3.h>
#include<pcl/surface/mls.h>

//features
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/normal_3d_omp.h>  
#include <pcl/features/fpfh_omp.h>//omp加速计算
#include <pcl/features/rops_estimation.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/integral_image_normal.h>

//registration
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/transformation_estimation_svd.h>  
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_organized_boundary.h>

//other_pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h> //PCL中支持的点类型头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/common/common_headers.h>

//忽略OpenGL警告
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

using namespace std;
using namespace pcl;
using namespace Eigen;

//定义的全局变量
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZRGBNormal PointRGBNormal;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::Normal NormalType;