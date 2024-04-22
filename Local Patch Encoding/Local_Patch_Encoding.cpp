#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common_headers.h>
#include <iostream>//标准C++库中的输入输出类相关头文件
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>//pcd 读写类相关的头文件。
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h> //PCL中支持的点类型头文件
#include<pcl\kdtree\kdtree_flann.h>
#define BOOST_TYPEOF_EMULATION
#include "pclSettings.h"
#include "ThreadPool.h"

//生成 PointRGBNormal形式的点
PointRGBNormal GeneratePoint(float x, float y, float z) {
	PointRGBNormal Result;
	Result.x = x;
	Result.y = y;
	Result.z = z;
	return Result;
}

//归一化
PointRGBNormal UnitizedVector(PointRGBNormal point, float length) {
	// 求模长
	float sum = point.x * point.x + point.y * point.y + point.z * point.z;
	float sqrt_sum = sqrt(sum);

	// 若模长为零，说明点在原点
	if (sqrt_sum == 0) {
		point.x = 0;
		point.y = 0;
		point.z = 0;
	}
	else {
		//按比例放缩
		point.x = point.x * length / sqrt_sum;
		point.y = point.y * length / sqrt_sum;
		point.z = point.z * length / sqrt_sum;
	}
	return point;
}

// 求距离某点距离最近的k个点
vector<int> SearchNearestPoints(PointCloud<PointRGBNormal>::Ptr cloud,
	PointRGBNormal searchPoint,
	int K)
{
	pcl::KdTreeFLANN<PointRGBNormal> kdtree;
	kdtree.setInputCloud(cloud);
	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		/*
		for (size_t i = 0; i < pointIdxNKNSearch.size(); i++)
			std::cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
			<< " " << cloud->points[pointIdxNKNSearch[i]].y
			<< " " << cloud->points[pointIdxNKNSearch[i]].z
			<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
		*/
	}

	return pointIdxNKNSearch;
}

//两点之间距离
float GetTwoPointLength(PointRGBNormal A, PointRGBNormal B)
{
	return sqrt(pow((A.x - B.x), 2) + pow((A.y - B.y), 2) + pow((A.z - B.z), 2));
}

//计算当前点的某一个邻域点到当前点的距离
vector<double> Distance(PointCloud<PointRGBNormal>::Ptr OriginalCloudNormal, PointCloud<PointRGBNormal>::Ptr ProcessCloudNormal, int INDEX, int NN_INDEX) {

	PointRGBNormal PointA = GeneratePoint(OriginalCloudNormal->points[INDEX].x, OriginalCloudNormal->points[INDEX].y, OriginalCloudNormal->points[INDEX].z);
	PointRGBNormal PointANormal = GeneratePoint(OriginalCloudNormal->points[INDEX].normal_x, OriginalCloudNormal->points[INDEX].normal_y, OriginalCloudNormal->points[INDEX].normal_z);
	//PointA = UnitizedVector(PointA, 1);
	PointANormal = UnitizedVector(PointANormal, 1);

	PointRGBNormal PointB = GeneratePoint(OriginalCloudNormal->points[NN_INDEX].x, OriginalCloudNormal->points[NN_INDEX].y, OriginalCloudNormal->points[NN_INDEX].z);
	PointRGBNormal PointBNormal = GeneratePoint(OriginalCloudNormal->points[NN_INDEX].normal_x, OriginalCloudNormal->points[NN_INDEX].normal_y, OriginalCloudNormal->points[NN_INDEX].normal_z);
	//PointB = UnitizedVector(PointB, 1);
	PointBNormal = UnitizedVector(PointBNormal, 1);

	//投影距离
	Vector3d Vec_PointA(PointA.x, PointA.y, PointA.z);
	Vector3d Vec_PointANormal(PointANormal.x, PointANormal.y, PointANormal.z);

	Vector3d Vec_PointB(PointB.x, PointB.y, PointB.z);
	Vector3d Vec_PointBNormal(PointBNormal.x, PointBNormal.y, PointBNormal.z);

	double dis1 = Vec_PointA.dot(Vec_PointANormal);
	double dis2 = Vec_PointB.dot(Vec_PointANormal);
	double dppi = abs(dis2 - dis1);

	if (dppi < 10e-06) {
		dppi = 0;
	}
	double dis3 = Vec_PointB.dot(Vec_PointBNormal);
	double dis4 = Vec_PointA.dot(Vec_PointBNormal);
	double dpip = abs(dis4 - dis3);

	if (dpip < 10e-06) {
		dpip = 0;
	}
	vector<double> res;
	res.push_back(dppi);
	res.push_back(dpip);

	return res;
}

//计算邻域点到该点之间的距离
vector<vector<double>> DistancesInNN(PointCloud<PointRGBNormal>::Ptr OriginalCloud, PointCloud<PointRGBNormal>::Ptr ProcessCloud, int i, vector<int> NNIndexList) {
	vector<vector<double>> ListDistanceInCube(2);

	for (int j = 1; j < 21; j++) {
		vector<double> dis = Distance(OriginalCloud, ProcessCloud, i, NNIndexList[j]);
		ListDistanceInCube[0].push_back(dis[0]);
		ListDistanceInCube[1].push_back(dis[1]);
	}
	return ListDistanceInCube;
}

//特征标签
vector<int> Find_Feature_Index(PointCloud<PointRGBNormal>::Ptr OriginalCloud, PointCloud<PointRGBNormal>::Ptr feature_cloud) {
	vector<int> FeaturePoint_Indexs;
	vector<int> temp;
	for (int i = 0; i < feature_cloud->points.size(); i++) {
		PointRGBNormal SearchPoint = GeneratePoint(feature_cloud->points[i].x, feature_cloud->points[i].y, feature_cloud->points[i].z);
		//在原点云上得到距离特征点最近的点，求其下标并记录
		vector<int> pointIdxNKNSearch = SearchNearestPoints(OriginalCloud, SearchPoint, 1);
		FeaturePoint_Indexs.push_back(pointIdxNKNSearch[0]);
		sort(FeaturePoint_Indexs.begin(), FeaturePoint_Indexs.end());
		auto temp = unique(FeaturePoint_Indexs.begin(), FeaturePoint_Indexs.end());  //”删除”序列中所有相邻的重复元素(只保留一个) 此处的删除，并不是真的删除，而是指重复元素的位置被不重复的元素给占领了
		FeaturePoint_Indexs.erase(temp, FeaturePoint_Indexs.end());
	}
	return FeaturePoint_Indexs;
}

int main(int argc, char *argv[3])
{
	string name = argv[1];
	string full_name = name + ".ply";

	//point cloud path
	string cloud_path = "D:/ABC_DATASET/abc_ply_normal_vertex/";
	//Ground Truth path
	string feature_path = "D:/ABC_DATASET/features/";
	//save_path
	string txt_path = "C:/Users/nwu/Desktop/ABC_dppi/" + name;

	// 创建cloud_Normal点云
	pcl::PointCloud<PointRGBNormal>::Ptr cloud_Normal(new pcl::PointCloud<PointRGBNormal>);
	//定义显示点云
	pcl::PointCloud<PointRGBNormal>::Ptr view_cloud_Normal(new pcl::PointCloud<PointRGBNormal>);

	//创建feature_cloud点云
	pcl::PointCloud<PointRGBNormal>::Ptr feature_cloud(new pcl::PointCloud<PointRGBNormal>);

	if (pcl::io::loadPLYFile(cloud_path + name + ".ply", *cloud_Normal) == -1) // 加载文件
	{
		PCL_ERROR("Couldn't read file test_ply.ply \n");
		system("PAUSE");
		return (-1);
	}

	if (pcl::io::loadPLYFile(feature_path + name + ".ply", *feature_cloud) == -1) // 加载文件
	{
		PCL_ERROR("Couldn't read file test_ply.ply \n");
		system("PAUSE");
		return (-1);
	}
	/*
	for (int i = 0; i < cloud_Normal->size(); i++) {
		cloud_Normal->points[i].r = 255;
		cloud_Normal->points[i].g = 255;
		cloud_Normal->points[i].b = 255;

	}
	*/
	ofstream labelfile, contentfile;
	
	struct stat info;
	if (stat(txt_path.c_str(), &info) != 0) {
		bool flag = CreateDirectory(txt_path.c_str(), NULL);
		//printf("created directory!");
	}
	else {
		//printf("path is exist!");
	}

	string label_path = txt_path + "/Label.txt";
	string content_path = txt_path + "/" + name + ".content";

	labelfile.open(label_path);
	contentfile.open(content_path);

	if (!labelfile) cout << "error" << endl;
	if (!contentfile) cout << "error" << endl;

	vector<int> NNIndexList;
	vector<vector<double>> ListDistanceInKNN;

	//算法运行时间计算
	clock_t start, end;
	start = clock();

	vector<int> FeaturePoint_Indexs = Find_Feature_Index(cloud_Normal, feature_cloud);

	int feature_index = 0;

	//存储点对应的 0、1 值
	for (int i = 0; i < cloud_Normal->size(); i++) {

		//content存入点编号
		contentfile << i << " ";

		// 找该点的K近邻，K=20（不包括自己）
		NNIndexList = SearchNearestPoints(cloud_Normal, cloud_Normal->points[i], 21);

		//计算点K近邻内的点到该点的距离
		ListDistanceInKNN = DistancesInNN(cloud_Normal, view_cloud_Normal, i, NNIndexList);

		for (int j = 0; j < ListDistanceInKNN.size(); j++) {
			for (int k = 0; k < ListDistanceInKNN[1].size(); k++) {
				{
					contentfile << ListDistanceInKNN[j][k] << " ";
				}
			}
		}

		if (i == FeaturePoint_Indexs[feature_index]) {
			feature_index += 1;
			labelfile << "1" << endl;
			contentfile << "1" << endl;
		}
		else {
			labelfile << "0" << endl;
			contentfile << "0" << endl;
		}
	}
	end = clock();//结束时间

	labelfile.close();
	contentfile.close();
	cout << name << endl;
	cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;
}