#include "highgui.hpp"
#include "core.hpp"
#include "features2d.hpp"
#include "xfeatures2d.hpp"
#include "calib3d.hpp"
#include "opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/*
	参数 Mat作为参数传递时 使用参数如果为 &img 则如果在调用 f（img）时修改Mat的值的话外面

*/

Mat stichingWithSURF(Mat mat1, Mat mat2);
void calCorners(const Mat& H, const Mat& src);//计算变换后的角点
Mat extractFeatureAndMatch(Mat mat1, Mat mat2); //特征提取和匹配
Mat splicImg(Mat& mat1, Mat& mat2, vector<DMatch> goodMatchPoints, vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2);
void optimizeSeam(Mat &mat1, Mat& trans, Mat& dst); //优化拼接

typedef struct
{
	//变换后的图片4个点
	Point2f left_top;  //左上
	Point2f left_bottom;  //左下
	Point2f right_top;    //右上
	Point2f right_bottom;   //右下
}four_corners_t;
four_corners_t corners;

//主函数输出优化拼接图像
void main()
{
	Mat img1, img2, img3;
	img1 = imread("1.jpg");   //读取同级目录下的图片素材
	img2 = imread("2.jpg");
	img3 = imread("3.jpg");
	resize(img1, img1, Size(img1.cols / 4, img1.rows / 4));  //图片宽高为原来的1/4
	resize(img2, img2, Size(img2.cols / 4, img2.rows / 4));
	resize(img3, img3, Size(img3.cols / 4, img3.rows / 4));

	Mat dst = stichingWithSURF(img1, img2);
	//如果图片大于2张，就前两张不断拼接，在进行下一张拼接
	Mat dst2 = stichingWithSURF(dst, img3);
	imwrite("拼好的图像.jpg", dst2);
	imshow("拼好的图像", dst2);
	waitKey();  //程序暂停，等待下一操作
}

Mat stichingWithSURF(Mat mat1, Mat mat2)
{
	/*
		用SURF 是因为 SURF有旋转不变性（比SITF更快）
		1.特征点提取和匹配
		2.图像配准
		3.图像拷贝
		4.图像融合

	*/
	return extractFeatureAndMatch(mat1, mat2);

}

//定位图像变换之后的四个角点
void calCorners(const Mat & H, const Mat & src)
{
	//H为 变换矩阵  src为需要变换的图像

	//计算配准图的角点（齐次坐标系描述）
	double v2[] = { 0,0,1 }; //左上角
	double v1[3];  //变换后的坐标值
	//构成列向量，这种构成方式将向量与Mat关联，Mat修改向量也相应修改
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  // 64F代表每一个像素点元素占64位浮点数，通道数为1
	Mat V1 = Mat(3, 1, CV_64FC1, v1);
	V1 = H * V2; //元素* 
	cout << "0v1:" << v1[0] << endl;
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	//如果在坐标轴中原点在左上角，向下向右延伸
	//左上角（转换为一般的二维坐标系）
	//corners.left_top.x = v1[0] / v1[2];
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角（0，src.rows，1）
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	cout << "1v1:" << v1[0] << endl;
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角（src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);
	V1 = Mat(3, 1, CV_64FC1, v1);
	V1 = H * V2;
	cout << "2v1:" << v1 << endl;
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	cout << "3v1:" << v1 << endl;
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;

	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

	cout << endl;
	cout << "left_top:" << corners.left_top << endl;
	cout << "left_bottom:" << corners.left_bottom << endl;
	cout << "right_top:" << corners.right_top << endl;
	cout << "right_bottom:" << corners.right_bottom << endl;
}


//特征提取和匹配
Mat extractFeatureAndMatch(Mat mat1, Mat mat2)
{
	Mat matg1, matg2;
	//转化成灰度图
	cvtColor(mat1, matg1, CV_RGB2GRAY);
	cvtColor(mat2, matg2, CV_RGB2GRAY);
	//surfDetector参数为门限值，调整这个可以调整检测精度，越大越高，不过相应的速度也会慢
	Ptr<SurfFeatureDetector> surfDetector = SurfFeatureDetector::create(1000.0f);//float
	//Ptr<SURF> surfDetector = SURF::create(1000.0f);
	vector<KeyPoint> keyPoint1, keyPoint2; //特征点
	Mat imgDesc1, imgDesc2; //特征点描述矩阵
	//检测 计算图像的关键点和特征描述矩阵存储到imgDesc1、imgDesc2
	surfDetector->detectAndCompute(matg1, noArray(), keyPoint1, imgDesc1);
	surfDetector->detectAndCompute(matg2, noArray(), keyPoint2, imgDesc2);

	cout << "特征点描述矩阵1大小:" << imgDesc1.cols << " * " << imgDesc1.rows << endl;
	cout << "特征点描述矩阵2大小:" << imgDesc2.cols << " * " << imgDesc2.rows << endl;
	
	FlannBasedMatcher matcher;  //匹配点  近似匹配 可以修改参数改变匹配精度
	vector<vector<DMatch>> matchPoints; //类似二位矩阵
	vector<DMatch> goodMatchPoints; //良好的匹配点

	/*
		DMatch 特征匹配相关结构
		distance  两个特征向量之间的欧氏距离，越小表明匹配度越高(两点距离)。
	*/

	//knn匹配特征点 这里将2作为训练集来训练 对应到后面DMatch的trainIdx
	//1作为测试集实现回归（1匹配2），对应quiryIdx  如此1会作为左边出现 
	vector<Mat> train_disc(1, imgDesc2);
	matcher.add(train_disc);
	matcher.train();
	//用1来匹配该模型（用分类器去分类1），对应到后面DMatch的quiryIdx
	matcher.knnMatch(imgDesc1, matchPoints, 2);//k临近 按顺序排 2表示有两个邻居
	cout << "total match points: " << matchPoints.size() << endl;

	/*
	查找集（Query Set）和训练集（Train Set），
	对于每个Query descriptor，DMatch中保存了和其最好匹配的Train descriptor。
	*/

	//获取优秀匹配点
	for (int i = 0; i < matchPoints.size(); i++)
	{
		if (matchPoints[i][0].distance < 0.4f*matchPoints[i][1].distance) //matchPoints[i][1].distance 欧式距离
		{
			goodMatchPoints.push_back(matchPoints[i][0]);
		}

	}

	Mat firstMatch;
	//这里drawMatches 第一个图片在左边，同时也对应了DMatch的quiryIdx，第二个图片在右边，同时也对应了DMatch的trainIdx
	drawMatches(mat1, keyPoint1, mat2, keyPoint2, goodMatchPoints, firstMatch);
	//在图上显示良好匹配点的连线
	imwrite("特征点匹配图.jpg", firstMatch);
	//imshow("特征点匹配图", firstMatch);

	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i < goodMatchPoints.size(); i++)
	{
		imagePoints1.push_back(keyPoint1[goodMatchPoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[goodMatchPoints[i].trainIdx].pt);
	}

	return splicImg(mat1, mat2, goodMatchPoints, keyPoint1, keyPoint2);

}

//图像拼接——未优化【以图像1为准（1在左半边）】
Mat splicImg(Mat & mat_left, Mat & mat2, vector<DMatch> goodMatchPoints, vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2)
{
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i < goodMatchPoints.size(); i++)
	{
		//这里的queryIdx代表了查询点的目录     
		//trainIdx代表了在匹配时训练分类器所用的点的目录
		imagePoints1.push_back(keyPoint1[goodMatchPoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[goodMatchPoints[i].trainIdx].pt);
	}

	//获取图像2到图像1的投影映射矩阵 3*3
	Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
	cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵  

	calCorners(homo, mat2); //计算配准图的四个顶点坐标
	Mat imgTransform2;

	//图像配准 warpPerspective 对图像进行透视变换 变换后矩阵的宽高都变化
	warpPerspective(mat2, imgTransform2, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), mat2.rows));
	imwrite("直接经过透视矩阵变换得到的图2.jpg", imgTransform2);
	//imshow("直接经过透视矩阵变换得到的img2", imgTransform2);

	//创建拼接后的图
	int distW = imgTransform2.cols; //长宽
	int distH = mat_left.rows;
	Mat dst(distH, distW, CV_8UC3); //3表示通道数
	dst.setTo(0);  //将图像的值全部变为0

	//构成图片  
	//复制img2到dist的右半部分 先复制transform2的图片（因为这个尺寸比较大，后来的图片可以覆盖到他）
	imgTransform2.copyTo(dst(Rect(0, 0, imgTransform2.cols, imgTransform2.rows)));
	mat_left.copyTo(dst(Rect(0, 0, mat_left.cols, mat_left.rows)));
	imwrite("未优化拼接图像.jpg", dst);
	//imshow("拼接（未优化）", dst);

	optimizeSeam(mat_left, imgTransform2, dst);

	return dst;
}



//优化链接处
void optimizeSeam(Mat &mat_left, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_bottom.x, corners.left_top.x);//重叠区域的左边界
	float processW = mat_left.cols - start; //重叠区的宽度
	cout << "开始值:" << start << endl;
	cout << "重叠宽度:" << processW << endl;
	int rows = dst.rows;
	int cols = mat_left.cols;
	float alpha = 1.0f; //mat1 中的像素透明度
	//修改dst中的透明度
	for (int i = 0; i < rows; i++)
	{
		//第i行地址
		uchar *p = mat_left.ptr<uchar>(i);//第i行第一个元素的指针
		uchar *t = trans.ptr<uchar>(i);
		uchar *d = dst.ptr<uchar>(i);


		for (int j = start; j < cols; j++)
		{
			//遇到trans中无像素的黑点，则完全拷贝mat_left中的像素
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				//RGB都为0
				alpha = 1;
			}
			else
			{
				//mat_left中像素的权重与当前处理点距重叠区域左边界的距离成正比
				alpha = (processW - (j - start)) / processW;

			}

			//修改dst中的像素
			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}

	}

}
