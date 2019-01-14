/*****************************
*文件说明：图像处理相关函数实现*                  
*创建日期：2016-5-2           *
*最近更新：2016-5-20          *
****************************/
#pragma once

#ifndef IMGPROC_H
#define IMGPROC_H

//边缘检测梯度方向
struct direction
{
	int x;
	int y;
	direction() :x(0), y(0){}
};

//虹膜图像大小数据
struct ImgSize
{
	int stride;/*Gdiplus读取的bitmapdata每行必须为4字节整数倍
			     当stride%4=0时，width等于width*/
	int height;
	int width;
	ImgSize() :stride(0), height(0), width(0){}
	void SetStride()
	{
		if (0 == width % 4) stride = width;
		else stride = width + 4 - width % 4;
	}
};

//Hough变换参数空间累加信息
struct Acc
{
	int x;
	int y;
	int r;
	int accmax;//半径r对应的最大累加值
	Acc() :x(0), y(0), r(0), accmax(0){}
};

////2D GaborFilter Kernel
//struct GaborKernel
//{
//	int width;
//	int height;
//	float *real;
//	float *img;
//	GaborKernel(int m, int n) :width(m), height(n)
//	{
//		real = new float[width*height];
//		img = new float[width*height];
//	}
//	~GaborKernel()
//	{
//		if (real)
//		{
//			delete[]real;
//			real = NULL;
//		}
//		if (img)
//		{
//			delete[]img;
//			img = NULL;
//		}
//	}
//};



//图像灰度调整，暂无用处
void adj(const unsigned char *input,unsigned char *output,const ImgSize &imgsize);

/*
	函数说明：基于灰度直方图的快速中值滤波方法
	参数说明
		输入：
			input--输入图像像素矩阵
			r--中值滤波器半径
			imgsize--输入图像大小
		输出：
			output--中值滤波后的像素矩阵
*/
void FastMedianFilter(const unsigned char *input, const int r,unsigned char *output, const ImgSize &imgsize);

/*
	函数说明：中值滤波灰度直方图中值获取，供FastMedianFilter调用
	参数说明
		输入：
			histgram--当前中值滤波窗口灰度脂肪图
			r--中值滤波器半径
		输出：
			（返回值）中值滤波滑动窗口灰度中值
*/
unsigned char Median(const int *histgram,const int r);

/*
	函数说明：基于统计排序的中值滤波方法，窗口大小7*7，暂无用处
	参数说明
		输入：
			input--输入图像像素矩阵
			imgsize--输入图像大小
		输出：
			output--中值滤波后的像素矩阵
*/
void MedianFilter(const unsigned char *input, unsigned char *output, const ImgSize &imgsize);

/*
	函数说明：高斯平滑方法，窗口大小5*5
	参数说明
		输入：
			input--输入图像像素矩阵
			imgsize--输入图像大小
		输出：
			output--高斯平滑后的像素矩阵
*/
void GaussianBlur(const unsigned char *input, unsigned char *output, const ImgSize &imgsize);

/*
	函数说明：基于sobel算子的边缘检测方法，窗口大小3*3
	参数说明
		输入：
			input--输入图像像素矩阵
			vert--垂直方向梯度权重，取值0、1
			horz--水平方向梯度权重，取值0，1
			imgsize--输入图像大小
		输出：
			direc--输入图像梯度
			output--边缘检测后的像素矩阵
*/
void Sobel(const unsigned char *input, unsigned char *output, direction *direc, const ImgSize &imgsize, const int vert, const int horz);

/*
	函数说明：基于canny算子的边缘检测方法，窗口大小3*3，暂无用处
	参数说明
		输入：
			input--输入图像像素矩阵
			vert--垂直方向梯度权重，取值0、1
			horz--水平方向梯度权重，取值0，1
			imgsize--输入图像大小
		输出：
			direc--输入图像梯度
			output--边缘检测后的像素矩阵
*/
void Canny(const unsigned char *input, unsigned char *output, direction *direc, const ImgSize &imgsize, const int vert, const int horz);

/*
	函数说明：对边缘检测图像进行非极大值抑制，细化边缘
	参数说明
		输入：
			input--输入边缘检测图像像素矩阵
			direc--输入边缘检测图像梯度
			imgsize--输入图像大小
		输出：		
			output--非极大值抑制后的像素矩阵
*/
void NonmaxSup(const unsigned char *input, unsigned char *output, const direction *direc, const ImgSize &imgsize);

/*
	函数说明：对非极大值抑制图像进行阈值化
	参数说明
		输入：
			gray--输入非极大值抑制图像像素矩阵
			grad--输入边缘检测图像像素矩阵
			hithres--高阈值
			lowthres--低阈值
			imgsize--输入图像大小
		输出：
			output--阈值化后的像素矩阵
*/
void Hysthresh(const unsigned char *gray, const unsigned char *grad, unsigned char *output, const unsigned char hithres, const unsigned char lowthres, const ImgSize &imgsize);

/*
	函数说明：阈值化边缘点连接
	参数说明
		输入：
			gray--输入非极大值抑制图像像素矩阵
			grad--输入边缘检测图像像素矩阵
			x,y--需连接边缘点的参考点坐标
			lowthres--低阈值
			imgsize--输入图像大小
		输出：
			output--阈值化后的像素矩阵
*/
void TraceEdge(unsigned char *gray, const unsigned char *grad, const int x, const int y, const unsigned char lowthres, const ImgSize &imgsize);

/*
	函数说明：对给定的半径范围，在阈值化图像中寻找圆，供Hysthresh调用
	参数说明
		输入：
			input--阈值化图像像素矩阵
			rmin--最小半径
			rmax--最大半径
			imgsize--输入图像大小
			inner--true(寻找内圆），false（寻找外圆）
		输出：
			x,y，r--寻找到的圆信息
*/
void FindCircle(const unsigned char *input, const int rmin, const int rmax, int &x, int &y, int &r, const ImgSize &imgsize, bool inner);

/*
	函数说明：给定半径，在阈值图像中进行Hough变换
	参数说明
		输入：
			input--阈值化图像像素矩阵
			imgsize--输入图像大小
			inner--true(寻找内圆），false（寻找外圆）
		输出：
			acc--Hough空间最大累加信息
*/
void ThreadCircle(const unsigned char *input, Acc &acc, const ImgSize &imgsize, bool inner);

/*
	函数说明：给定半径，在阈值图像中进行Hough变换，线程启动
	参数说明
		输入：
			input--阈值化图像像素矩阵
			r--Hough变换半径
			imgsize--输入图像大小
			circle--true(寻找内圆），false（寻找外圆）
		输出：
			output--Hough空间累加矩阵
*/
void HoughCircle(const unsigned char *input, int *output, const int r, const ImgSize &imgsize,bool circle);

/*
	函数说明：以给定坐标为圆心、以给定半径及角度范围画扇形（角度0~360时画的即为圆）
	参数说明
		输入：
			x,y--给定圆心
			r--给定半径
			lowangle，higangle--角度范围
			imgsize--输入图像大小
		输出：
			output--给定圆心，半径Hough变换累加矩阵
*/
void AddPie(int *output, const int x, const int y, const int r, const int lowangle, const int highangle, const ImgSize &imgsize);

/*
	函数说明：Hough变换累加控件最大累加信息搜索
	参数说明
		输入：
			input--Hough变换累加矩阵
		输出：
			x,y--最大Hough累加值坐标
			（返回值）给定圆心，半径Hough变换累加矩阵
*/
int HoughSearch(const int *input, int &x, int &y, const ImgSize &imgsize);

/*
	函数说明：给定半径的hough变换像素矩阵图像生成
	参数说明
		输入：
			input--阈值化图像像素矩阵
			radius--Hough变换半径
			imgsize--输入图像大小
			inner--true(寻找内圆），false（寻找外圆）
		输出：
			output--Hough变换后的图像像素矩阵

*/
void HoughMap(const unsigned char *input, unsigned char *output, const int radius, bool inner, const ImgSize &imgsize);

//双线性插值,用于图像归一化变换，图像缩放变换等
void Interpolation(const unsigned char *input, unsigned char *output, const double *x, const double *y,const ImgSize &srcsize, const ImgSize &dstsize);

//Gabor滤波器
void GaborFilter(float *real, float *img, int m, int n, const unsigned char *bmpdata, unsigned char *output, const ImgSize &bmpsize);

#endif