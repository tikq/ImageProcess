#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include <vector>
#include <cmath>
#include <algorithm>

#include "ImgProc.h"

void adj(const unsigned char *input, unsigned char *output, const ImgSize &imgsize)
{
	memset(output, 0, imgsize.stride*imgsize.height);
	unsigned char temp = 0;
	for (int y = 1; y < imgsize.height - 1; ++y)
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			if (input[y*imgsize.stride + x] < temp) temp = input[y*imgsize.stride + x];
		}
	for (int y = 1; y < imgsize.height - 1; ++y)
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			output[y*imgsize.stride + x] = input[y*imgsize.stride + x] - temp;
		}
	temp = 0;
	for (int y = 1; y < imgsize.height - 1; ++y)
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			if (output[y*imgsize.stride + x] > temp) temp = output[y*imgsize.stride + x];
		}
	double ratio = (double)temp / 255;
	for (int y = 1; y < imgsize.height - 1; ++y)
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			output[y*imgsize.stride + x] =(unsigned char)(output[y*imgsize.stride + x] / ratio);
		}
	

}

void FastMedianFilter(const unsigned char *input,const int r, unsigned char *output, const ImgSize &imgsize)
{
	memcpy(output, input, imgsize.stride*imgsize.height);

	int histgram[256] = { 0 };

	for (int y = r; y < imgsize.height - r; ++y)
	{
		//灰度直方图初始化
		memset(histgram, 0, 256 * sizeof(int));

		for (int radius = -r; radius <= r; ++radius)
		{
			for (int k = 0; k < 2*r+1; ++k)
			{
				histgram[input[(y + radius)*imgsize.stride + k]] += 1;
			}
		}
		output[y*imgsize.stride + r] = Median(histgram,r);

		//移动滤波窗口，变更灰度直方图，计算灰度直方图中值
		for (int x = r; x < imgsize.width - r-1; ++x)
		{
			for (int radius = -r; radius <= r; ++radius)
			{
				histgram[input[(y + radius)*imgsize.stride + x - 3]] -= 1;
				histgram[input[(y + radius)*imgsize.stride + x + 4]] += 1;
			}
			output[y*imgsize.stride + x + 1] = Median(histgram,r);
		}
	}
		
}

unsigned char Median(const int *histgram,const int r)
{
	int sum = 0;
	int count = (2 * r + 1)*(2 * r + 1) / 2 + 1;
	for (int i = 0; i < 256; ++i)
	{
		sum += histgram[i];
		if (sum >=count ) return (unsigned char)i;
	}
	return 0;
}

void MedianFilter(const unsigned char *input, unsigned char *output, const ImgSize &imgsize)
{
	memcpy(output, input, imgsize.stride*imgsize.height);

	unsigned char data[49] = { 0 };

	int pos = 0;

	for (int y = 3; y < imgsize.height - 3; ++y)
	{
		for (int x = 3; x < imgsize.width - 3; ++x)
		{
			//遍历滤波窗口灰度值
			for (int j = y - 3; j <= y + 3; j++)
				for (int i = x - 3; i <= x + 3; i++)
					data[pos++] = input[j*imgsize.stride + i];

			//排序滤波窗口灰度值，获取中值
			std::stable_sort(data, data + 49);
			output[y*imgsize.stride + x] = data[24];
			pos = 0;
		}
	}
}

void GaussianBlur(const unsigned char *input, unsigned char *output, const ImgSize &imgsize)
{
	memcpy(output, input, imgsize.stride*imgsize.height);
	//高斯平滑卷积模板，273
	int  gaussian[25] = {
		1, 4, 7, 4, 1,
		4, 16, 26, 16, 4,
		7, 26, 41, 26, 7,
		4, 16, 26, 16, 4,
		1, 4, 7, 4, 1,
	};
	/*159
	int gaussian[25] = {
	2, 4, 5, 4, 2,
	4, 9, 12, 9, 4,
	5, 12, 15, 12, 5,
	4, 9, 12, 9, 4,
	2, 4, 5, 4, 2,
	};*/

	double max = 0.0;

	//滑动高斯窗口，卷积
	for (int y = 2; y <= imgsize.height - 2; ++y)
		for (int x = 2; x <= imgsize.width - 2; ++x)
		{
			for (int j = 0; j < 5; ++j)
				for (int i = 0; i < 5; ++i)
					max += gaussian[j * 5 + i] * input[(y + j - 2)*imgsize.stride + x + i - 2];

			max = max / 273;
			output[y*imgsize.stride + x] = (unsigned char)max;

		}
}

void Sobel(const unsigned char *input, unsigned char *output, direction *direc, const ImgSize &imgsize,const int vert,const int horz)
{
	memset(output, 0, imgsize.stride*imgsize.height);

	//sobel算子卷积模板
	int sobel[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	double *sum =new double[imgsize.stride*imgsize.height];
	double max = 0;

	for (int y = 1; y < imgsize.height - 1; ++y)
	{
		for (int x = 1; x < imgsize.width-1; ++x)
		{		
			//direc[y*imgsize.stride + x].x = input[(y - 1)*imgsize.stride + x + 1] - input[(y - 1)*imgsize.stride + x - 1] + input[(y + 1)*imgsize.stride + x + 1] - input[(y + 1)*imgsize.stride + x - 1] + 2 * (input[y*imgsize.stride + x + 1] - input[y*imgsize.stride + x - 1]);
			//direc[y*imgsize.stride + x].y = input[(y + 1)*imgsize.stride + x - 1] - input[(y - 1)*imgsize.stride + x - 1] + input[(y + 1)*imgsize.stride + x + 1] - input[(y - 1)*imgsize.stride + x + 1] + 2 * (input[(y + 1)*imgsize.stride + x] - input[(y - 1)*imgsize.stride + x]);
		
			//direc[y*imgsize.stride + x].x = 0;
			//direc[y*imgsize.stride + x].y = 0;

			//水平、垂直、总梯度计算
			for (int j = 0; j <3; ++j)
				for (int i = 0; i < 3; ++i)
				{
					direc[y*imgsize.stride + x].x += sobel[j * 3 + i] * input[(y + j - 1)*imgsize.stride + x + i - 1];
					direc[y*imgsize.stride + x].y += sobel[ i* 3 + j] * input[(y + j - 1)*imgsize.stride + x + i - 1];
				}
			direc[y*imgsize.stride + x].x = direc[y*imgsize.stride + x].x*vert;
			direc[y*imgsize.stride + x].y = direc[y*imgsize.stride + x].y*horz;

			sum[y*imgsize.stride + x] = direc[y*imgsize.stride + x].x*direc[y*imgsize.stride + x].x + direc[y*imgsize.stride + x].y*direc[y*imgsize.stride + x].y;
			sum[y*imgsize.stride + x] = sqrt(sum[y*imgsize.stride + x]);
			//output[y*imgsize.stride + x] = sum[y*imgsize.stride + x];
			if (sum[y*imgsize.stride + x]>max)
			max = sum[y*imgsize.stride + x];
		}
	}
	
	//梯度归一化（0~255）
	double ratio = max / 255;
	for (int y = 1; y < imgsize.height - 1; ++y)
	{
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			output[y*imgsize.stride + x] = (unsigned char)(sum[y*imgsize.stride + x] / ratio);
		}
	}
	
	delete[]sum;
	sum = NULL;
}

void Canny(const unsigned char *input, unsigned char *output, direction *direc, const ImgSize &imgsize, const int vert, const int horz)
{
	memset(output, 0, imgsize.stride*imgsize.height);

	double *sum = new double[imgsize.stride*imgsize.height];
	double max = 0;

	//水平、垂直、总梯度计算
	for (int y = 1; y < imgsize.height - 1; ++y)
	{
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			//direc[y*imgsize.stride + x].x = (input[y *imgsize.stride + x + 1] - input[y *imgsize.stride + x ] + input[(y + 1)*imgsize.stride + x + 1] - input[(y + 1)*imgsize.stride + x ])/2 ;
			//direc[y*imgsize.stride + x].y = (input[y *imgsize.stride + x ] - input[(y + 1)*imgsize.stride + x ] + input[y *imgsize.stride + x + 1] - input[(y + 1)*imgsize.stride + x + 1])/2 ;
			int h= input[y *imgsize.stride + x + 1] - input[y *imgsize.stride + x-1 ] ;
			int v= input[(y+1)*imgsize.stride + x ] - input[(y -1)*imgsize.stride + x ] ;
			int d1 = input[(y + 1)*imgsize.stride + x + 1] - input[(y - 1)*imgsize.stride + x - 1];
			int d2 = input[(y + 1)*imgsize.stride + x - 1] - input[(y - 1)*imgsize.stride + x - 1];

			direc[y*imgsize.stride + x].x = (h+(d1+d2)/2)*vert;
			direc[y*imgsize.stride + x].y = (v+(d1-d2)/2)*horz;

			sum[y*imgsize.stride + x] = direc[y*imgsize.stride + x].x*direc[y*imgsize.stride + x].x + direc[y*imgsize.stride + x].y*direc[y*imgsize.stride + x].y;
			sum[y*imgsize.stride + x] = sqrt(sum[y*imgsize.stride + x])+0.5;
			//output[y*imgsize.stride + x] = (unsigned char)sum[y*imgsize.stride + x] ;
			if (sum[y*imgsize.stride + x]>max)
				max = sum[y*imgsize.stride + x];
		}
	}
	
	//梯度归一化（0~255）
	double ratio = max / 255;
	for (int y = 1; y < imgsize.height - 1; ++y)
	{
		for (int x = 1; x < imgsize.width - 1; ++x)
		{
			output[y*imgsize.stride + x] = (unsigned char)(sum[y*imgsize.stride + x] / ratio);
		}
	}
	
	delete[]sum;
	sum = NULL;
}

void NonmaxSup(const unsigned char *input, unsigned char *output, const direction *direc,const ImgSize &imgsize)
{
	memset(output, 0, imgsize.stride*imgsize.height);

	unsigned char g1, g2, g3, g4;

	//梯度方向相邻边缘点梯度，权重
	double gtmp1, gtmp2, weight;

	for (int y = 2; y < imgsize.height - 2; ++y)
	{
		for (int x = 2; x < imgsize.width - 2; ++x)
		{
			if (abs(direc[y*imgsize.stride + x].y) > abs(direc[y*imgsize.stride + x].x))
			{
				g2 = input[(y - 1)*imgsize.stride + x];
				g4 = input[(y + 1)*imgsize.stride + x];
				if (0 != abs(direc[y*imgsize.stride + x].y))
					weight = abs(direc[y*imgsize.stride + x].x) / abs(direc[y*imgsize.stride + x].y);
				else weight = 0;
				if (direc[y*imgsize.stride + x].y*direc[y*imgsize.stride + x].x > 0)
				{
					g1 = input[(y - 1)*imgsize.stride + x - 1];
					g3 = input[(y + 1)*imgsize.stride + x + 1];
				}
				else
				{
					g1 = input[(y - 1)*imgsize.stride + x + 1];
					g3 = input[(y + 1)*imgsize.stride + x - 1];
				}
			}
			else
			{
				g2 = input[y*imgsize.stride + x + 1];
				g4 = input[y*imgsize.stride + x - 1];
				if (0 != abs(direc[y*imgsize.stride + x].x))
					weight = abs(direc[y*imgsize.stride + x].y) / abs(direc[y*imgsize.stride + x].x);
				else
					weight = 0;
				if (direc[y*imgsize.stride + x].y*direc[y*imgsize.stride + x].x > 0)
				{
					g1 = input[(y + 1)*imgsize.stride + x + 1];
					g3 = input[(y - 1)*imgsize.stride + x - 1];
				}
				else
				{
					g1 = input[(y - 1)*imgsize.stride + x + 1];
					g3 = input[(y + 1)*imgsize.stride + x - 1];
				}
			}
			gtmp1 = weight*g1 + (1 - weight)*g2;
			gtmp2 = weight*g3 + (1 - weight)*g4;

			//如果当前边缘点梯度值大于梯度方向边缘点梯度，保留此边缘点
			if (input[y*imgsize.stride + x] > gtmp1&&input[y*imgsize.stride + x] > gtmp2)
				output[y*imgsize.stride + x] = 128;
			else
				output[y*imgsize.stride + x] = 0;
		}
	}
}

void Hysthresh(const unsigned char *gray,const unsigned char *grad, unsigned char *output,const unsigned char hithres, const unsigned char lowthres, const ImgSize &imgsize)
{
	memcpy(output, gray, imgsize.stride*imgsize.height);

	for (int y = 0; y < imgsize.height ; ++y)
		for (int x = 0; x < imgsize.width ; ++x)
		{
			if (output[y*imgsize.stride + x] == 128 && grad[y*imgsize.stride + x] >= hithres)
			{
				output[y*imgsize.stride + x] = 255;

				//连接当前边缘点8邻域内的边缘点
				TraceEdge(output, grad,x,y, lowthres, imgsize);
			}
		}

	for (int y = 0; y < imgsize.height ; ++y)
		for (int x = 0; x < imgsize.width ; ++x)
		{
			if (output[y*imgsize.stride + x] != 255)
				output[y*imgsize.stride + x] = 0;
		}
}

void TraceEdge(unsigned char *gray, const unsigned char *grad,const int x,const int y, const unsigned char lowthres, const ImgSize &imgsize)
{
	for (int j = y - 1; j <= y + 1; ++j)
		for (int i = x - 1; i <= x + 1; ++i)
		{
			if (i >= 0 && i<imgsize.width&&j >= 0 && j<imgsize.height&&i != x&&j != y)
				if (gray[j*imgsize.stride + i] != 255)
				{
					if (grad[j*imgsize.width + i] >= lowthres&& gray[j*imgsize.stride + i] == 128)
					{
						gray[j*imgsize.stride + i] = 255;
						TraceEdge(gray, grad,i,j,lowthres, imgsize);
					}
					else
						gray[j*imgsize.stride + i] = 0;
				}
		}
}

void FindCircle(const unsigned char *input, const int rmin, const int rmax, int &x, int &y, int &r, const ImgSize &imgsize, bool inner)
{
	//Hough累加空间
	Acc *acc = new Acc[rmax - rmin + 1];

	int max = 0;

	//HoughCircle线程向量
	std::vector<std::thread>threads;

	//HoughCircle线程控制
	int threadnum = 10;
	std::mutex thread_mutex;
	std::condition_variable condition;
	//对半径范围内的每个半径启动一个HougCircle线程进行计算
	for (int pr = rmin; pr <= rmax; ++pr)
	{
		std::unique_lock<std::mutex> lock(thread_mutex);
		if (threadnum <= 0)
		{
			condition.wait(lock);
		}
		--threadnum;
		lock.unlock();

		acc[pr - rmin].r = pr;
		threads.push_back(std::thread(ThreadCircle, input, std::ref(acc[pr - rmin]), std::ref(imgsize), inner));

		
		lock.lock();
		++threadnum;
		condition.notify_one();
		lock.unlock();
	}
	for (auto &t : threads)
	{
		t.join();
	}

	//遍历累加空间，获取最大累加信息，也即寻找到的圆的信息
	for (int i = 0; i < rmax - rmin + 1; ++i)
	{
		if (acc[i].accmax > max)
		{
			max = acc[i].accmax;
			x = acc[i].x;
			y = acc[i].y;
			r = acc[i].r;
		}
	}

	delete[]acc;
	acc = NULL;
}

void ThreadCircle(const unsigned char *input, Acc &acc, const ImgSize &imgsize, bool inner)
{
	int *houghspace = new int[imgsize.stride*imgsize.height];
	memset(houghspace, 0, imgsize.stride*imgsize.height*sizeof(int));
	
	HoughCircle(input, houghspace, acc.r, imgsize, inner);
	acc.accmax = HoughSearch(houghspace, acc.x, acc.y, imgsize);

	delete[]houghspace;
	houghspace = NULL;
}

void HoughCircle(const unsigned char *input, int *output, const int r, const ImgSize &imgsize,bool circle)
{
	for (int y = 0; y < imgsize.height; ++y)
	{
		for (int x = 0; x < imgsize.width; ++x)
		{
			if (input[y*imgsize.stride + x] == 255)
			{
				if (circle)  
					AddPie(output, x, y, r, 0, 360, imgsize);
				else
				{
					AddPie(output, x, y, r, 0, 60, imgsize);
					AddPie(output, x, y, r, 120, 240, imgsize);
					AddPie(output, x, y, r, 300, 360, imgsize);
				}		
			}
		}
	}
}

void AddPie(int *output, const int x, const int y, const int r, const int lowangle, const int highangle, const ImgSize &imgsize)
{
	double t = 0.0;
	int x0 = 0;
	int y0 = 0;
	for (int theta = lowangle; theta < highangle; ++theta)
	{
		t = (theta * 3.14159265358) / 180;
		x0 = (int)round(x + r * cos(t));
		y0 = (int)round(y + r * sin(t));

		if (x0>0 && x0 < imgsize.width&&y0>0 && y0 < imgsize.height)
		{
			output[y0*imgsize.stride + x0] += 1;
		}
	}
}

int HoughSearch(const int *input, int &x, int &y, const ImgSize &imgsize)
{
	int max = 0;
	for (int y0 = 0; y0 < imgsize.height; ++y0)
		for (int x0 = 0; x0 < imgsize.width; ++x0)
		{
			if (input[y0*imgsize.stride + x0] > max)
			{
				max = input[y0*imgsize.stride + x0];
				x = x0;
				y = y0;
			}
		}
	return max;
}

void HoughMap(const unsigned char *input, unsigned char *output, const int radius, bool inner, const ImgSize &imgsize)
{
	int *houghspace = new int[imgsize.stride*imgsize.height];

	//初始化houghspace，并计算输入半径Hough变换
	memset(houghspace, 0, imgsize.stride*imgsize.height*sizeof(int));
	HoughCircle(input, houghspace, radius, imgsize,inner);

	int maxacc = 0;
	for (int j = 0; j < imgsize.height; ++j)
		for (int i = 0; i < imgsize.width; ++i)
		{
			if (houghspace[j*imgsize.stride + i]>maxacc)
				maxacc = houghspace[j*imgsize.stride + i];
		}

	//归一化Hough累加空间，得到Hough变换图像像素矩阵
	double ratio = maxacc / 255.0;
	memset(output, 0, imgsize.stride*imgsize.height);

	for (int j = 0; j < imgsize.height; ++j)
		for (int i = 0; i < imgsize.width; ++i)
		{
			output[j*imgsize.stride + i] = (unsigned char)(houghspace[j*imgsize.stride + i] / ratio);
		}

	delete[]houghspace;
	houghspace = NULL;
}

void Interpolation(const unsigned char *input, unsigned char *output, const double *x, const double *y, const ImgSize &srcsize,const ImgSize &dstsize )
{
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	double f1 = 0.0, f2 = 0.0, fmax = 0.0, x0 = 0.0, y0 = 0.0;
	double *f = new double[dstsize.stride*dstsize.height];

	for (int radius = 0; radius < dstsize.height; ++radius)
	{
		for (int angle = 0; angle < dstsize.width; ++angle)
		{
			x0 = x[radius*dstsize.width + angle];
			y0 = y[radius*dstsize.width + angle];
			
			x2 = (int)ceil(x0);
			x1 = x2-1;		
			y2 = (int)ceil(y0);
			y1 = y2-1;

			if (x1 != x2&&y1 != y2)
			{
				f1 = input[y1*srcsize.stride + x1] * ((x2 - x0) / (x2 - x1)) + input[y1*srcsize.stride + x2] * ((x0 - x1) / (x2 - x1));
				f2 = input[y2*srcsize.stride + x1] * ((x2 - x0) / (x2 - x1)) + input[y2*srcsize.stride + x2] * ((x0 - x1) / (x2 - x1));
				f[radius*dstsize.stride + angle] = f1*((y2 - y0) / (y2 - y1)) + f2*((y0 - y1) / (y2 - y1));
			}
			else
				f[radius*dstsize.stride + angle] = input[y2*srcsize.stride + x2];
			
			
			if (fmax < f[radius*dstsize.stride + angle]) fmax = f[radius*dstsize.stride + angle];
		}
	}
	//灰度值归一化
	double ratio = fmax / 255;
	for (int radius = 0; radius < dstsize.height; ++radius)
	{
		for (int angle = 0; angle < dstsize.width; ++angle)
		{
			output[radius*dstsize.stride + angle] =(unsigned char) (f[radius*dstsize.stride + angle] / ratio);
		}
	}	
	delete[]f;
	f = NULL;
}

void GaborFilter(float *real,float *img,int m,int n , const unsigned char *bmpdata,unsigned char *output, const ImgSize &bmpsize)
{
	int xmin = m / 2;
	int xmax =bmpsize.width - xmin;
	int ymin = n / 2;
	int ymax = bmpsize.height - ymin;

	float realsum = 0.0;
	float imgsum = 0.0;
	float *sum = new float[bmpsize.stride*bmpsize.height];
	float max = 0.0;
	
	float realsummax = 0.0;
	float imgsummax = 0.0;

	for (int y = ymin; y < ymax; ++y)
	{
		for (int x = xmin; x < xmax; ++x)
		{
			for (int j = 0; j < n; j++)
			{
				for (int i = 0; i < m; ++i)
				{
					realsum += bmpdata[(y - ymin + j)*bmpsize.stride + x - xmin + i] * real[j*m + i];
					imgsum += bmpdata[(y - ymin + j)*bmpsize.stride + x - xmin + i] * img[j*m + i];
				}
			}
			sum[y*bmpsize.stride + x] = sqrt(realsum*realsum + imgsum*imgsum);
			if (sum[y*bmpsize.stride + x]>max) max = sum[y*bmpsize.stride + x];
			realsum = 0.0;
			imgsum = 0.0;
		}
	}

	
	float ratio = max / 255;
	for (int y = ymin; y < ymax; ++y)
	{
		for (int x = xmin; x < xmax; ++x)
		{
			output[y*bmpsize.stride + x] = sum[y*bmpsize.stride + x] / ratio;
		}
	}
	
	delete[]sum;
	sum = NULL;
}
/*
void GaborFilterBank(int u, int v, int m, int n)
{
	
	
	float *real = new float[m*n];
	float *img = new float[m*n];

	float pi = 3.14159265358;
	float fmax = 0.25;
	float gama = sqrt(2);
	float eta = sqrt(2);

	for (int i = 0; i < u; ++i)
	{
		float fu = fmax / (pow(sqrt(2), i));
		float alpha = fu / gama;
		float beta = fu / eta;

		for (int j = 0; j < v; ++j)
		{
			float tetav = (j*pi) / v;
			for (int y = 0; y < n; ++y)
			{
				for (int x = 0; x < m; ++x)
				{
					float xprime = (x - m / 2)*cos(tetav) + (y - n / 2)*sin(tetav);
					float yprime = -(x - m / 2)*sin(tetav) + (y - n / 2)*cos(tetav);
					real[y*m + x] = (pow(fu, 2) / (pi*gama*eta))*exp(-(pow(alpha, 2)*pow(xprime, 2) + pow(beta, 2)*pow(yprime, 2)))*cos(-2 * pi*fu*xprime);
					img[y*m + x] = (pow(fu, 2) / (pi*gama*eta))*exp(-(pow(alpha, 2)*pow(xprime, 2) + pow(beta, 2)*pow(yprime, 2)))*sin(-2 * pi*fu*xprime);
					GaborFilter(real, img, m, n, normalbmpdata, output, bmpsize)
				}
			}
		
		}
	}
}
*/