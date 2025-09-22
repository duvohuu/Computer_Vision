#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void histogram(string const& name, Mat const& Image)
{
	int bin = 255;
	int histsize[] = { bin };
	float range[] = { 0,255 };
	const float* ranges[] = { range };
	Mat hist;
	int chanel[] = { 0 };
	int hist_heigt = 256;
	Mat hist_image = Mat::zeros(hist_heigt, bin, CV_8SC3);
	calcHist(&Image, 1, chanel, Mat(), hist, 1, histsize, ranges, true, false);
	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);
	for (int i = 0; i < bin; i++)
	{
		float binV = hist.at<float>(i);
		int height = cvRound(binV * hist_heigt / max_val);
		line(hist_image, Point(i, hist_heigt - height), Point(i, hist_heigt), Scalar::all(255));
	}
	imshow(name, hist_image);
}
int main(int argv, char** argc)
{
	float img[16] = { 50,20,30,50,
					80,50,100,110,
					120,160,50,150,
					220,230,240,250 };
	Mat gray_img = Mat(4, 4, CV_32F, img);
	Mat gray_img_his;
	namedWindow("Old gray", WINDOW_FREERATIO);
	namedWindow("New gray", WINDOW_FREERATIO);
	gray_img.convertTo(gray_img, CV_8UC1);
	equalizeHist(gray_img, gray_img_his);
	cout << "Matrix = " << endl << "" << gray_img << endl << endl;
	cout << "Matrix_histogram = " << endl << "" << gray_img_his << endl << endl;
	imshow("Old gray", gray_img);
	imshow("New gray", gray_img_his);
	histogram("Old histogram", gray_img);
	histogram("New histogram ", gray_img_his);
	waitKey();
}
