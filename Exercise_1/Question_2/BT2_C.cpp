#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat source = (Mat_<uint8_t>(4, 4) << 50, 20, 30, 50,
        80, 50, 100, 110,
        120, 160, 50, 150,
        220, 230, 240, 250);
    Mat dst;
    double thresh = 0, maxValue = 255;
    long double thres = threshold(source, dst, thresh, maxValue, THRESH_OTSU);

    cout << "Image matrix = " << endl << "" << source << endl;
    cout << "Otsu Threshold T = " << thres << endl;
    cout << "Otsu Threshold result" << endl << "" << dst << endl;

    waitKey(0);
    return 0;
}
