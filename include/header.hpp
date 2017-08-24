#ifndef HEADER_HPP
#define HEADER_HPP
#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>

#define N 1500

using namespace std;
using namespace cv;

enum UNDISTORT_MODE{
    UNDISTORT_PTS, UNDISTORT_IMG, NONE
};

class undistorter{
public:

    //Construction from image calibration parameter file
    undistorter(const string& calib_file, bool usefisheye){
        useFisheye = usefisheye;
        read_calibfile(calib_file);
    }

    //Read calib file
    void read_calibfile(const string& in){
        FileStorage fs(in, FileStorage::READ);
        if(!in.data())return;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
    }

    void undistort_map(const Mat&img){
        Size img_size = img.size();
        if(useFisheye)
            fisheye::initUndistortRectifyMap(M1, D1, Matx33d::eye(), M2, img_size,CV_16SC2, map1, map2);
        else
        {
            M2 = getOptimalNewCameraMatrix(M1, D1, img_size, 1, img_size, 0);
            initUndistortRectifyMap(M1, D1, Mat(), M2, img_size, CV_16SC2, map1, map2);
        }
    }

    Mat get_M1(){return M1;}

    //Undistortion
    void undistort_img(Mat & img,Mat & img_und){
        remap(img, img_und, map1, map2, INTER_LINEAR);
    }


    void undistort_pts(vector<Point2f> &prepoint){
        if(useFisheye)
            cv::fisheye::undistortPoints(prepoint, prepoint,M1,D1,noArray(),M1);
        else
            cv::undistortPoints(prepoint, prepoint, M1,D1,noArray(),M1);
    }

private:

    //M1 intrinsics matrix, D1 distortion vector, M2 new camera matrix.
    Mat M1, D1, M2;

    //Undistort map
    Mat map1, map2;

    //Use fisheye or not
    bool useFisheye;
};



class MOD{
public:

    MOD(int capture, string& calib,bool usefisheye =true, UNDISTORT_MODE m = NONE, double r_thre = 0.2, double thre_2epi = 5, double h_d = 0.167, double w_d = 0.167 ):
        mode(m),thre_RANSAC(r_thre),thre_dist2epipolar(thre_2epi),h_div(h_d), w_div(w_d)
    {
        cap.open(capture);
        undist = new undistorter(calib,usefisheye);
    }

    MOD(string capture, string& in,bool usefisheye =true, UNDISTORT_MODE m = NONE, double r_thre = 0.2, double thre_2epi = 5, double h_d = 0.167, double w_d = 0.167 ):
        mode(m),thre_RANSAC(r_thre),thre_dist2epipolar(thre_2epi),h_div(h_d), w_div(w_d)
    {
        cap.open(capture);
        undist = new undistorter(in,usefisheye);
    }


    //Judge if point is in ROI
    bool ROI_mod(int x1, int y1);

    //Image preprocessing
    void pre_process();

    //Check optical flow
    void optical_flow_check();
    bool stable_judge();

    //Draw the detection of the moving object
    void draw_detection();

    //Process the video stream
    void process();

    //Parameter tuning
    void setmargin(int m)
    {
        margin = m;
    }

    void setthre_dist2epipolar(double m)
    {
        thre_dist2epipolar = m;
    }

    void setRANSAC_threshold(double thre)
    {
        thre_RANSAC = thre;
    }

    void show_current_params()
    {
        cout<<"Threshold of the distance to regard as outlier in RANSAC fundamental finding is "<< thre_RANSAC<<" pixel."<<endl;
        cout<<"Threshold of the distance to regard as outlier in RANSAC fundamental finding is "<< thre_dist2epipolar<<" pixel."<<endl;
        cout<<"The detection is conducted every "<< margin<<" frames."<<endl;
        cout<<"ROI discards upper and lower "<< h_div<<" parts plus left and right "<<w_div<<"parts."<<endl;
        cout<<"Image is scaled with factor "<<scale<<"."<<endl;
    }

private:

    //Input video
    VideoCapture cap;

    //How to undistort
    UNDISTORT_MODE mode;

    //Parameters
    double thre_RANSAC, h_div, w_div, thre_dist2epipolar;


    //Undistorter
    undistorter* undist;

    //Global variables
    Mat prevgray, gray, flow, cflow, frame, pre_frame, img_scale, img_temp, mask;
    Size dsize;
    vector<Point2f> prepoint, nextpoint;
    vector<Point2f> F_prepoint, F_nextpoint;
    vector<uchar> state;
    vector<float> err;
    double dis[N];

    //T: outliers
    vector<Point2f> T;

    //Frame number
    int cal = 0;

    //Frame width and height and rectangle width
    int width, height, rec_width;

    //Number of Harris feature
    int Harris_num = 0;
    int margin = 4;

    //Scaling factor
    double scale = 1;
};


//Int to string
string itos(int i);

#endif // HEADER_HPP

