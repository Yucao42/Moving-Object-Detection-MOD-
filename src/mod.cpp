#include"header.hpp"

//Int to string
string itos(int i)
{
    stringstream s;
    s << i;
    return s.str();
}

//Judge if point is in ROI
bool MOD::ROI_mod(int x1, int y1)
{
    if (x1 >= width * w_div && x1 <= width - width * w_div && y1 >= height * h_div && y1 <= height - height * h_div)
        return true;
    return false;
}

//Image pre-processing
void MOD::pre_process()
{

    Harris_num = 0;
    F_prepoint.clear();
    F_nextpoint.clear();
    height = frame.rows*scale;
    width = frame.cols*scale;
    dsize = Size(frame.cols*scale, frame.rows*scale);
    img_scale = Mat(dsize, CV_32SC3);
    img_temp = Mat(dsize, CV_32SC3);
    img_scale = frame.clone();
    img_temp = frame.clone();
    cvtColor(img_scale, gray, CV_BGR2GRAY);
    if(mode == UNDISTORT_IMG)
    {
        undist -> undistort_img(gray, gray);
        imshow("oh gray",gray);
    }
    //Detection square's width
    rec_width = frame.cols / 16;

    //Current frame number
    cout << " Frame No. :   " << cal << endl;
    return;
}


void MOD::optical_flow_check()
{
    double limit_of_check = 2120;
    int limit_edge_corner = 5;
    for (int i = 0; i < state.size(); i++)
        if (state[i] != 0)
        {

           int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
           int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
           int x1 = prepoint[i].x, y1 = prepoint[i].y;
           int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
           if ((x1 < limit_edge_corner || x1 >= gray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= gray.cols - limit_edge_corner
            || y1 < limit_edge_corner || y1 >= gray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= gray.rows - limit_edge_corner))
           {
               state[i] = 0;
               continue;
           }
        double sum_check = 0;
        for (int j = 0; j < 9; j++)
            sum_check += abs(prevgray.at<uchar>(y1 + dy[j], x1 + dx[j]) - gray.at<uchar>(y2 + dy[j], x2 + dx[j]));
        if (sum_check>limit_of_check) state[i] = 0;

        if (state[i])
         {
            Harris_num++;
            F_prepoint.push_back(prepoint[i]);
            F_nextpoint.push_back(nextpoint[i]);
         }
        }
    return;
}


bool MOD::stable_judge()
{
    int stable_num = 0;
    double limit_stalbe = 0.5;
    for (int i = 0; i < state.size(); i++)
        if (state[i])
        {
        if (sqrt((prepoint[i].x - nextpoint[i].x)*(prepoint[i].x - nextpoint[i].x) + (prepoint[i].y - nextpoint[i].y)*(prepoint[i].y - nextpoint[i].y)) < limit_stalbe) stable_num++;
        }
    if (stable_num*1.0 / Harris_num > 0.2) return 1;
    return 0;
}

//Draw the  detection results
void MOD::draw_detection()
{
    int tt = 10;
    double flag_meiju[100][100];
    memset(flag_meiju, 0, sizeof(flag_meiju));
    for (int i = 0; i < gray.rows / tt; i++)
        for (int j = 0; j < gray.cols / tt; j++)
        {
            double x1 = i*tt + tt / 2;
            double y1 = j*tt + tt / 2;
            for (int k = 0; k < T.size(); k++)
                if (ROI_mod(T[k].x, T[k].y) && sqrt((T[k].x - y1)*(T[k].x - y1) + (T[k].y - x1)*(T[k].y - x1)) < tt*sqrt(2)) flag_meiju[i][j]++;
        }
    double mm = 0;
    int mark_i = 0, mark_j = 0;
    for (int i = 0; i < gray.rows / tt; i++)
        for (int j = 0; j < gray.cols / tt; j++)
            if (ROI_mod(j*tt, i*tt) && flag_meiju[i][j] > mm)
            {
                mark_i = i;
                mark_j = j;
                mm = flag_meiju[i][j];
                if (mm < 2) continue;
                rectangle(frame, Point(mark_j*tt / scale - rec_width, mark_i*tt / scale + rec_width), Point(mark_j*tt / scale + rec_width, mark_i*tt / scale - rec_width), Scalar(0, 255, 255), 3);
            }
    if (mm > 1111) rectangle(frame, Point(mark_j*tt / scale - rec_width, mark_i*tt / scale + rec_width), Point(mark_j*tt / scale + rec_width, mark_i*tt / scale - rec_width), Scalar(0, 255, 255), 3);
}

void MOD::process()
{

    cap>>frame;
    if(mode!=NONE)
        undist->undistort_map(frame);
    for (;;)
    {
        double t = (double)cvGetTickCount();
        cap>>frame;
        if (frame.empty()) break;
        cal++;
        pre_process();
        //Process every margin
        if (cal % margin != 0)
        {
            continue;
        }

        if (prevgray.data)
        {
            //calcOpticalFlowPyrLK
            goodFeaturesToTrack(prevgray, prepoint, 1000, 0.01, 8, Mat(), 3, true, 0.04);
            if(mode ==UNDISTORT_PTS)
                undist ->undistort_pts(prepoint);
            cornerSubPix(prevgray, prepoint, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
            calcOpticalFlowPyrLK(prevgray, gray, prepoint, nextpoint, state, err, Size(22, 22), 5, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
            optical_flow_check();

            //Find corners
            for (int i = 0; i < state.size(); i++)
            {

                double x1 = prepoint[i].x, y1 = prepoint[i].y;
                double x2 = nextpoint[i].x, y2 = nextpoint[i].y;
                if (state[i] != 0)
                {

                    //Draw all corners
                    circle(img_scale, nextpoint[i], 3, Scalar(255, 0, 255));
                    circle(pre_frame, prepoint[i], 2, Scalar(255, 0, 255));
                }
            }
            cout << Harris_num << endl;

            //F-Matrix
            vector<Point2f> F2_prepoint, F2_nextpoint;
            F2_prepoint.clear();
            F2_nextpoint.clear();
            double errs = 0;
            Mat F = findFundamentalMat(F_prepoint, F_nextpoint, mask, FM_RANSAC, thre_RANSAC, 0.99);
            for (int i = 0; i < mask.rows; i++)
            {
                if (mask.at<uchar>(i, 0) == 0);
                else
                {
                    ///circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
                    double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
                    double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
                    double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
                    double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B);
                    errs += dd;
                    if (dd > 0.1)
                        circle(pre_frame, F_prepoint[i], 6, Scalar(255, 0, 0), 3);
                    else
                    {
                        F2_prepoint.push_back(F_prepoint[i]);
                        F2_nextpoint.push_back(F_nextpoint[i]);
                    }
                }
            }

            F_prepoint = F2_prepoint;
            F_nextpoint = F2_nextpoint;
            cout << "Errors in total is " << errs << "pixels" << endl;


            T.clear();
            for (int i = 0; i < prepoint.size(); i++)
            {
                if (state[i] != 0)
                {
                    double A = F.at<double>(0, 0)*prepoint[i].x + F.at<double>(0, 1)*prepoint[i].y + F.at<double>(0, 2);
                    double B = F.at<double>(1, 0)*prepoint[i].x + F.at<double>(1, 1)*prepoint[i].y + F.at<double>(1, 2);
                    double C = F.at<double>(2, 0)*prepoint[i].x + F.at<double>(2, 1)*prepoint[i].y + F.at<double>(2, 2);
                    double dd = fabs(A*nextpoint[i].x + B*nextpoint[i].y + C) / sqrt(A*A + B*B);
                    line(img_scale, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar{ 255, 255, 0 }, 2);
                    line(pre_frame, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar{ 0, 255, 0 }, 1);

                    //Judge outliers
                    if (dd <= thre_dist2epipolar) continue;
//                    cout << "dis: " << dd << endl;
                    dis[T.size()] = dd;
                    T.push_back(nextpoint[i]);


                    //Draw outliers
                    circle(pre_frame, prepoint[i], 3, Scalar(255, 255, 255), 2);

                    //Epipolar lines
                    if (fabs(B) < 0.0001)
                    {
                        double xx = C / A, yy = 0;
                        double xxx = C / A, yyy = gray.cols;
                        line(pre_frame, Point(xx, yy), Point(xxx, yyy), Scalar::all(-1), 0.01);
                        continue;
                    }
                    double xx = 0, yy = -C / B;
                    double xxx = gray.cols, yyy = -(C + A*gray.cols) / B;
                    if (fabs(yy) > 12345 || fabs(yyy) > 12345)
                    {
                        yy = 0;
                        xx = -C / A;
                        yyy = gray.rows;
                        xxx = -(C + B*yyy) / A;
                    }
                    line(img_scale, Point(xx, yy), Point(xxx, yyy), Scalar::all(-1), 0.01);
                    line(pre_frame, Point(xx, yy), Point(xxx, yyy), Scalar::all(-1), 0.01);
                }
            }

            //Draw detections
            draw_detection();

            //Draw ROI
            rectangle(frame, Point(width * w_div / scale, height * h_div / scale), Point((width - width * w_div) / scale, height * (1 - h_div) / scale), Scalar(255, 0, 0), 1, 0);

            //Show results
            string a = itos(cal / margin), b = ".jpg";
            cvNamedWindow("img_scale", 0);
            imshow("img_scale", img_scale);
            cvNamedWindow("pre", 0);
            imshow("pre", pre_frame);
            cvNamedWindow("frame", 0);
            imshow("frame", frame);
        }

        if (waitKey(27) >= 0)
            break;
        std::swap(prevgray, gray);
        resize(img_temp, pre_frame, dsize);
        t = (double)cvGetTickCount() - t;
        cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
    }
}
