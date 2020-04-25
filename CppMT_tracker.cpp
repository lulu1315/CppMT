#include "CMT.h"
#include "gui.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <algorithm>    // std::min
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace cv;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cmt::CMT;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Rect;
using cv::imread;
using cv::namedWindow;
using cv::Scalar;
using cv::VideoCapture;
using cv::waitKey;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::cerr;
using std::istream;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::min_element;
using std::max_element;
using ::atof;

static string WIN_NAME = "CppMT_tracker";
static string PLY_HEADERS =
    "ply\n"\
    "format ascii 1.0\n"\
    "comment created by lulu the master of everything.\n"\
    "element vertex 5\n"\
    "property float x\n"\
    "property float y\n"\
    "property float z\n"\
    "element face 1\n"\
    "property list uchar int vertex_indices\n"\
    "property float angle\n"\
    "end_header";
    
int main(int argc, char *argv[])
{
    // arguments
    if (!argv[1])
        {
        cout << "please specify input video file" << endl;
        return EXIT_FAILURE;
        }
        
    // argv[1] path to image in [/work1/cgi/Perso/lulu/Projects/Joan/originales/P7/ima_crop]
    // argv[2] path to files out [/work1/cgi/Perso/lulu/Projects/Joan/track/P7/ima_crop]
    // argv[3] image extension [png]
    // argv[4] track_id 
    int fstart  = std::stoi(argv[5]);
    int fend    = std::stoi(argv[6]);
    int finit   = std::stoi(argv[7]);
    int estim_rotation  = std::stoi(argv[8]); //0 : false , 1 : true
    int output_sizex  = std::stoi(argv[9]);
    int showpoints  = std::stoi(argv[10]);
    int shownumbers  = std::stoi(argv[11]);
    int linewidth  = std::stoi(argv[12]);
    
    
    // cmt
    CMT cmt;
    int verbose_flag = 0;
    FILELog::ReportingLevel() = verbose_flag ? logDEBUG : logINFO;
    Output2FILE::Stream() = stdout; //Log to stdout
    cmt.str_detector = "FAST";
    cmt.str_descriptor = "BRISK";
    cmt.consensus.estimate_scale = true;
    cmt.consensus.estimate_rotation = estim_rotation;
    
    // preview window
	cv::namedWindow(WIN_NAME, 1);

    //declare variables
    Mat init_frame, init_frame_gray, trackmask;
    char inputframename[200];
    char initframename[200];
    char outputframename[200];
    char outputmaskname[200];
    char outputplyname[200];
    char fileinfoname[200];
    char verbosetext[200];
    cv::Rect cmtrect;
    ofstream outputply;
    ofstream outputinfo;
    
    //read init frame and convert to gray
    sprintf(initframename,"%s.%04d.%s",argv[1],finit,argv[3]);
    // cout << initframename << endl;
    init_frame=imread(initframename,CV_LOAD_IMAGE_COLOR);
    
    //get res
    int output_sizey;
    int yres = init_frame.rows;
    int xres = init_frame.cols;
    float iratio=(float)xres/(float)yres;
    if (output_sizex == 0) {
        output_sizex = xres;
        output_sizey = yres;
    }
    else {
        output_sizey = int((float)output_sizex/iratio);
    }
    float resize_ratio=(float)output_sizex/(float)xres;
    
    cout << "input  image size : " << xres << "x" << yres << " ratio : " << iratio << endl;
    cout << "output image size : " << output_sizex << "x" << output_sizey << " resize ratio : " << resize_ratio << endl;
    
    cv::resize(init_frame, init_frame, Size(), resize_ratio, resize_ratio, INTER_AREA );
    cv::cvtColor(init_frame,init_frame_gray,cv::COLOR_BGR2GRAY);
    cv::imshow(WIN_NAME, init_frame);
    
    //input init rectangle
    cmtrect = getRect(init_frame,WIN_NAME);
            
    //cmt
    cmt.initialize(init_frame_gray, cmtrect);
    
    //write init infos
    cout << "cmtrect : " << cmtrect << endl;
    cout << "cmtrect.x : " << cmtrect.x << endl;
    cout << "cmtrect.y : " << cmtrect.y << endl;
    cout << "cmtrect.br : " << cmtrect.br() << endl;
    cout << "cmtrect.tl : " << cmtrect.tl() << endl;
    cout << "cmtrect.size : " << cmtrect.size() << endl;
    cout << "cmtrect.height : " << cmtrect.height << endl;
    cout << "cmtrect.width : " << cmtrect.width << endl;
    sprintf(fileinfoname,"%s_%s_info.txt",argv[2],argv[4]);
    outputinfo.open(fileinfoname);
    outputinfo << finit << endl;
    outputinfo << output_sizex << endl;
    outputinfo << output_sizey << endl;
    outputinfo << cmtrect.x << endl;
    outputinfo << cmtrect.y << endl;
    outputinfo << cmtrect.width << endl;
    outputinfo << cmtrect.height << endl;
    outputinfo.close();
    
    //draw things
    Point2f trackcenter_init = (cmtrect.br() + cmtrect.tl())*0.5;
    cv::rectangle(init_frame, cmtrect ,cv::Scalar(0,0,255));
    //sprintf(verbosetext,"cmt init frame");
    //putText(init_frame,verbosetext,cv::Point(cv::Point(cmtrect.x,cmtrect.y)),cv::FONT_HERSHEY_DUPLEX,.5,cv::Scalar(0,0,255),1,CV_AA);
    cv::circle(init_frame,trackcenter_init,4,cv::Scalar(0,0,255),1,LINE_AA);
    cv::line(init_frame, cv::Point(trackcenter_init.x+10,trackcenter_init.y),cv::Point(trackcenter_init.x-10,trackcenter_init.y), Scalar(0,0,255) ,1,LINE_4);
    cv::line(init_frame, cv::Point(trackcenter_init.x,trackcenter_init.y+10),cv::Point(trackcenter_init.x,trackcenter_init.y-10), Scalar(0,0,255) ,1,LINE_4);
    sprintf(verbosetext,"%.2f,%.2f",trackcenter_init.x,trackcenter_init.y);
    if (shownumbers == 1) {putText(init_frame,verbosetext,trackcenter_init,cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(0,0,255),1,CV_AA);}
        
    //write output image
    sprintf(outputframename,"%s_%s_preview.%04d.%s",argv[2],argv[4],finit,argv[3]);
    // cout << outputframename << endl;
    cv::imwrite(outputframename,init_frame);

    trackmask = Mat::zeros( output_sizey, output_sizex, CV_8UC1 );
    cv::rectangle(trackmask, cmtrect , 255, CV_FILLED);
    sprintf(outputmaskname,"%s_%s_mask.%04d.%s",argv[2],argv[4],finit,argv[3]);
    cv::imwrite(outputmaskname,trackmask);
    
    //write ply file
    sprintf(outputplyname,"%s_%s.%04d.ply",argv[2],argv[4],finit);
    outputply.open(outputplyname);
    outputply << PLY_HEADERS << endl;
    outputply << (float)cmtrect.x/output_sizex << " 0 " << (float)cmtrect.y/output_sizey << endl;
    outputply << (float)(cmtrect.x + cmtrect.width)/output_sizex << " 0 " << (float)cmtrect.y/output_sizey << endl;
    outputply << (float)(cmtrect.x + cmtrect.width)/output_sizex << " 0 " << (float)(cmtrect.y + cmtrect.height)/output_sizey << endl;
    outputply << (float)cmtrect.x/output_sizex << " 0 " << (float)(cmtrect.y + cmtrect.height)/output_sizey << endl;
    outputply << (float)trackcenter_init.x/output_sizex << " 0 " << (float)trackcenter_init.y/output_sizey << endl;
    outputply << "4 0 1 2 3 0" << endl; 
    outputply.close();
    
    cout << "forward tracking" << endl;
	for (int f = finit+1; f <= fend; f++) 
	{
        sprintf(verbosetext,"forward tracking  : %04d",f);
        //local variables
        Mat frame, frame_gray;
        Point2f vertices[4],trackcenter;
        
        //read input image and convert to gray
        sprintf(inputframename,"%s.%04d.%s",argv[1],f,argv[3]);
        cout << "processing : " << inputframename << endl;
        frame=imread(inputframename,CV_LOAD_IMAGE_COLOR);
        
        cv::resize(frame, frame, Size(), resize_ratio, resize_ratio, INTER_AREA );
        cv::cvtColor(frame,frame_gray,cv::COLOR_BGR2GRAY);
        
        //copy image for overlay
        //frame.copyTo(overlay);
        
        // cmt
        cmt.processFrame(frame_gray);
        //tracked points
        if (showpoints == 1) {
            for(size_t i = 0; i < cmt.points_active.size(); i++)
                {
                cv::circle(frame, cmt.points_active[i], 2, Scalar(255,255,255));
                }
        }
        //
        cmt.bb_rot.points(vertices);
        cout << "angle : " << cmt.bb_rot.angle << endl;
        //draw things
        putText(frame,verbosetext,cv::Point(10,30),cv::FONT_HERSHEY_DUPLEX,.5,cv::Scalar(255,255,255),1,CV_AA);
        trackcenter.x=0;
        trackcenter.y=0;
        
        for (int i = 0; i < 4; i++)
        {
            cv::line(frame, vertices[i], vertices[(i+1)%4], Scalar(255,255,255) ,linewidth,LINE_4);
            sprintf(verbosetext,"%.2f,%.2f",vertices[i].x,vertices[i].y);
            if (shownumbers == 1) {putText(frame,verbosetext,cv::Point(vertices[i]),cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,255,255),1,CV_AA);}
            trackcenter.x+=vertices[i].x;
            trackcenter.y+=vertices[i].y;
        }
        trackcenter.x=trackcenter.x/4;
        trackcenter.y=trackcenter.y/4;
        cv::circle(frame,cv::Point(trackcenter.x,trackcenter.y),4,cv::Scalar(255,255,255),1,LINE_AA);
        cv::line(frame, cv::Point(trackcenter.x+10,trackcenter.y),cv::Point(trackcenter.x-10,trackcenter.y), Scalar(255,255,255) ,1,LINE_4);
        cv::line(frame, cv::Point(trackcenter.x,trackcenter.y+10),cv::Point(trackcenter.x,trackcenter.y-10), Scalar(255,255,255) ,1,LINE_4);
        sprintf(verbosetext,"%.2f,%.2f",trackcenter.x,trackcenter.y);
        if (shownumbers == 1) {putText(frame,verbosetext,trackcenter,cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,255,255),1,CV_AA);}
        //cv::addWeighted(overlay, .9 , frame , .1 , 0, frame);
        
        //write output image
        sprintf(outputframename,"%s_%s_preview.%04d.%s",argv[2],argv[4],f,argv[3]);
        cout << "writing : " << outputframename << endl;
        cv::imwrite(outputframename,frame);
        
        //draw mask
        trackmask = Mat::zeros( output_sizey, output_sizex, CV_8UC1 );
        Point maskpoints[1][4];
        for (int i = 0; i < 4; i++)
        {
        maskpoints[0][i]  = Point(vertices[i].x,vertices[i].y);
        maskpoints[0][i]  = Point(vertices[i].x,vertices[i].y);
        }
        const Point* ppt[1] = { maskpoints[0] };
        int npt[] = { 4 };
        cv::fillPoly(trackmask, ppt, npt, 1, 255);
        sprintf(outputmaskname,"%s_%s_mask.%04d.%s",argv[2],argv[4],f,argv[3]);
        cv::imwrite(outputmaskname,trackmask);
    
        //write ply file
        sprintf(outputplyname,"%s_%s.%04d.ply",argv[2],argv[4],f);
        outputply.open(outputplyname);
        outputply << PLY_HEADERS << endl;
        outputply << vertices[1].x/output_sizex << " 0 " << vertices[1].y/output_sizey << endl;
        outputply << vertices[2].x/output_sizex << " 0 " << vertices[2].y/output_sizey << endl;
        outputply << vertices[3].x/output_sizex << " 0 " << vertices[3].y/output_sizey << endl;
        outputply << vertices[0].x/output_sizex << " 0 " << vertices[0].y/output_sizey << endl;
        outputply << trackcenter.x/output_sizex << " 0 " << trackcenter.y/output_sizey << endl;
        outputply << "4 0 1 2 3 " << cmt.bb_rot.angle << endl; 
        outputply.close();
    
        //update preview
		cv::imshow(WIN_NAME, frame);
		cv::waitKey(25);
	}
	
	   cout << "backward tracking" << endl;
       CMT cmt_back;
       cmt_back.initialize(init_frame_gray, cmtrect);
       
	for (int f = finit-1; f >= fstart; f--) 
	{
        sprintf(verbosetext,"backward tracking  : %04d",f);
        //local variables
        Mat frame, frame_gray;
        Point2f vertices[4],trackcenter;
        
        //read input image and convert to gray
        sprintf(inputframename,"%s.%04d.%s",argv[1],f,argv[3]);
        cout << "processing : " << inputframename << endl;
        frame=imread(inputframename,CV_LOAD_IMAGE_COLOR);
        
        cv::resize(frame, frame, Size(), resize_ratio, resize_ratio, INTER_AREA );
        cv::cvtColor(frame,frame_gray,cv::COLOR_BGR2GRAY);
        
        //copy image for overlay
        //frame.copyTo(overlay);
        
        // cmt
        cmt_back.processFrame(frame_gray);
        //tracked points
        if (showpoints == 1) {
        for(size_t i = 0; i < cmt.points_active.size(); i++)
            {
            cv::circle(frame, cmt_back.points_active[i], 2, Scalar(255,255,255));
            }
        }
        //
        cmt_back.bb_rot.points(vertices);
        
        //draw things
        putText(frame,verbosetext,cv::Point(10,30),cv::FONT_HERSHEY_DUPLEX,.5,cv::Scalar(255,255,255),1,CV_AA);
        trackcenter.x=0;
        trackcenter.y=0;
        for (int i = 0; i < 4; i++)
        {
            cv::line(frame, vertices[i], vertices[(i+1)%4], Scalar(255,255,255) ,linewidth,LINE_4);
            sprintf(verbosetext,"%.2f,%.2f",vertices[i].x,vertices[i].y);
            if (shownumbers == 1) {putText(frame,verbosetext,cv::Point(vertices[i]),cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,255,255),1,CV_AA);}
            trackcenter.x+=vertices[i].x;
            trackcenter.y+=vertices[i].y;
        }
        trackcenter.x=trackcenter.x/4;
        trackcenter.y=trackcenter.y/4;
        cv::circle(frame,cv::Point(trackcenter.x,trackcenter.y),4,cv::Scalar(255,255,255),1,LINE_AA);
        cv::line(frame, cv::Point(trackcenter.x+10,trackcenter.y),cv::Point(trackcenter.x-10,trackcenter.y), Scalar(255,255,255) ,1,LINE_4);
        cv::line(frame, cv::Point(trackcenter.x,trackcenter.y+10),cv::Point(trackcenter.x,trackcenter.y-10), Scalar(255,255,255) ,1,LINE_4);
        sprintf(verbosetext,"%.2f,%.2f",trackcenter.x,trackcenter.y);
        if (shownumbers == 1) {putText(frame,verbosetext,trackcenter,cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,255,255),1,CV_AA);}
        //cv::addWeighted(overlay, .9 , frame , .1 , 0, frame);
        
        //write output image
        sprintf(outputframename,"%s_%s_preview.%04d.%s",argv[2],argv[4],f,argv[3]);
        cout << "writing : " << outputframename << endl;
        cv::imwrite(outputframename,frame);
        
        //draw mask
        trackmask = Mat::zeros( output_sizey, output_sizex, CV_8UC1 );
        Point maskpoints[1][4];
        for (int i = 0; i < 4; i++)
        {
        maskpoints[0][i]  = Point(vertices[i].x,vertices[i].y);
        maskpoints[0][i]  = Point(vertices[i].x,vertices[i].y);
        }
        const Point* ppt[1] = { maskpoints[0] };
        int npt[] = { 4 };
        cv::fillPoly(trackmask, ppt, npt, 1, 255);
        sprintf(outputmaskname,"%s_%s_mask.%04d.%s",argv[2],argv[4],f,argv[3]);
        cv::imwrite(outputmaskname,trackmask);
        
        //write ply file
        sprintf(outputplyname,"%s_%s.%04d.ply",argv[2],argv[4],f);
        outputply.open(outputplyname);
        outputply << PLY_HEADERS << endl;
        outputply << vertices[1].x/output_sizex << " 0 " << vertices[1].y/output_sizey << endl;
        outputply << vertices[2].x/output_sizex << " 0 " << vertices[2].y/output_sizey << endl;
        outputply << vertices[3].x/output_sizex << " 0 " << vertices[3].y/output_sizey << endl;
        outputply << vertices[0].x/output_sizex << " 0 " << vertices[0].y/output_sizey << endl;
        outputply << trackcenter.x/output_sizex << " 0 " << trackcenter.y/output_sizey << endl;
        outputply << "4 0 1 2 3 " << cmt_back.bb_rot.angle << endl; 
        outputply.close();
    
        //update preview
		cv::imshow(WIN_NAME, frame);
		cv::waitKey(25);
	}

	return EXIT_SUCCESS;
};
