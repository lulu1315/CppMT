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
    "element face 0\n"\
    "property list uchar int vertex_indices\n"\
    "end_header";
    
int main(int argc, char *argv[])
{
    // arguments
    if (!argv[1])
        {
        cout << "please specify input video file" << endl;
        return EXIT_FAILURE;
        }
        
    // argv[1] path to image [/work1/cgi/Perso/lulu/Projects/Joan/originales/P7/ima_crop]
    // argv[2] image extension [png]
    // argv[3] track_id 
    int fstart  = std::stoi(argv[4]);
    int fend    = std::stoi(argv[5]);
    int finit   = std::stoi(argv[6]);
    int estim_rotation  = std::stoi(argv[7]); //0 : false , 1 : true

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
    Mat init_frame, init_frame_gray;
    char inputframename[200];
    char initframename[200];
    char outputframename[200];
    char outputplyname[200];
    char verbosetext[200];
    cv::Rect cmtrect;
    ofstream outputply;
    
    //read init frame and convert to gray
    sprintf(initframename,"%s.%04d.%s",argv[1],finit,argv[2]);
    // cout << initframename << endl;
    init_frame=imread(initframename,CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(init_frame,init_frame_gray,cv::COLOR_BGR2GRAY);
    cv::imshow(WIN_NAME, init_frame);
    
    //input init rectangle
    cmtrect = getRect(init_frame,WIN_NAME);
            
    //cmt
    cmt.initialize(init_frame_gray, cmtrect);
    
    //draw things
    Point2f trackcenter_init = (cmtrect.br() + cmtrect.tl())*0.5;
    cv::rectangle(init_frame, cmtrect ,cv::Scalar(0,255,0));
    sprintf(verbosetext,"cmt init frame");
    putText(init_frame,verbosetext,cv::Point(cv::Point(cmtrect.x,cmtrect.y)),cv::FONT_HERSHEY_DUPLEX,.5,cv::Scalar(0,255,0),1,CV_AA);
    cv::circle(init_frame,trackcenter_init,4,cv::Scalar(0,255,0),1,LINE_AA);
    cv::line(init_frame, cv::Point(trackcenter_init.x+10,trackcenter_init.y),cv::Point(trackcenter_init.x-10,trackcenter_init.y), Scalar(0,255,0) ,1,LINE_4);
    cv::line(init_frame, cv::Point(trackcenter_init.x,trackcenter_init.y+10),cv::Point(trackcenter_init.x,trackcenter_init.y-10), Scalar(0,255,0) ,1,LINE_4);
    sprintf(verbosetext,"%.2f,%.2f",trackcenter_init.x,trackcenter_init.y);
    putText(init_frame,verbosetext,trackcenter_init,cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(0,255,0),1,CV_AA);
        
    //write output image
    sprintf(outputframename,"%s_%s.%04d.%s",argv[1],argv[3],finit,argv[2]);
    // cout << outputframename << endl;
    cv::imwrite(outputframename,init_frame);
    
    //write ply file
    sprintf(outputplyname,"%s_%s.%04d.ply",argv[1],argv[3],finit);
    outputply.open(outputplyname);
    outputply << PLY_HEADERS << endl;
    outputply << cmtrect.x << " " << cmtrect.y << " 0" << endl;
    outputply << cmtrect.x + cmtrect.width << " " << cmtrect.y << " 0" << endl;
    outputply << cmtrect.x + cmtrect.width << " " << cmtrect.y + cmtrect.height << " 0" << endl;
    outputply << cmtrect.x << " " << cmtrect.y + cmtrect.height << " 0" << endl;
    outputply << trackcenter_init.x << " " << trackcenter_init.y << " 0" << endl;
    outputply.close();
    
    cout << "forward tracking" << endl;
	for (int f = finit+1; f <= fend; f++) 
	{
        sprintf(verbosetext,"forward tracking  : %04d",f);
        //local variables
        Mat frame, frame_gray,overlay;
        Point2f vertices[4],trackcenter;
        
        //read input image and convert to gray
        sprintf(inputframename,"%s.%04d.%s",argv[1],f,argv[2]);
        cout << "processing : " << inputframename << endl;
        frame=imread(inputframename,CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(frame,frame_gray,cv::COLOR_BGR2GRAY);
        
        //copy image for overlay
        frame.copyTo(overlay);
        
        // cmt
        cmt.processFrame(frame_gray);
        cmt.bb_rot.points(vertices);
        
        //draw things
        putText(frame,verbosetext,cv::Point(10,30),cv::FONT_HERSHEY_DUPLEX,.5,cv::Scalar(0,255,0),1,CV_AA);
        trackcenter.x=0;
        trackcenter.y=0;
        for (int i = 0; i < 4; i++)
        {
            cv::line(overlay, vertices[i], vertices[(i+1)%4], Scalar(255,0,0) ,1,LINE_4);
            sprintf(verbosetext,"%.2f,%.2f",vertices[i].x,vertices[i].y);
            putText(overlay,verbosetext,cv::Point(vertices[i]),cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,0,0),1,CV_AA);
            trackcenter.x+=vertices[i].x;
            trackcenter.y+=vertices[i].y;
        }
        trackcenter.x=trackcenter.x/4;
        trackcenter.y=trackcenter.y/4;
        cv::circle(overlay,cv::Point(trackcenter.x,trackcenter.y),4,cv::Scalar(255,0,0),1,LINE_AA);
        cv::line(overlay, cv::Point(trackcenter.x+10,trackcenter.y),cv::Point(trackcenter.x-10,trackcenter.y), Scalar(255,0,0) ,1,LINE_4);
        cv::line(overlay, cv::Point(trackcenter.x,trackcenter.y+10),cv::Point(trackcenter.x,trackcenter.y-10), Scalar(255,0,0) ,1,LINE_4);
        sprintf(verbosetext,"%.2f,%.2f",trackcenter.x,trackcenter.y);
        putText(overlay,verbosetext,trackcenter,cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,0,0),1,CV_AA);
        cv::addWeighted(overlay, .5 , frame , .5 , 0, frame);
        
        //write output image
        sprintf(outputframename,"%s_%s.%04d.%s",argv[1],argv[3],f,argv[2]);
        cout << "writing : " << outputframename << endl;
        cv::imwrite(outputframename,frame);
        
        //write ply file
        sprintf(outputplyname,"%s_%s.%04d.ply",argv[1],argv[3],f);
        outputply.open(outputplyname);
        outputply << PLY_HEADERS << endl;
        outputply << vertices[1].x << " " << vertices[1].y << " 0.0" << endl;
        outputply << vertices[2].x << " " << vertices[2].y << " 0.0" << endl;
        outputply << vertices[3].x << " " << vertices[3].y << " 0.0" << endl;
        outputply << vertices[0].x << " " << vertices[0].y << " 0.0" << endl;
        outputply << trackcenter.x << " " << trackcenter.y << " 0.0" << endl;
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
        Mat frame, frame_gray,overlay;
        Point2f vertices[4],trackcenter;
        
        //read input image and convert to gray
        sprintf(inputframename,"%s.%04d.%s",argv[1],f,argv[2]);
        cout << "processing : " << inputframename << endl;
        frame=imread(inputframename,CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(frame,frame_gray,cv::COLOR_BGR2GRAY);
        
        //copy image for overlay
        frame.copyTo(overlay);
        
        // cmt
        cmt_back.processFrame(frame_gray);
        cmt_back.bb_rot.points(vertices);
        
        //draw things
        putText(frame,verbosetext,cv::Point(10,30),cv::FONT_HERSHEY_DUPLEX,.5,cv::Scalar(0,255,0),1,CV_AA);
        trackcenter.x=0;
        trackcenter.y=0;
        for (int i = 0; i < 4; i++)
        {
            cv::line(overlay, vertices[i], vertices[(i+1)%4], Scalar(255,0,0) ,1,LINE_4);
            sprintf(verbosetext,"%.2f,%.2f",vertices[i].x,vertices[i].y);
            putText(overlay,verbosetext,cv::Point(vertices[i]),cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,0,0),1,CV_AA);
            trackcenter.x+=vertices[i].x;
            trackcenter.y+=vertices[i].y;
        }
        trackcenter.x=trackcenter.x/4;
        trackcenter.y=trackcenter.y/4;
        cv::circle(overlay,cv::Point(trackcenter.x,trackcenter.y),4,cv::Scalar(255,0,0),1,LINE_AA);
        cv::line(overlay, cv::Point(trackcenter.x+10,trackcenter.y),cv::Point(trackcenter.x-10,trackcenter.y), Scalar(255,0,0) ,1,LINE_4);
        cv::line(overlay, cv::Point(trackcenter.x,trackcenter.y+10),cv::Point(trackcenter.x,trackcenter.y-10), Scalar(255,0,0) ,1,LINE_4);
        sprintf(verbosetext,"%.2f,%.2f",trackcenter.x,trackcenter.y);
        putText(overlay,verbosetext,trackcenter,cv::FONT_HERSHEY_DUPLEX,.4,cv::Scalar(255,0,0),1,CV_AA);
        cv::addWeighted(overlay, .5 , frame , .5 , 0, frame);
        
        //write output image
        sprintf(outputframename,"%s_%s.%04d.%s",argv[1],argv[3],f,argv[2]);
        cout << "writing : " << outputframename << endl;
        cv::imwrite(outputframename,frame);
        
        //write ply file
        sprintf(outputplyname,"%s_%s.%04d.ply",argv[1],argv[3],f);
        outputply.open(outputplyname);
        outputply << PLY_HEADERS << endl;
        outputply << vertices[1].x << " " << vertices[1].y << " 0.0" << endl;
        outputply << vertices[2].x << " " << vertices[2].y << " 0.0" << endl;
        outputply << vertices[3].x << " " << vertices[3].y << " 0.0" << endl;
        outputply << vertices[0].x << " " << vertices[0].y << " 0.0" << endl;
        outputply << trackcenter.x << " " << trackcenter.y << " 0.0" << endl;
        outputply.close();
    
        //update preview
		cv::imshow(WIN_NAME, frame);
		cv::waitKey(25);
	}

	return EXIT_SUCCESS;
};
