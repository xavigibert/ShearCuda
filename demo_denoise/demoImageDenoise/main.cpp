#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <string.h>

using namespace std;

int main(int argc, char *argv[])
{   
    QApplication a(argc, argv);

    // Parse options
    bool showWindow = true;
    bool detailedTiming = true;
    int numOpt;
    for( numOpt = 0; numOpt < argc-1; numOpt++ )
    {
        if( strcmp("-nw", argv[numOpt+1]) == 0 )
            showWindow = false;
        else if( strcmp("-nt", argv[numOpt+1]) == 0 )
            detailedTiming = false;
        else
            break;
    }

    // Command line arguments
    if( argc - numOpt < 4 )
    {
        cerr << "ERROR: Invalid number of command line arguments!" << endl << endl;
        cout << "Usage:" << endl;
        cout << "    " << argv[0] << " [-nw] [-nt] <image_file> <filters_file> <noise_stdev>" << endl << endl;
        cout << "    Where '-nw' disables display, and '-nt' disables timing" << endl << endl;
        cout << "Example:" << endl;
        cout << "    " << argv[0] << " barbara.gif shearFilters_3_3_4_4_single.bin 20" << endl << endl;
        return 1;
    }
    const char* image_file = argv[1+numOpt];
    const char* filters_file = argv[2+numOpt];
    int sigma = atof(argv[3+numOpt]);

    // TEST CODE
//    const char* image_file = "/scratch0/github/ShearCuda/ShearLab-1.1/Image_Data/barbara.jpg";
//    const char* filters_file = "/scratch0/github/ShearCuda/shearcuda/shearFilters_3_3_4_4_single.bin";
//    int sigma = 20;
    // END TEST

    MainWindow w;
    w.setImageFile(image_file);
    w.setFiltersFile(filters_file);
    w.setSigma(sigma);
    w.enableTiming(detailedTiming);

    if(!w.process())
    {
        cerr << "ERROR: " << argv[0] << " has failed!" << endl;
        return 1;
    }

    if( showWindow )
    {
        w.show();
        return a.exec();
    }
    else
        return 0;
}
