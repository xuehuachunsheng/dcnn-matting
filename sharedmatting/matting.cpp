// g++ -o sharedmatting.so -shared -fPIC sharedmatting.cpp matting.cpp sharedmatting.h `pkg-config opencv --cflags --libs`

#include "sharedmatting.h"
#include <time.h>
#include <string>

using namespace std;

extern "C"{

    extern "C" void test1()
	{
        std::cout << "OK" << std::endl;
	}

	// Directly deal with the image array and trimap array
	extern "C" void sharedMatting(uchar* im, uchar* trimap, uchar* matting, const int height, const int width)
	{
		SharedMatting sm;
        clock_t start, finish;
        //expandKnown()
        start = clock();
        cout << "Load Image";
        sm.loadImage(im, height, width);
        cout << "    over!!!" << endl;
        finish = clock();
        cout <<  double(finish - start) / (CLOCKS_PER_SEC) << endl;

        start = clock();
        cout << "Load trimap";
        sm.loadTrimap(trimap, height, width);
        cout << "    over!!!" << endl;
        finish = clock();
        cout <<  double(finish - start) / (CLOCKS_PER_SEC) << endl;

        start = clock();
        sm.solveAlpha(matting);
        cout << "Solve Alpha    over!!!" << endl;
        finish = clock();
        cout <<  double(finish - start) / (CLOCKS_PER_SEC) << endl;

        start = clock();
        cout << "Set Matte";
        sm.setMatte(matting);
        cout << "    over!!!" << endl;
        finish = clock();
        cout <<  double(finish - start) / (CLOCKS_PER_SEC) << endl;

	}

}