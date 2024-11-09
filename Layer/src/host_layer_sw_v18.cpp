//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <datatype.h>
#include <accelerator.h>
#include <accload.h>
#include <utils2.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>


// XRT includes
#include "xrt/xrt_bo.h"
#include <experimental/xrt_xclbin.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"


int main(int argc, char** argv) {

    std::cout << "argc = " << argc << std::endl;
	for(int i=0; i < argc; i++){
	    std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
	}

    std::string binaryFile = "./combine_top.xclbin";

    acceleratorArray myaccArracy(binaryFile);
    myaccArracy.acclist[0].preparation();
    myaccArracy.acclist[0].loadweight();
    

    for(int i = 1; i < 2700; i++){
        std::string sfilename = "../data/input_images/image_";
        // std::string sfilename = "../data/input_images/image_0.bin";
        sfilename += std::to_string(i);
        sfilename += ".bin";
        std::string labelfilename = "../data/input_images_label/label_";
        // std::string labelfilename = "../data/input_images_label/label_0.bin"
        labelfilename += std::to_string(i);
        labelfilename += ".bin";
        std::cout << sfilename << labelfilename << std::endl;
        myaccArracy.acclist[0].loadinput(sfilename);
        myaccArracy.acclist[0].inference(););
        myaccArracy.acclist[0].checkresult(labelfilename);
    }

    // v_dt * result_holder_map = result_holder.map<v_dt *>();

    return 0;
    
}