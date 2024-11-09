//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <datatype.h>
#include <acceleratorduo.h>
#include <accloadduo.h>
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
    // int device_index = 0;

    // std::cout << "Open the device" << device_index << std::endl;
    // auto device = xrt::device(device_index);
    // std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
    // std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

    // std::cout << "Load the xclbin " << binaryFile << std::endl;
    // xrt::uuid uuid = device.load_xclbin("./combine_top.xclbin");


    acceleratorArray myaccArracy(binaryFile);

    // std::vector<accelerator> myacc = loadACC(binaryFile);



    // accelerator ACC1 = myacc[0];
    // accelerator ACC2 = myacc[1];

    myaccArracy.acclist.preparation();
    myaccArracy.acclist.loadweight();


    for(int i = 1000; i < 1001; i++){
        std::string sfilename = "../data/input_images/image_";
        // std::string sfilename = "../data/input_images/image_0.bin";
        sfilename += std::to_string(i);
        sfilename += ".bin";
        std::string labelfilename = "../data/input_images_label/label_";
        // std::string labelfilename = "../data/input_images_label/label_0.bin"
        labelfilename += std::to_string(i);
        labelfilename += ".bin";

        std::cout << sfilename << labelfilename << std::endl;
        
        myaccArracy.acclist.loadinput(sfilename);

        myaccArracy.acclist.inference();

        myaccArracy.acclist.checkresult(labelfilename);
    }

    // v_dt * result_holder_map = result_holder.map<v_dt *>();

    return 0;
    
}