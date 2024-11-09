#include <iostream>
#include <cstring>
#include <datatype.h>
#include <acceleratorduo.h>
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

// #include <pybind11/pybind11.h>

// namespace py = pybind11;

#ifndef _ACC_load
#define _ACC_load

class acceleratorArray{
public:
    accelerator acclist;

    acceleratorArray(std::string binaryFile){

        int device_index = 0;

        std::cout << "Open the device" << device_index << std::endl;
        auto device = xrt::device(device_index);
        std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
        std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

        std::cout << "Load the xclbin " << binaryFile << std::endl;
        xrt::uuid uuid = device.load_xclbin(binaryFile);

        acclist = accelerator(device, uuid, 1);
    }
    
    void preparation(){
        acclist.preparation();
    }
    void loadweight(){
        acclist.loadweight();
    }
    void inference(){
        acclist.inference();
    }
    void loadinput(std::string filename){
        acclist.loadinput(filename);
    }
    void checkresult(std::string filename){
        acclist.checkresult(filename);
    }

};



#endif