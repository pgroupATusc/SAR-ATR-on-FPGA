#include <iostream>
#include <cstring>
#include <datatype.h>
#include <accelerator.h>
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
    std::vector<accelerator> acclist;

    acceleratorArray(std::string binaryFile){
        std::vector<accelerator> results;

        int device_index = 0;

        std::cout << "Open the device" << device_index << std::endl;
        auto device = xrt::device(device_index);
        std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
        std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

        std::cout << "Load the xclbin " << binaryFile << std::endl;
        xrt::uuid uuid = device.load_xclbin(binaryFile);

        acclist.push_back(accelerator(device, uuid, 1));
        // acclist.push_back(accelerator(device, uuid, 2));
    }
    
    void preparation(){
        acclist[0].preparation();
        // acclist[1].preparation();
    }
    void loadweight(){
        acclist[0].loadweight();
        // acclist[1].loadweight();
    }
    void inference(){
        acclist[0].inference();
        // acclist[1].inference();
    }
    void loadinput(std::string filename){
        acclist[0].loadinput(filename);
        // acclist[1].loadinput(filename);
    }
    void checkresult(std::string filename){
        acclist[0].checkresult(filename);
        // acclist[1].checkresult(filename);
    }

};



#endif