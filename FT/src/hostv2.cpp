//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <datatype.h>
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


    int BM = 1026;
    printf("BM is %d\n", BM);

    // Read settings
    std::string binaryFile = "./mmult.xclbin";
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
    std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin("./mmult.xclbin");

    auto krnl = xrt::kernel(device, uuid, "mmult:{mmult_1}", xrt::kernel::cu_access_mode::exclusive);



    std::cout << "Allocate Buffer in Global Memory hello\n";
    
    std::cout << "Allocate boAX at bank" << "\n";
    auto boAX = xrt::bo(device, size_t(sizeof(float) * BM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0)); //Match kernel arguments to RTL kernel
    std::cout << "Allocate boW at bank"  << "\n";
    auto boW = xrt::bo(device, size_t(sizeof(float) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(1));
    std::cout << "Allocate boC at bank"  << "\n";
    auto boC = xrt::bo(device, size_t(sizeof(float) * BM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(2));

    std::cout << "Define the data mapping\n";

    auto boAX_map = boAX.map<v_dt*>();
    auto boW_map = boW.map<v_dt*>();
    auto boC_map = boC.map<v_dt*>();

    srand(time(0));


    float fdata;
    std::ifstream fin("../data/input_test1.bin", std::ios::binary);
    

    // generate random numbers for boAX_map, boW_map
    printf("initailze the data in boAX_map\n");
    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384 && j == 0 && kk == 0){
                    fin.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boAX_map[i*BLOCK_NUM + kk].data[j] = fdata;
                }
                else{
                    boAX_map[i*BLOCK_NUM + kk].data[j] = 0;
                }
                // printf("%.0f ", boAX_map[i].data[j]);
            }
        }
        // if( i % BLOCK_NUM == BLOCK_NUM - 1) printf("\n");
    }

    std::ifstream fin_weight("../data/gconv1-lin_r-weight.bin", std::ios::binary);
    
    printf("initailze the data in W\n");
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM * VDATA_SIZE ; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){  
            for(int j = 0; j < VDATA_SIZE; j++)
            {
                if(i == 0){
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_map[i*BLOCK_NUM + kk].data[j] = fdata;
                }
                else{
                    boW_map[i*BLOCK_NUM + kk].data[j] = 0;
                }
                // printf("%.0f ", boW_map[i].data[j]);
            }
        }
        // if( i % BLOCK_NUM == BLOCK_NUM - 1) printf("\n");
    }


    std::vector<v_dt> boC_golden;

    boC_golden.resize(BM * BLOCK_NUM * VDATA_SIZE);

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();

    printf("calculate the golden result\n");
    for(int i = 0; i < BM; i++){
        for(int j = 0; j < BLOCK_NUM; j++){
            for(int ii = 0; ii < VDATA_SIZE; ii++){
                for(int jj = 0; jj < VDATA_SIZE; jj++){
                    int index1 = (i * VDATA_SIZE + ii) * BLOCK_NUM + j;
                    int index2 = jj;

                    boC_golden[index1].data[index2]  = 0;

                    for(int k = 0; k < BLOCK_NUM; k++){
                        for(int kk = 0; kk < VDATA_SIZE; kk++){
                            int in1 = (i * VDATA_SIZE + ii) * BLOCK_NUM + k;
                            int in2 = kk;

                            int in3 = (k * VDATA_SIZE + kk) * BLOCK_NUM + j;
                            int in4 = jj;

                            boC_golden[index1].data[index2] += boAX_map[in1].data[in2] * boW_map[in3].data[in4];
                        }
                    }
                }
            }
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "software multiplication took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    boAX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW.sync(XCL_BO_SYNC_BO_TO_DEVICE);



    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the kernel\n";
    auto run = krnl(boAX, boW, boC, BM); //DATA_SIZE=size
    run.wait();
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    std::cout << "Get the output data from the device" << std::endl;
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::ifstream fin_selfT("../data/selfT.bin", std::ios::binary);

    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384){
                    fin_selfT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    if(fdata != boC_map[i * BLOCK_NUM + kk].data[j])
                    printf("(%d, %d) golden: %f, output: %f\n", i * BLOCK_NUM + kk, j, fdata, boC_map[i * BLOCK_NUM + kk].data[j]);
                }
            }
        }
    }



    std::cout << "TEST PASSED\n";
    return 0;
}
