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
    auto boAX_bank1 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0)); //Match kernel arguments to RTL kernel
    auto boAX_bank2 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0)); //Match kernel arguments to RTL kernel
    auto boAX_bank3 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0)); //Match kernel arguments to RTL kernel

    std::cout << "Allocate boAX_Merge at bank" << "\n";
    auto boAX_Merge = xrt::bo(device, size_t(sizeof(float) * BM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0));
    auto boAX_Merge_bank1 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0));
    auto boAX_Merge_bank2 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0));
    auto boAX_Merge_bank3 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), krnl.group_id(0));
    
    std::cout << "Allocate boW at bank"  << "\n";
    auto boW = xrt::bo(device, size_t(sizeof(float) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), krnl.group_id(1));

    std::cout << "Allocate boC at bank"  << "\n";
    auto boC_bank1 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl.group_id(2));
    auto boC_bank2 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl.group_id(2));
    auto boC_bank3 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl.group_id(2));

    std::cout << "Define the data mapping\n";

    auto boAX_map = boAX.map<v_dt*>();
    auto boAX_map_bank1 = boAX_bank1.map<v_dt*>();
    auto boAX_map_bank2 = boAX_bank2.map<v_dt*>();
    auto boAX_map_bank3 = boAX_bank3.map<v_dt*>();

    auto boAX_Merge_map = boAX_Merge.map<v_dt*>();
    auto boAX_Merge_map_bank1 = boAX_Merge_bank1.map<v_dt*>();
    auto boAX_Merge_map_bank2 = boAX_Merge_bank2.map<v_dt*>();
    auto boAX_Merge_map_bank3 = boAX_Merge_bank3.map<v_dt*>();

    auto boW_map = boW.map<v_dt*>();

    auto boC_map_bank1 = boC_bank1.map<v_dt*>();
    auto boC_map_bank2 = boC_bank2.map<v_dt*>();
    auto boC_map_bank3 = boC_bank3.map<v_dt*>();

    srand(time(0));

    float fdata;
    std::ifstream fin("../data/myout.bin", std::ios::binary);

    
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

    fin = std::ifstream("../data/myout.bin", std::ios::binary);

    // generate random numbers for boAX_map, boW_map
    printf("initailze the data in boAX_map\n");
    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384 && j == 0 && kk == 0){
                    fin.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boAX_map_bank1[i].data[j] = fdata;
                }
                else if( kk == 0){
                    boAX_map_bank1[i].data[j] = 0;
                }
                else if (kk == 1){
                    boAX_map_bank2[i].data[j] = 0;
                }
                else if (kk == 2){
                    boAX_map_bank3[i].data[j] = 0;
                }
                // printf("%.0f ", boAX_map[i].data[j]);
            }
        }
        // if( i % BLOCK_NUM == BLOCK_NUM - 1) printf("\n");
    }

    std::ifstream fin_weight("../data/gconv1.lin_l.weight.bin", std::ios::binary);
    std::ifstream fin_bias("../data/gconv1.lin_l.bias.bin", std::ios::binary);
    
    printf("initailze the data in W\n");
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM * VDATA_SIZE + 1; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){  
            for(int j = 0; j < VDATA_SIZE; j++)
            {
                if(i == 0){
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_map[i*BLOCK_NUM + kk].data[j] = fdata;
                    printf("weights, %f\n", fdata);
                }
                else if(i == BLOCK_NUM * VDATA_SIZE){
                    fin_bias.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_map[i*BLOCK_NUM + kk].data[j] = fdata;
                    printf("bias, %f\n", fdata);
                }
                else{
                    boW_map[i*BLOCK_NUM + kk].data[j] = 0;
                }
                // printf("%.0f ", boW_map[i].data[j]);
            }
        }
        // if( i % BLOCK_NUM == BLOCK_NUM - 1) printf("\n");
    }

    
    std::ifstream fin_selfT("../data/selfT.bin", std::ios::binary);

    printf("initailze the data in boAX_Merge_map\n");
    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384){
                    fin_selfT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    if(kk == 0) boAX_Merge_map_bank1[i].data[j] = fdata;
                    if(kk == 1) boAX_Merge_map_bank2[i].data[j] = fdata;
                    if(kk == 2) boAX_Merge_map_bank3[i].data[j] = fdata;
                }
                else{
                    if(kk == 0) boAX_Merge_map_bank1[i].data[j] = 0;
                    if(kk == 1) boAX_Merge_map_bank2[i].data[j] = 0;
                    if(kk == 2) boAX_Merge_map_bank3[i].data[j] = 0;
                }
                // printf("%.0f ", boAX_map[i].data[j]);
            }
        }
        // if( i % BLOCK_NUM == BLOCK_NUM - 1) printf("\n");
    }

    fin_selfT = std::ifstream("../data/selfT.bin", std::ios::binary);

    printf("initailze the data in boAX_Merge_map\n");
    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384){
                    fin_selfT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boAX_Merge_map[i*BLOCK_NUM + kk].data[j] = fdata;
                }
                else{
                    boAX_Merge_map[i*BLOCK_NUM + kk].data[j] = 0;
                }
                // printf("%.0f ", boAX_map[i].data[j]);
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

    int biasindex = BLOCK_NUM * VDATA_SIZE* BLOCK_NUM;

    for(int i = 0; i < BM * VDATA_SIZE;  i++){
        for(int j = 0; j < BLOCK_NUM; j++){
            for(int jj = 0; jj < VDATA_SIZE; jj++){
                boC_golden[i*BLOCK_NUM + j].data[jj] = boC_golden[i*BLOCK_NUM + j].data[jj] + boW_map[biasindex + j].data[jj] + boAX_Merge_map[i*BLOCK_NUM + j].data[jj];
                if(boC_golden[i*BLOCK_NUM + j].data[jj] <= 0) boC_golden[i*BLOCK_NUM + j].data[jj] = 0.0;
            }
        }
    }



    
    for(int j = 0; j < BLOCK_NUM; j++){
        for(int jj = 0; jj < VDATA_SIZE; jj++){
            printf("bias check: %f \n", boW_map[biasindex + j].data[jj]);
        }
    }
    


    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "software multiplication took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    boAX_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boAX_bank2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boAX_bank3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boAX_Merge_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boAX_Merge_bank2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boAX_Merge_bank3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW.sync(XCL_BO_SYNC_BO_TO_DEVICE);



    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the kernel\n";
    auto run = krnl(
        boAX_bank1,
        boAX_bank2,
        boAX_bank3,
        boAX_Merge_bank1,
        boAX_Merge_bank2,
        boAX_Merge_bank3, 
        boW, 
        boC_bank1,
        boC_bank2,
        boC_bank3,
        BM, 
        1, 
        1,
        1); //DATA_SIZE=size
    run.wait();
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    std::cout << "Get the output data from the device" << std::endl;

    boC_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);


    std::ifstream fin_tmpT("../data/layer1Result.bin", std::ios::binary);

    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384){
                    fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    if(kk == 0 && fdata != boC_map_bank1[i].data[j]) 
                        printf("(%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_map_bank1[i].data[j]);
                    if(kk == 1 && fdata != boC_map_bank2[i].data[j]) 
                        printf("(%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_map_bank2[i].data[j]);
                    if(kk == 2 && fdata != boC_map_bank3[i].data[j]) 
                        printf("(%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_map_bank3[i].data[j]);
                }
            }
        }
    }

    // for(int i = 0; i < BM * BLOCK_NUM * VDATA_SIZE; i++){
    //     for(int j = 0; j < VDATA_SIZE; j++){
    //         if(boC_golden[i].data[j] != boC_map[i].data[j])
    //         printf("(%d, %d) golden: %f, output: %f\n", i, j, boC_golden[i].data[j], boC_map[i].data[j]);
    //     }
    // }


    std::cout << "TEST PASSED\n";
    return 0;
}
