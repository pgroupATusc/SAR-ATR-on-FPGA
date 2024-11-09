//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <datatype.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <cmath>

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

    const int num_vertex = 128 * 128;
    const int num_edge_block = 128 * 128;

    std::string binaryFile = "./feagg_top.xclbin";
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
    std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin("./feagg_top.xclbin");

    auto krnl = xrt::kernel(device, uuid, "feagg_top:{feagg_top_1}", xrt::kernel::cu_access_mode::exclusive);

    std::cout << "Allocate Buffer in Global Memory hello\n";

    std::vector<v_float> boAX_golden;
    boAX_golden.resize(num_vertex * F_BLOCK_LEN);

    std::cout << "Allocate boAX at bank " << krnl.group_id(0) << "\n";
    auto boX = xrt::bo(device, size_t(sizeof(v_float) * num_vertex * F_BLOCK_LEN), krnl.group_id(0)); //Match kernel arguments to RTL kernel
    std::cout << "Allocate boW at bank " << krnl.group_id(1) << "\n";
    auto boA = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block), krnl.group_id(1));
    std::cout << "Allocate boC at bank"  << krnl.group_id(2) << "\n";
    auto boAX = xrt::bo(device, size_t(sizeof(v_float) * num_vertex * F_BLOCK_LEN), krnl.group_id(2));

    std::cout << "Define the data mapping\n";

    auto boX_map = boX.map<v_float *>();
    auto boA_map = boA.map<v_edge *>();
    auto boAX_map = boAX.map<v_float *>();

    srand(time(0));

    // generate random numbers for boX
    printf("initialize the data in boX_map\n");
    for(int i = 0; i < num_vertex * F_BLOCK_LEN; i++){
        for(int j = 0; j < VDATA_SIZE; j++){
            boX_map[i].data[j] = rand() % 10;
            // if(i < 2) printf("boX_map[%d].data[%d]: %f\n", i, j, boX_map[i].data[j]);
            // if(std::isnan(boX_map[i].data[j])) printf("[init] There exist nan in boX\n");
        }
    }


    // initialze the data in AX
    for(int k = 0; k < 4; k++){  // 4 means four directions of the edges, up, down, left, right
        for(int i = 0; i < 32; i++){  // row index
            for(int j = 0; j < 128; j++){  // column index
                edge iedge;
                if(k == 0){ // define the in-going edge from the up
                    iedge.src = (i - 1) * 128 + j;
                    if(i - 1 < 0) iedge.flag = 0;
                    else iedge.flag = 1;
                }
                else if (k == 1){ // define the in-going edge from the bottom
                    iedge.src = (i + 1)  *128 + j;
                    if(i + 1>= 32) iedge.flag = 0;
                    else iedge.flag = 1;
                }
                else if (k == 2){ // define the in-going edge from the left
                    iedge.src = i * 128 + (j - 1);
                    if(j - 1 < 0) iedge.flag = 0;
                    else iedge.flag = 1;
                }
                else{ // define the in-going edge from the right
                    iedge.src = i * 128 + (j + 1);
                    if(j + 1 >= 128) iedge.flag = 0;
                    else iedge.flag = 1;
                }
                iedge.dst = i * 128 + j;
                iedge.value = 0.25;
                int index = k * 32 * 128 + i * 128 + j;
                // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                boA_map[index].data[0] = iedge;
                boA_map[index].data[1] = iedge;
                boA_map[index].data[2] = iedge;
                boA_map[index].data[3] = iedge;
            }
        }
    }

    // initialize boAX_golden
    for(int i = 0; i < num_vertex * F_BLOCK_LEN; i++){
        for(int j = 0; j < VDATA_SIZE; j++){
            boAX_golden[i].data[j] = 0;
        }
    }

    printf("calculate the golden result\n");

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < NUM_FEA_BANK; i++){
        for(int j = 0; j < num_edge_block; j++){
            edge iedge = boA_map[j].data[i];
            int index_offset = i * 128 * 32;

            int src = iedge.src + index_offset;
            int dst = iedge.dst + index_offset;
            float value = iedge.value;
            for(int k = 0; k < F_BLOCK_LEN; k++){
                if(iedge.flag == 1){
                    v_float iv = boX_map[src * F_BLOCK_LEN + k];
                    for(int kk = 0; kk < VDATA_SIZE; kk++){
                        if(std::isnan(iv.data[kk])) printf("There exist nan in iv; %d\n", src * F_BLOCK_LEN + k);
                        boAX_golden[dst * F_BLOCK_LEN + k].data[kk] += value * iv.data[kk];
                        // if(std::isnan(boAX_golden[dst * F_BLOCK_LEN + k].data[kk])) printf("There exist nan in boAX_golden\n");
                        // if(std::isnan(value)) printf("There exist nan in value\n");
                        // if(std::isnan(iv.data[kk])) printf("There exist nan in iv\n");
                    }
                }
            }
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "software multiplication took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    boX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    int num_vertex_bank = 128*32;

    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the kernel\n";
    auto run = krnl(boX, boA, boAX, num_vertex_bank, num_edge_block); 
    run.wait();
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    std::cout << "Get the output data from the device" << std::endl;
    boAX.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    printf("initialize the data in boX_map\n");
    for(int i = 0; i < num_vertex * F_BLOCK_LEN; i++){
        for(int j = 0; j < VDATA_SIZE; j++){
            if(boAX_golden[i].data[j] != boAX_map[i].data[j])
            printf("(%d, %d) golden: %f, output: %f\n", i, j, boAX_golden[i].data[j], boAX_map[i].data[j]);
        }
    }

    std::cout << "TEST PASSED\n";
    return 0;


}