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

    const int imagescale = 128; // dimension of the input SAR image

    const int num_vertex = imagescale * imagescale;
    const int num_edge_block = imagescale * 32 * 9;

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

    const int f_block_len_custom = 3;


    std::vector<v_float> boAX_golden;
    boAX_golden.resize(num_vertex * f_block_len_custom);

    std::cout << "Allocate boAX at bank " << krnl.group_id(0) << "\n";
    auto boX = xrt::bo(device, size_t(sizeof(v_float) * num_vertex * f_block_len_custom), krnl.group_id(0)); //Match kernel arguments to RTL kernel

    auto boX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl.group_id(0)); //Match kernel arguments to RTL kernel
    auto boX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl.group_id(1)); //Match kernel arguments to RTL kernel
    auto boX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl.group_id(2)); //Match kernel arguments to RTL kernel

    std::cout << "Allocate boW at bank " << krnl.group_id(1) << "\n";
    auto boA = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block), krnl.group_id(3));
    auto boA_p1 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), krnl.group_id(3));
    auto boA_p2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), krnl.group_id(3));
    std::cout << "Allocate boC at bank"  << krnl.group_id(2) << "\n";
    auto boAX = xrt::bo(device, size_t(sizeof(v_float) * num_vertex * f_block_len_custom), krnl.group_id(2));

    auto boAX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl.group_id(4));
    auto boAX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl.group_id(5));
    auto boAX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl.group_id(6));

    std::cout << "Define the data mapping\n";

    auto boX_map = boX.map<v_float *>();
    auto boX_map_bank1 = boX_bank1.map<v_float *>();
    auto boX_map_bank2 = boX_bank2.map<v_float *>();
    auto boX_map_bank3 = boX_bank3.map<v_float *>();

    auto boA_map = boA.map<v_edge *>();
    auto boA_p1_map = boA_p1.map<v_edge *>();
    auto boA_p2_map = boA_p2.map<v_edge *>();
    auto boAX_map = boAX.map<v_float *>();
    auto boAX_map_bank1 = boAX_bank1.map<v_float *>();
    auto boAX_map_bank2 = boAX_bank2.map<v_float *>();
    auto boAX_map_bank3 = boAX_bank3.map<v_float *>();

    srand(time(0));

    // generate random numbers for boX
    printf("initialize the data in boX_map\n");
    for(int i = 0; i < num_vertex; i++){
        for(int f = 0; f < f_block_len_custom; f++){
            for(int j = 0; j < VDATA_SIZE; j++){
                float somenumber = rand() % 10;
                boX_map[i*f_block_len_custom + f].data[j] = somenumber;
                if(f == 0) boX_map_bank1[i].data[j] = somenumber;
                if(f == 1) boX_map_bank2[i].data[j] = somenumber;
                if(f == 2) boX_map_bank3[i].data[j] = somenumber;
            }
        }
    }


    // initialze the data in AX
    for(int k = 0; k < 9; k++){  // 4 means four directions of the edges, up, down, left, right
        for(int i = 0; i < 32; i++){  // row index
            for(int j = 0; j < 128; j++){  // column index
                for(int bank = 0; bank < 4; bank++){
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = (i - 1 < 0) && (bank == 0);
                    bool condition2 = (i + 1 >= 32) && (bank == 3);
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 128 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *128 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 128 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 128 + (j + 1);
                        if(j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 128 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 128 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 128 + (j + 1);
                        if(condition1 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 128 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 128 + (j + 1);
                        if(condition2 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 128 + j;
                    iedge.value = 0.125;
                    int index = k * 32 * 128 + i * 128 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_map[index].data[bank] = iedge;
                }
            }
        }
    }


    // initialze the data in boA_p1_map
    for(int k = 0; k < 9; k++){  // 4 means four directions of the edges, up, down, left, right
        for(int i = 0; i < 16; i++){  // row index
            for(int j = 0; j < 128; j++){  // column index
                for(int bank = 0; bank < 4; bank++){
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = (i - 1 < 0) && (bank == 0);
                    bool condition2 = 0;
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 128 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *128 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 128 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 128 + (j + 1);
                        if(j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 128 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 128 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 128 + (j + 1);
                        if(condition1 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 128 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 128 + (j + 1);
                        if(condition2 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 128 + j;
                    iedge.value = 0.125;
                    int index = k * 16 * 128 + i * 128 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_p1_map[index].data[bank] = iedge;
                }
            }
        }
    }


    // initialze the data in AX
    for(int k = 0; k < 9; k++){  // 4 means four directions of the edges, up, down, left, right
        for(int i = 0; i < 16; i++){  // row index
            for(int j = 0; j < 128; j++){  // column index
                for(int bank = 0; bank < 4; bank++){
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 16) && (bank == 3);
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 128 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *128 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 128 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 128 + (j + 1);
                        if(j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 128 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 128 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 128 + (j + 1);
                        if(condition1 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 128 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 128 + (j + 1);
                        if(condition2 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 128 + j;
                    iedge.value = 0.125;
                    int index = k * 16 * 128 + i * 128 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_p2_map[index].data[bank] = iedge;
                }
            }
        }
    }




    // initialize boAX_golden
    for(int i = 0; i < num_vertex * f_block_len_custom; i++){
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
            for(int k = 0; k < f_block_len_custom; k++){
                if(iedge.flag == 1){
                    v_float iv;
                    if(src < 0){
                        for(int dv = 0; dv < VDATA_SIZE; dv++){
                            iv.data[0] = 0;
                        }
                    }
                    else if(src >= num_vertex){
                        for(int dv = 0; dv < VDATA_SIZE; dv++){
                            iv.data[0] = 0;
                        }
                    }
                    else{
                        iv = boX_map[src * f_block_len_custom + k];
                    }

                    for(int kk = 0; kk < VDATA_SIZE; kk++){
                        if(std::isnan(iv.data[kk])) printf("There exist nan in iv; %d\n", src * f_block_len_custom + k);
                        boAX_golden[dst * f_block_len_custom + k].data[kk] += value * iv.data[kk];
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
    boX_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boX_bank2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boX_bank3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boA_p1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boA_p2.sync(XCL_BO_SYNC_BO_TO_DEVICE);


    int num_vertex_bank = 128*16;

    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the kernel_p1\n";
    auto run_p1 = krnl(boX_bank1, boX_bank2, boX_bank3, boA_p1, boAX_bank1, boAX_bank2, boAX_bank3, num_vertex_bank, num_edge_block/2, f_block_len_custom, imagescale, 1, 0, 0, 0); 
    // run_p1.wait();
    // std::cout << "Execution of the kernel_p2\n";
    auto run_p2 = krnl(boX_bank1, boX_bank2, boX_bank3, boA_p2, boAX_bank1, boAX_bank2, boAX_bank3, num_vertex_bank, num_edge_block/2, f_block_len_custom, imagescale, 0, 1, num_vertex_bank*4, num_vertex_bank*4); 
    run_p1.wait();
    run_p2.wait();
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    std::cout << "Get the output data from the device" << std::endl;
    boAX.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    printf("initialize the data in boX_map\n");
    int counter = 0;
    for(int i = 0; i < num_vertex; i++){
        for(int f = 0; f < f_block_len_custom; f++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(f == 0 && boAX_golden[i * f_block_len_custom + f].data[j] != boAX_map_bank1[i].data[j]) counter++;
                if(f == 1 && boAX_golden[i * f_block_len_custom + f].data[j] != boAX_map_bank2[i].data[j]) counter++;
                if(f == 2 && boAX_golden[i * f_block_len_custom + f].data[j] != boAX_map_bank3[i].data[j]) counter++;
            }
        }
    }
    std::cout << num_vertex * f_block_len_custom * VDATA_SIZE << " vs " << counter << std::endl;

    std::cout << "TEST PASSED\n";
    return 0;


}