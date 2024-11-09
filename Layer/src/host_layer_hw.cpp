//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <datatype.h>
#include <utils.h>
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
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
    std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin("./combine_top.xclbin");

    auto krnl_feagg = xrt::kernel(device, uuid, "feagg_top:{feagg_top_1}", xrt::kernel::cu_access_mode::exclusive);
    auto krnl_trans = xrt::kernel(device, uuid, "mmult:{mmult_1}", xrt::kernel::cu_access_mode::exclusive);


    std::cout << "Allocate Buffer in Global Memory hello\n";

    const int imagescale = 128; 
    const int num_vertex = imagescale * imagescale;
    const int num_edge_block = imagescale * 32 * 9;
    const int num_vertex_bank = 128*16;
    const int f_block_len_custom = 1;
    const int BM = 1026;

    std::cout << "Allocate boAX at bank" << "\n";
    xrt::bo boX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(0)); //Match kernel arguments to RTL kernel
    xrt::bo boX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(1)); //Match kernel arguments to RTL kernel
    xrt::bo boX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(2)); //Match kernel arguments to RTL kernel

    xrt::bo boA_p1 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), krnl_feagg.group_id(3));
    xrt::bo boA_p2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), krnl_feagg.group_id(3));

    xrt::bo boAX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(4));
    xrt::bo boAX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(5));
    xrt::bo boAX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(6));

    std::cout << "Define the data mapping\n";

    v_float * boX_map_bank1 = boX_bank1.map<v_float *>();
    v_float * boX_map_bank2 = boX_bank2.map<v_float *>();
    v_float * boX_map_bank3 = boX_bank3.map<v_float *>();
    v_edge * boA_p1_map = boA_p1.map<v_edge *>();
    v_edge * boA_p2_map = boA_p2.map<v_edge *>();
    v_float * boAX_map_bank1 = boAX_bank1.map<v_float *>();
    v_float * boAX_map_bank2 = boAX_bank2.map<v_float *>();
    v_float * boAX_map_bank3 = boAX_bank3.map<v_float *>();

    float fdata;

    load_input(boX_map_bank1, boX_map_bank2, boX_map_bank3, num_vertex);
    create_edges_layer1(boA_p1_map, boA_p2_map);

    boX_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boX_bank2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boX_bank3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boA_p1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boA_p2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // allocate space for self transformation
    xrt::bo boW_self = xrt::bo(device, size_t(sizeof(float) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(6));
    xrt::bo boAX_Merge_bank1 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(7));
    xrt::bo boAX_Merge_bank2 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(8));
    xrt::bo boAX_Merge_bank3 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(9));
    
    v_dt* boW_map_self = boW_self.map<v_dt*>();
    v_dt* boAX_Merge_map_bank1 = boAX_Merge_bank1.map<v_dt*>();
    v_dt* boAX_Merge_map_bank2 = boAX_Merge_bank2.map<v_dt*>();
    v_dt* boAX_Merge_map_bank3 = boAX_Merge_bank3.map<v_dt*>();
     // allocate space for neighbor transformation

    xrt::bo boW_neighbor = xrt::bo(device, size_t(sizeof(float) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), krnl_trans.group_id(6));
    xrt::bo boC_bank1 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(7));
    xrt::bo boC_bank2 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(8));
    xrt::bo boC_bank3 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(9));

    v_dt* boW_neighbor_map = boW_neighbor.map<v_dt*>();
    v_dt* boC_map_bank1 = boC_bank1.map<v_dt*>();
    v_dt* boC_map_bank2 = boC_bank2.map<v_dt*>();
    v_dt* boC_map_bank3 = boC_bank3.map<v_dt*>();

    load_gconv1(boW_map_self, boW_neighbor_map);

    boW_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // define and allocate the space the first pooling layer =====================================================================
    // const int num_pool1_edges = 8*64*4;
    // const int BM_pooling1 = 516;
    // const int num_vertex_pooling1 = 64*64;

    // xrt::bo boA_pool1 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool1_edges), krnl_feagg.group_id(3));
    // xrt::bo boC_pooling1_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex_pooling1), krnl_feagg.group_id(4));
    // xrt::bo boC_pooling1_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex_pooling1), krnl_feagg.group_id(5));
    // xrt::bo boC_pooling1_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex_pooling1), krnl_feagg.group_id(6));

    // v_edge * boA_pool1_map = boA_pool1.map<v_edge*>();

    // create_edges_pooling1(boA_pool1_map);
    // boA_pool1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // ===========================================================================================================================


    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the feature aggregation\n";
    auto run_p1 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p1, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2, 
        f_block_len_custom, imagescale, 
        1, 0, 0, 0, num_vertex_bank, 0); 
    auto run_p2 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p2, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
         f_block_len_custom, imagescale,
          0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0); 
    
    run_p1.wait();
    run_p2.wait();
    std::cout << "Finish the execution of feature aggregation\n";
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the self transformation\n";

    auto run_self_trans = krnl_trans(
        boX_bank1, boX_bank2, boX_bank3,
        (v_dt *) 0, (v_dt *) 0,  (v_dt *) 0, 
        boW_self, boAX_Merge_bank1, boAX_Merge_bank2, boAX_Merge_bank3,
        BM, 0, 0, 0);

    run_self_trans.wait();
    std::cout << "Finish the execution of self transformation\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";



    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the neighbor transformation\n";

    auto run_neigh_trans = krnl_trans(
        boAX_bank1, boAX_bank2, boAX_bank3,
        boAX_Merge_bank1, boAX_Merge_bank2,  boAX_Merge_bank3, 
        boW_neighbor, boC_bank1, boC_bank2, boC_bank3,
        BM, 1, 1, 1);

    run_neigh_trans.wait();
    std::cout << "Finish the execution of neighbor transformation\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    boAX_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_Merge_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_Merge_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_Merge_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    verify_gconv1(boC_map_bank1, boC_map_bank2, boC_map_bank3);

}