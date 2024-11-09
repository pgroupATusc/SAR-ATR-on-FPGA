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
    auto load_mm1 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{read_AX_fromDDR_bank_1}", xrt::kernel::cu_access_mode::exclusive);
    auto load_mm2 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{read_AX_fromDDR_bank_2}", xrt::kernel::cu_access_mode::exclusive);
    auto load_mm3 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{read_AX_fromDDR_bank_3}", xrt::kernel::cu_access_mode::exclusive);
    auto store_mm1 = xrt::kernel(device, uuid, "store_output_matrix:{store_output_matrix_1}", xrt::kernel::cu_access_mode::exclusive);
    auto store_mm2 = xrt::kernel(device, uuid, "store_output_matrix:{store_output_matrix_2}", xrt::kernel::cu_access_mode::exclusive);
    auto store_mm3 = xrt::kernel(device, uuid, "store_output_matrix:{store_output_matrix_3}", xrt::kernel::cu_access_mode::exclusive);
    auto krnl_mlp = xrt::kernel(device, uuid, "mlp:{mlp_1}");
    
    std::cout << "Allocate Buffer in Global Memory hello\n";

    const int imagescale = 128; 
    const int num_vertex = imagescale * imagescale;
    const int num_edge_block = imagescale * 32 * 9;
    const int num_vertex_bank = 128*16;
    const int f_block_len_custom = 1;
    const int BM = 1026;

    std::cout << "Allocate boAX at bank" << "\n";
    xrt::bo boX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(0)); // 2 //Match kernel arguments to RTL kernel
    xrt::bo boX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(1)); // 14 //Match kernel arguments to RTL kernel
    xrt::bo boX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(2)); // 30  // Match kernel arguments to RTL kernel

    xrt::bo boA_p1 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), krnl_feagg.group_id(3));
    xrt::bo boA_p2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), krnl_feagg.group_id(3));

    xrt::bo boAX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(4)); // 3
    xrt::bo boAX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(5)); // 15
    xrt::bo boAX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), krnl_feagg.group_id(6)); // 31

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
    xrt::bo boW_self = xrt::bo(device, size_t(sizeof(float) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boAX_Merge_bank1 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), load_mm1.group_id(0)); //
    xrt::bo boAX_Merge_bank2 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), load_mm2.group_id(0)); //
    xrt::bo boAX_Merge_bank3 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), load_mm3.group_id(0)); //
    
    v_dt* boW_map_self = boW_self.map<v_dt*>();
    v_dt* boAX_Merge_map_bank1 = boAX_Merge_bank1.map<v_dt*>();
    v_dt* boAX_Merge_map_bank2 = boAX_Merge_bank2.map<v_dt*>();
    v_dt* boAX_Merge_map_bank3 = boAX_Merge_bank3.map<v_dt*>();
     // allocate space for neighbor transformation

    xrt::bo boW_neighbor = xrt::bo(device, size_t(sizeof(float) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boC_bank1 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), store_mm1.group_id(0));
    xrt::bo boC_bank2 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), store_mm2.group_id(0));
    xrt::bo boC_bank3 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), store_mm3.group_id(0));

    v_dt* boW_neighbor_map = boW_neighbor.map<v_dt*>();
    v_dt* boC_map_bank1 = boC_bank1.map<v_dt*>();
    v_dt* boC_map_bank2 = boC_bank2.map<v_dt*>();
    v_dt* boC_map_bank3 = boC_bank3.map<v_dt*>();

    load_gconv1(boW_map_self, boW_neighbor_map);

    boW_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // define and allocate the edges and space the first pooling layer =====================================================================
    const int num_pool1_edges = 8*64*4;
    const int BM_pooling1 = 516;
    const int num_vertex_pooling1 = 64*64;

    xrt::bo boA_pool1 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool1_edges), krnl_feagg.group_id(3));
    xrt::bo boC_pooling1_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 4128), krnl_feagg.group_id(0));
    xrt::bo boC_pooling1_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 4128), krnl_feagg.group_id(1));
    xrt::bo boC_pooling1_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 4128), krnl_feagg.group_id(2));

    v_edge * boA_pool1_map = boA_pool1.map<v_edge*>();

    create_edges_pooling1(boA_pool1_map);
    v_float * boC_pooling1_bank1_map = boC_pooling1_bank1.map<v_float *>();
    v_float * boC_pooling1_bank2_map = boC_pooling1_bank2.map<v_float *>();
    v_float * boC_pooling1_bank3_map = boC_pooling1_bank3.map<v_float *>();
    boA_pool1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
        // ===========================================================================================================================
    // define and allocate edges and space for the gconv2

    const int imagescale_gconv2 = 64; 
    const int num_vertex_gconv2 = imagescale_gconv2 * imagescale_gconv2;
    const int num_edge_block_gconv2 = imagescale_gconv2 * 16 * 9;
    const int num_vertex_bank_gconv2 = 64*16;
    const int f_block_len_custom_gconv2 = 3;
    const int BM_gconv2 = 258;

    xrt::bo boA_gconv2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv2), krnl_feagg.group_id(3));
    xrt::bo boAX_gconv2_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 4128), krnl_feagg.group_id(4));
    xrt::bo boAX_gconv2_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 4128), krnl_feagg.group_id(5));
    xrt::bo boAX_gconv2_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 4128), krnl_feagg.group_id(6));

    v_edge * boA_gconv2_map = boA_gconv2.map<v_edge *>();
    v_float * boAX_gconv2_map_bank1 = boAX_gconv2_bank1.map<v_float *>();
    v_float * boAX_gconv2_map_bank2 = boAX_gconv2_bank2.map<v_float *>();
    v_float * boAX_gconv2_map_bank3 = boAX_gconv2_bank3.map<v_float *>();

    create_edges_gconv2(boA_gconv2_map);
    boA_gconv2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo boW_gconv2_self = xrt::bo(device, size_t(sizeof(float) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE),  krnl_trans.group_id(0));
    xrt::bo boAX_Merge_gconv2_bank1 = xrt::bo(device, size_t(sizeof(float) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), load_mm1.group_id(0));
    xrt::bo boAX_Merge_gconv2_bank2 = xrt::bo(device, size_t(sizeof(float) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), load_mm2.group_id(0));
    xrt::bo boAX_Merge_gconv2_bank3 = xrt::bo(device, size_t(sizeof(float) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), load_mm3.group_id(0));
    
    v_dt* boW_gconv2_map_self = boW_gconv2_self.map<v_dt*>();
    v_dt* boAX_Merge_gconv2_map_bank1 = boAX_Merge_gconv2_bank1.map<v_dt*>();
    v_dt* boAX_Merge_gconv2_map_bank2 = boAX_Merge_gconv2_bank2.map<v_dt*>();
    v_dt* boAX_Merge_gconv2_map_bank3 = boAX_Merge_gconv2_bank3.map<v_dt*>();
     // allocate space for neighbor transformation

    xrt::bo boW_gconv2_neighbor = xrt::bo(device, size_t(sizeof(float) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE),  krnl_trans.group_id(0));
    xrt::bo boC_gconv2_bank1 = xrt::bo(device, size_t(sizeof(float) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), store_mm1.group_id(0));
    xrt::bo boC_gconv2_bank2 = xrt::bo(device, size_t(sizeof(float) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), store_mm2.group_id(0));
    xrt::bo boC_gconv2_bank3 = xrt::bo(device, size_t(sizeof(float) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), store_mm3.group_id(0));

    v_dt* boW_gconv2_neighbor_map = boW_gconv2_neighbor.map<v_dt*>();
    v_dt* boC_gconv2_map_bank1 = boC_gconv2_bank1.map<v_dt*>();
    v_dt* boC_gconv2_map_bank2 = boC_gconv2_bank2.map<v_dt*>();
    v_dt* boC_gconv2_map_bank3 = boC_gconv2_bank3.map<v_dt*>();

    load_gconv2(boW_gconv2_map_self, boW_gconv2_neighbor_map);

    boW_gconv2_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW_gconv2_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //=============================================================================================================================
    // define and allocate the edges and space the secpnd pooling layer 

    const int num_pool2_edges = 8*32*4;
    const int BM_pooling2 = 66;
    const int num_vertex_pooling2 = 32*32;

    xrt::bo boA_pool2 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool2_edges), krnl_feagg.group_id(3));
    xrt::bo boC_pooling2_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 1056), krnl_feagg.group_id(0));
    xrt::bo boC_pooling2_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 1056), krnl_feagg.group_id(1));
    xrt::bo boC_pooling2_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 1056), krnl_feagg.group_id(2));
    v_edge * boA_pool2_map = boA_pool2.map<v_edge*>();
    create_edges_pooling2(boA_pool2_map);
    v_float * boC_pooling2_bank1_map = boC_pooling2_bank1.map<v_float *>();
    v_float * boC_pooling2_bank2_map = boC_pooling2_bank2.map<v_float *>();
    v_float * boC_pooling2_bank3_map = boC_pooling2_bank3.map<v_float *>();
    boA_pool2.sync(XCL_BO_SYNC_BO_TO_DEVICE);     

    //=============================================================================================================================
    // define and allocate edges and space for the gconv3

    const int imagescale_gconv3 = 32; 
    const int num_vertex_gconv3 = imagescale_gconv3 * imagescale_gconv3;
    const int num_edge_block_gconv3 = imagescale_gconv3 * 8 * 9;
    const int num_vertex_bank_gconv3 = 32*8;
    const int f_block_len_custom_gconv3 = 3;
    const int BM_gconv3 = 66;

    xrt::bo boA_gconv3 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv3), krnl_feagg.group_id(3));
    xrt::bo boAX_gconv3_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 1056), krnl_feagg.group_id(0));
    xrt::bo boAX_gconv3_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 1056), krnl_feagg.group_id(1));
    xrt::bo boAX_gconv3_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 1056), krnl_feagg.group_id(2));

    v_edge * boA_gconv3_map = boA_gconv3.map<v_edge *>();
    v_float * boAX_gconv3_map_bank1 = boAX_gconv3_bank1.map<v_float *>();
    v_float * boAX_gconv3_map_bank2 = boAX_gconv3_bank2.map<v_float *>();
    v_float * boAX_gconv3_map_bank3 = boAX_gconv3_bank3.map<v_float *>();

    create_edges_gconv3(boA_gconv3_map);
    boA_gconv3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo boW_gconv3_self = xrt::bo(device, size_t(sizeof(float) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boAX_Merge_gconv3_bank1 = xrt::bo(device, size_t(sizeof(float) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), load_mm1.group_id(0));
    xrt::bo boAX_Merge_gconv3_bank2 = xrt::bo(device, size_t(sizeof(float) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), load_mm2.group_id(0));
    xrt::bo boAX_Merge_gconv3_bank3 = xrt::bo(device, size_t(sizeof(float) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), load_mm3.group_id(0));
    
    v_dt* boW_gconv3_map_self = boW_gconv3_self.map<v_dt*>();
    v_dt* boAX_Merge_gconv3_map_bank1 = boAX_Merge_gconv3_bank1.map<v_dt*>();
    v_dt* boAX_Merge_gconv3_map_bank2 = boAX_Merge_gconv3_bank2.map<v_dt*>();
    v_dt* boAX_Merge_gconv3_map_bank3 = boAX_Merge_gconv3_bank3.map<v_dt*>();

    xrt::bo boW_gconv3_neighbor = xrt::bo(device, size_t(sizeof(float) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boC_gconv3_bank1 = xrt::bo(device, size_t(sizeof(float) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), store_mm1.group_id(0));
    xrt::bo boC_gconv3_bank2 = xrt::bo(device, size_t(sizeof(float) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), store_mm2.group_id(0));
    xrt::bo boC_gconv3_bank3 = xrt::bo(device, size_t(sizeof(float) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), store_mm3.group_id(0));

    v_dt* boW_gconv3_neighbor_map = boW_gconv3_neighbor.map<v_dt*>();
    v_dt* boC_gconv3_map_bank1 = boC_gconv3_bank1.map<v_dt*>();
    v_dt* boC_gconv3_map_bank2 = boC_gconv3_bank2.map<v_dt*>();
    v_dt* boC_gconv3_map_bank3 = boC_gconv3_bank3.map<v_dt*>();

    load_gconv3(boW_gconv3_map_self, boW_gconv3_neighbor_map);

    boW_gconv3_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW_gconv3_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);   

       // define and allocate the edges and space the third pooling layer 

    const int num_pool3_edges = 4*16*4;
    const int BM_pooling3 = 18;
    const int num_vertex_pooling3 = 16*16;

    xrt::bo boA_pool3 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool3_edges), krnl_feagg.group_id(3));
    xrt::bo boC_pooling3_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 288), krnl_feagg.group_id(0));
    xrt::bo boC_pooling3_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 288), krnl_feagg.group_id(1));
    xrt::bo boC_pooling3_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 288), krnl_feagg.group_id(2));
    v_edge * boA_pool3_map = boA_pool3.map<v_edge*>();
    create_edges_pooling3(boA_pool3_map);
    v_float * boC_pooling3_bank1_map = boC_pooling3_bank1.map<v_float *>();
    v_float * boC_pooling3_bank2_map = boC_pooling3_bank2.map<v_float *>();
    v_float * boC_pooling3_bank3_map = boC_pooling3_bank3.map<v_float *>();
    boA_pool3.sync(XCL_BO_SYNC_BO_TO_DEVICE);        


        //=============================================================================================================================

     // define and allocate edges and space for the gcon4

    const int imagescale_gconv4 = 16; 
    const int num_vertex_gconv4 = imagescale_gconv4 * imagescale_gconv4;
    const int num_edge_block_gconv4 = imagescale_gconv4 * 4 * 9;
    const int num_vertex_bank_gconv4 = 16*4;
    const int f_block_len_custom_gconv4 = 3;
    const int BM_gconv4 = 18;

    xrt::bo boA_gconv4 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv4), krnl_feagg.group_id(3));
    xrt::bo boAX_gconv4_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 288), krnl_feagg.group_id(0));
    xrt::bo boAX_gconv4_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 288), krnl_feagg.group_id(1));
    xrt::bo boAX_gconv4_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 288), krnl_feagg.group_id(2));

    v_edge * boA_gconv4_map = boA_gconv4.map<v_edge *>();
    v_float * boAX_gconv4_map_bank1 = boAX_gconv4_bank1.map<v_float *>();
    v_float * boAX_gconv4_map_bank2 = boAX_gconv4_bank2.map<v_float *>();
    v_float * boAX_gconv4_map_bank3 = boAX_gconv4_bank3.map<v_float *>();

    create_edges_gconv4(boA_gconv4_map);
    boA_gconv4.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo boW_gconv4_self = xrt::bo(device, size_t(sizeof(float) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boAX_Merge_gconv4_bank1 = xrt::bo(device, size_t(sizeof(float) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), load_mm1.group_id(0));
    xrt::bo boAX_Merge_gconv4_bank2 = xrt::bo(device, size_t(sizeof(float) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), load_mm2.group_id(0));
    xrt::bo boAX_Merge_gconv4_bank3 = xrt::bo(device, size_t(sizeof(float) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), load_mm3.group_id(0));
    
    v_dt* boW_gconv4_map_self = boW_gconv4_self.map<v_dt*>();
    v_dt* boAX_Merge_gconv4_map_bank1 = boAX_Merge_gconv4_bank1.map<v_dt*>();
    v_dt* boAX_Merge_gconv4_map_bank2 = boAX_Merge_gconv4_bank2.map<v_dt*>();
    v_dt* boAX_Merge_gconv4_map_bank3 = boAX_Merge_gconv4_bank3.map<v_dt*>();

    xrt::bo boW_gconv4_neighbor = xrt::bo(device, size_t(sizeof(float) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boC_gconv4_bank1 = xrt::bo(device, size_t(sizeof(float) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), store_mm1.group_id(0));
    xrt::bo boC_gconv4_bank2 = xrt::bo(device, size_t(sizeof(float) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), store_mm2.group_id(0));
    xrt::bo boC_gconv4_bank3 = xrt::bo(device, size_t(sizeof(float) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), store_mm3.group_id(0));

    v_dt* boW_gconv4_neighbor_map = boW_gconv4_neighbor.map<v_dt*>();
    v_dt* boC_gconv4_map_bank1 = boC_gconv4_bank1.map<v_dt*>();
    v_dt* boC_gconv4_map_bank2 = boC_gconv4_bank2.map<v_dt*>();
    v_dt* boC_gconv4_map_bank3 = boC_gconv4_bank3.map<v_dt*>();

    load_gconv4(boW_gconv4_map_self, boW_gconv4_neighbor_map);

    boW_gconv4_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW_gconv4_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //==========================================================
    // allocate space for MLP
    xrt::bo weight_l1 = xrt::bo(device, size_t(sizeof(v_dt) * (64 * 256 * 3 + 4)), krnl_mlp.group_id(0));
    xrt::bo weight_l2 = xrt::bo(device, size_t(sizeof(v_dt) * (64 + 1)), krnl_mlp.group_id(1));
    xrt::bo result_holder = xrt::bo(device, size_t(sizeof(v_dt)), krnl_mlp.group_id(3));
    
    v_dt * weight_l1_map = weight_l1.map<v_dt *>();
    v_dt * weight_l2_map = weight_l2.map<v_dt *>();
    v_dt * result_holder_map = result_holder.map<v_dt *>();

    allocateMLP(weight_l1_map, weight_l2_map);
    
    weight_l1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    weight_l2.sync(XCL_BO_SYNC_BO_TO_DEVICE);


    //==========================================================================================================================
    // loading model weight
    //==========================================================================================================================

    auto load_gconv1_weight_self = krnl_trans(boW_self, BM, 0, 0, 0, 0, 1, 0); load_gconv1_weight_self.wait();
    auto load_gconv1_weight_neigbhor = krnl_trans(boW_neighbor, BM, 1, 1, 1, 0, 1, 1); load_gconv1_weight_neigbhor.wait();
    auto load_gconv2_weight_self = krnl_trans(boW_gconv2_self, BM, 0, 0, 0, 0, 1, 2); load_gconv2_weight_self.wait();
    auto load_gconv2_weight_neigbhor = krnl_trans(boW_gconv2_neighbor, BM, 1, 1, 1, 0, 1, 3); load_gconv2_weight_neigbhor.wait();
    auto load_gconv3_weight_self = krnl_trans(boW_gconv3_self, BM, 0, 0, 0, 0, 1, 4); load_gconv3_weight_self.wait();
    auto load_gconv3_weight_neigbhor = krnl_trans(boW_gconv3_neighbor, BM, 1, 1, 1, 0, 1, 5); load_gconv3_weight_neigbhor.wait();
    auto load_gconv4_weight_self = krnl_trans(boW_gconv4_self, BM, 0, 0, 0, 0, 1, 6); load_gconv4_weight_self.wait();
    auto load_gconv4_weight_neigbhor = krnl_trans(boW_gconv4_neighbor, BM, 1, 1, 1, 0, 1, 7); load_gconv4_weight_neigbhor.wait();

    auto load_weight_run = xrt::run(krnl_mlp); 
    load_weight_run.set_arg(0, weight_l1);
    load_weight_run.set_arg(1, weight_l2);
    load_weight_run.set_arg(2, 1);
    load_weight_run.set_arg(3, result_holder);
    load_weight_run.start();
    load_weight_run.wait();


    //============================================================================================================================
    // start execution
    //=============================================================================================================================

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of self transformation + fea agg\n";
    auto run_p1 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p1, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2, 
        f_block_len_custom, imagescale, 
        1, 0, 0, 0, num_vertex_bank, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        );  
    
    auto run_p2 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p2, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 

    // std::cout << "Finish the execution of feature aggregation\n";
    // auto finish = std::chrono::high_resolution_clock::now();
    // std::cout << "hardware multiplication took "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
    //           << " microseconds\n";

    // start = std::chrono::high_resolution_clock::now();
    // std::cout << "Execution of the self transformation\n";

    auto run_self_trans = krnl_trans(
        boW_self,
        BM, 0, 0, 0, 0,
        2, 0 // new parameter v11
        );
    auto run_self_trans_Load1 = load_mm1(boX_bank1, BM);
    auto run_self_trans_Load2 = load_mm2(boX_bank2, BM);
    auto run_self_trans_Load3 = load_mm3(boX_bank3, BM);
    auto run_self_trans_Store1 = store_mm1(boAX_Merge_bank1, BM);
    auto run_self_trans_Store2 = store_mm2(boAX_Merge_bank2, BM);
    auto run_self_trans_Store3 = store_mm3(boAX_Merge_bank3, BM);
    
    run_p2.wait();
    run_p1.wait();
    run_self_trans_Load1.wait();
    run_self_trans_Load2.wait();
    run_self_trans_Load3.wait();
    run_self_trans_Store1.wait();
    run_self_trans_Store2.wait();
    run_self_trans_Store3.wait();
    run_self_trans.wait();

    std::cout << "Finish the execution of self transformation + fea agg\n";
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the neighbor transformation\n";\

    auto run_loaddata = krnl_feagg(
        boAX_bank1, boAX_bank2, boAX_bank3, 
        (v_edge *) 0, 
        (v_float *) 0, (v_float *) 0, (v_float *) 0, 
        0, 0,
        0, 0,
        0, 0, 0, 0, 0, 0,
        2, BM, 1, // v8 new parameters
        boAX_Merge_bank1, boAX_Merge_bank2, boAX_Merge_bank3 // v8 new parameters
    ); 

    auto run_neigh_trans = krnl_trans(
        boW_neighbor,
        BM, 1, 1, 1, 0, 
        2, 1 // new parameter v11
        );

    auto run_neigh_trans_Load1 = load_mm1(boAX_bank1, BM);
    auto run_neigh_trans_Load2 = load_mm2(boAX_bank2, BM);
    auto run_neigh_trans_Load3 = load_mm3(boAX_bank3, BM);
    auto run_neigh_trans_Store1 = store_mm1(boC_bank1, BM);
    auto run_neigh_trans_Store2 = store_mm2(boC_bank2, BM);
    auto run_neigh_trans_Store3 = store_mm3(boC_bank3, BM);

    run_neigh_trans_Load1.wait();
    run_neigh_trans_Load2.wait();
    run_neigh_trans_Load3.wait();
    run_neigh_trans_Store1.wait();
    run_neigh_trans_Store2.wait();
    run_neigh_trans_Store3.wait();

    run_loaddata.wait();
    run_neigh_trans.wait();
    std::cout << "Finish the execution of neighbor transformation\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    // execute the pooling 1
    // =============================
    // =============================

    start = std::chrono::high_resolution_clock::now();
    std::cout << "[event] Execution of the pooling layer 1\n";
    auto run_poo1_part1 = krnl_feagg(
        boC_bank1, boC_bank2, boC_bank3, boA_pool1, 
        boC_pooling1_bank1, boC_pooling1_bank2, boC_pooling1_bank3, 
        num_vertex_bank, num_pool1_edges, 
        3, imagescale, 
        1, 1, 0, 0, num_vertex_pooling1/8, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        );  
    auto run_poo1_part2 = krnl_feagg(
        boC_bank1, boC_bank2, boC_bank3, boA_pool1, 
        boC_pooling1_bank1, boC_pooling1_bank2, boC_pooling1_bank3, 
        num_vertex_bank, num_pool1_edges,
        3, imagescale,
        1, 1, num_vertex_bank*4, num_vertex_pooling1/2, num_vertex_pooling1/8, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        );  
    run_poo1_part1.wait();
    run_poo1_part2.wait();
    std::cout << "[event] Finish the execution of the pooling layer 1\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";
       // synchronization for the results  ======================================================================

    // execute the aggregate for gconv2
    // =============================
    // =============================

    start = std::chrono::high_resolution_clock::now();
    std::cout << "[event] Execution of the gconv2\n";
    auto run_gconv2_agg = krnl_feagg(
        boC_pooling1_bank1, boC_pooling1_bank2, boC_pooling1_bank3, boA_gconv2, 
        boAX_gconv2_bank1, boAX_gconv2_bank2, boAX_gconv2_bank3, 
        num_vertex_bank_gconv2, num_edge_block_gconv2, 
        f_block_len_custom_gconv2, imagescale_gconv2, 
        1, 1, 0, 0, num_vertex_bank_gconv2, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    
    auto run_self_trans_gconv2 = krnl_trans(
        boW_gconv2_self,
        BM_gconv2, 0, 0, 0, 0,
        2, 2 // new parameter v11
        );
    auto run_self_trans_Load1_gconv2 = load_mm1(boC_pooling1_bank1, BM_gconv2);
    auto run_self_trans_Load2_gconv2 = load_mm2(boC_pooling1_bank2, BM_gconv2);
    auto run_self_trans_Load3_gconv2 = load_mm3(boC_pooling1_bank3, BM_gconv2);
    auto run_self_trans_Store1_gconv2 = store_mm1(boAX_Merge_gconv2_bank1, BM_gconv2);
    auto run_self_trans_Store2_gconv2 = store_mm2(boAX_Merge_gconv2_bank2, BM_gconv2);
    auto run_self_trans_Store3_gconv2 = store_mm3(boAX_Merge_gconv2_bank3, BM_gconv2);
    
    run_self_trans_Load1_gconv2.wait();
    run_self_trans_Load2_gconv2.wait();
    run_self_trans_Load3_gconv2.wait();
    run_self_trans_Store1_gconv2.wait();
    run_self_trans_Store2_gconv2.wait();
    run_self_trans_Store3_gconv2.wait();
    run_gconv2_agg.wait();
    run_self_trans_gconv2.wait();

    auto run_loaddata_gconv2 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p2, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        2, BM_gconv2, 1, // v8 new parameters
        boAX_Merge_gconv2_bank1, boAX_Merge_gconv2_bank2,  boAX_Merge_gconv2_bank3  // v8 new parameters
    ); 
    auto run_neigh_trans_gconv2 = krnl_trans(
        boW_gconv2_neighbor,
        BM_gconv2, 1, 1, 1, 0,
        2, 3 // new parameter v11
        );

    auto run_neigh_trans_Load1_gconv2 = load_mm1(boAX_gconv2_bank1, BM_gconv2);
    auto run_neigh_trans_Load2_gconv2 = load_mm2(boAX_gconv2_bank2, BM_gconv2);
    auto run_neigh_trans_Load3_gconv2 = load_mm3(boAX_gconv2_bank3, BM_gconv2);
    auto run_neigh_trans_Store1_gconv2 = store_mm1(boC_gconv2_bank1, BM_gconv2);
    auto run_neigh_trans_Store2_gconv2 = store_mm2(boC_gconv2_bank2, BM_gconv2);
    auto run_neigh_trans_Store3_gconv2 = store_mm3(boC_gconv2_bank3, BM_gconv2);

    run_neigh_trans_Load1_gconv2.wait();
    run_neigh_trans_Load2_gconv2.wait();
    run_neigh_trans_Load3_gconv2.wait();
    run_neigh_trans_Store1_gconv2.wait();
    run_neigh_trans_Store2_gconv2.wait();
    run_neigh_trans_Store3_gconv2.wait();
    run_loaddata_gconv2.wait();
    run_neigh_trans_gconv2.wait();
    std::cout << "[event] Finish the execution of ntransform for gconv2\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    // execute the pooling 2
    // =============================
    // =============================

    start = std::chrono::high_resolution_clock::now();
    std::cout << "[event] Execution of the pooling layer 2\n";
    auto run_poo2 = krnl_feagg(
        boC_gconv2_bank1, boC_gconv2_bank2, boC_gconv2_bank3, boA_pool2, 
        boC_pooling2_bank1, boC_pooling2_bank2, boC_pooling2_bank3, 
        num_vertex_pooling1/4, num_pool2_edges, 
        3, imagescale_gconv2, 
        1, 1, 0, 0, num_vertex_pooling2/4, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    run_poo2.wait();
    std::cout << "[event] Finish the execution of the pooling layer 2\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

        // =====================================================================================================================
    // execute the aggregate for gconv3
    // =============================
    // =============================

    start = std::chrono::high_resolution_clock::now();
    std::cout << "[event] Execution of the gconv3\n";
    auto run_gconv3_agg = krnl_feagg(
        boC_pooling2_bank1, boC_pooling2_bank2, boC_pooling2_bank3, boA_gconv3, 
        boAX_gconv3_bank1, boAX_gconv3_bank2, boAX_gconv3_bank3, 
        num_vertex_bank_gconv3, num_edge_block_gconv3, 
        f_block_len_custom_gconv3, imagescale_gconv3, 
        1, 1, 0, 0, num_vertex_bank_gconv3, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    auto run_self_trans_gconv3 = krnl_trans(
        boW_gconv3_self, 
        BM_gconv3, 0, 0, 0, 0,
        2, 4 // new parameter v11
        );
    auto run_self_trans_Load1_gconv3 = load_mm1(boC_pooling2_bank1, BM_gconv3);
    auto run_self_trans_Load2_gconv3 = load_mm2(boC_pooling2_bank2, BM_gconv3);
    auto run_self_trans_Load3_gconv3 = load_mm3(boC_pooling2_bank3, BM_gconv3);
    auto run_self_trans_Store1_gconv3 = store_mm1(boAX_Merge_gconv3_bank1, BM_gconv3);
    auto run_self_trans_Store2_gconv3 = store_mm2(boAX_Merge_gconv3_bank2, BM_gconv3);
    auto run_self_trans_Store3_gconv3 = store_mm3(boAX_Merge_gconv3_bank3, BM_gconv3);

    run_self_trans_Load1_gconv3.wait();
    run_self_trans_Load2_gconv3.wait();
    run_self_trans_Load3_gconv3.wait();
    run_self_trans_Store1_gconv3.wait();
    run_self_trans_Store2_gconv3.wait();
    run_self_trans_Store3_gconv3.wait();
    run_gconv3_agg.wait();
    run_self_trans_gconv3.wait();

    auto run_loaddata_gconv3 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p2, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        2, BM_gconv3, 1, // v8 new parameters
        boAX_Merge_gconv3_bank1, boAX_Merge_gconv3_bank2,  boAX_Merge_gconv3_bank3  // v8 new parameters
    ); 
    auto run_neigh_trans_gconv3 = krnl_trans(
        boW_gconv3_neighbor,
        BM_gconv3, 1, 1, 1, 0,
        2, 5 // new parameter v11
        );

    auto run_neigh_trans_Load1_gconv3 = load_mm1(boAX_gconv3_bank1, BM_gconv3);
    auto run_neigh_trans_Load2_gconv3 = load_mm2(boAX_gconv3_bank2, BM_gconv3);
    auto run_neigh_trans_Load3_gconv3 = load_mm3(boAX_gconv3_bank3, BM_gconv3);
    auto run_neigh_trans_Store1_gconv3 = store_mm1(boC_gconv3_bank1, BM_gconv3);
    auto run_neigh_trans_Store2_gconv3 = store_mm2(boC_gconv3_bank2, BM_gconv3);
    auto run_neigh_trans_Store3_gconv3 = store_mm3(boC_gconv3_bank3, BM_gconv3);

    run_neigh_trans_Load1_gconv3.wait();
    run_neigh_trans_Load2_gconv3.wait();
    run_neigh_trans_Load3_gconv3.wait();
    run_neigh_trans_Store1_gconv3.wait();
    run_neigh_trans_Store2_gconv3.wait();
    run_neigh_trans_Store3_gconv3.wait();
    run_loaddata_gconv3.wait();
    run_neigh_trans_gconv3.wait();
    std::cout << "Finish the execution of  gconv3\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    // execute the pooling 3
    // =============================
    // =============================

    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the pooling layer 3\n";
    auto run_poo3 = krnl_feagg(
        boC_gconv3_bank1, boC_gconv3_bank2, boC_gconv3_bank3, boA_pool3, 
        boC_pooling3_bank1, boC_pooling3_bank2, boC_pooling3_bank3, 
        num_vertex_pooling2/4, num_pool3_edges, 
        3, imagescale_gconv3, 
        1, 1, 0, 0, num_vertex_pooling3/4, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    run_poo3.wait();
    std::cout << "Finish the execution of the pooling layer 3\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

       // execute the aggregate for gconv4
    // =============================
    // =============================

    start = std::chrono::high_resolution_clock::now();
    std::cout << "Execution of the gconv4\n";
    auto run_gconv4_agg = krnl_feagg(
        boC_pooling3_bank1, boC_pooling3_bank2, boC_pooling3_bank3, boA_gconv4, 
        boAX_gconv4_bank1, boAX_gconv4_bank2, boAX_gconv4_bank3, 
        num_vertex_bank_gconv4, num_edge_block_gconv4, 
        f_block_len_custom_gconv4, imagescale_gconv4, 
        1, 1, 0, 0, num_vertex_bank_gconv4, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    auto run_self_trans_gconv4 = krnl_trans(
        boW_gconv4_self, 
        BM_gconv4, 0, 0, 0, 0,
        2, 6 // new parameter v11
        );

    auto run_self_trans_Load1_gconv4 = load_mm1(boC_pooling3_bank1, BM_gconv4);
    auto run_self_trans_Load2_gconv4 = load_mm2(boC_pooling3_bank2, BM_gconv4);
    auto run_self_trans_Load3_gconv4 = load_mm3(boC_pooling3_bank3, BM_gconv4);
    auto run_self_trans_Store1_gconv4 = store_mm1(boAX_Merge_gconv4_bank1, BM_gconv4);
    auto run_self_trans_Store2_gconv4 = store_mm2(boAX_Merge_gconv4_bank2, BM_gconv4);
    auto run_self_trans_Store3_gconv4 = store_mm3(boAX_Merge_gconv4_bank3, BM_gconv4);

    run_self_trans_Load1_gconv4.wait();
    run_self_trans_Load2_gconv4.wait();
    run_self_trans_Load3_gconv4.wait();
    run_self_trans_Store1_gconv4.wait();
    run_self_trans_Store2_gconv4.wait();
    run_self_trans_Store3_gconv4.wait();
    run_gconv4_agg.wait();
    run_self_trans_gconv4.wait();
    
    auto run_loaddata_gconv4 = krnl_feagg(
        boX_bank1, boX_bank2, boX_bank3, 
        boA_p2, 
        boAX_bank1, boAX_bank2, boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        2, BM_gconv4, 1, // v8 new parameters
        boAX_Merge_gconv4_bank1, boAX_Merge_gconv4_bank2,  boAX_Merge_gconv4_bank3 // v8 new parameters
    ); 
    auto run_neigh_trans_gconv4 = krnl_trans(
        boW_gconv4_neighbor,
        BM_gconv4, 1, 1, 1, 1,
        2, 7 // new parameter v11
        );

 

    auto run_neigh_trans_Load1_gconv4 = load_mm1(boAX_gconv4_bank1, BM_gconv4);
    auto run_neigh_trans_Load2_gconv4 = load_mm2(boAX_gconv4_bank2, BM_gconv4);
    auto run_neigh_trans_Load3_gconv4 = load_mm3(boAX_gconv4_bank3, BM_gconv4);
    auto mlp_run = xrt::run(krnl_mlp); 
    mlp_run.set_arg(0, weight_l1);
    mlp_run.set_arg(1, weight_l2);
    mlp_run.set_arg(2, 0);
    mlp_run.set_arg(3, result_holder);
    mlp_run.start(); 


    run_neigh_trans_Load1_gconv4.wait();
    run_neigh_trans_Load2_gconv4.wait();
    run_neigh_trans_Load3_gconv4.wait();
    run_loaddata_gconv4.wait();
    run_neigh_trans_gconv4.wait();
    mlp_run.wait();

    std::cout << "Finish the execution of gconv4\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";


    // boAX_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boAX_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boAX_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boAX_Merge_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boAX_Merge_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boAX_Merge_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling1_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling1_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling1_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv2_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv2_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv2_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling2_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling2_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling2_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv3_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv3_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv3_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling3_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling3_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_pooling3_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv4_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv4_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // boC_gconv4_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    result_holder.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // float perdata = 0;

    // for(int i = 0; i < num_vertex; i++){
    //     for(int f = 0; f < 3; f++){
    //         for(int j = 0; j < VDATA_SIZE; j++){
    //             if(f == 0) perdata = boAX_Merge_map_bank1[i].data[j]; //boAX_Merge_map_bank1 boAX_map_bank1
    //             if(f == 1) perdata = boAX_Merge_map_bank1[i].data[j];
    //             if(f == 2) perdata = boAX_Merge_map_bank1[i].data[j];
    //             printf("perdata: %f\n", perdata);
    //         }
    //     }
    // }


    //verify_gconv1(boC_map_bank1, boC_map_bank2, boC_map_bank3);
    // verify_pooling1(boC_pooling1_bank1_map, boC_pooling1_bank2_map, boC_pooling1_bank3_map);
    //verify_gconv2(boC_gconv2_map_bank1, boC_gconv2_map_bank2, boC_gconv2_map_bank3);
    //verify_pooling2(boC_pooling2_bank1_map, boC_pooling2_bank2_map, boC_pooling2_bank3_map);
    //verify_gconv3(boC_gconv3_map_bank1, boC_gconv3_map_bank2, boC_gconv3_map_bank3);
    //verify_pooling3(boC_pooling3_bank1_map, boC_pooling3_bank2_map, boC_pooling3_bank3_map);
    //verify_gconv4(boC_gconv4_map_bank1, boC_gconv4_map_bank2, boC_gconv4_map_bank3);
    std::ifstream fin_After_fc2("../../Layer/data/intermediate/After_fc2.bin", std::ios::binary);
    float fdata_f;
    for(int i = 0; i < 10; i++){
        fin_After_fc2.read(reinterpret_cast<char*>(&fdata_f), sizeof(float));
        printf("golden %f, myresult %f\n", fdata_f, result_holder_map[0].data[i]);
    }

}