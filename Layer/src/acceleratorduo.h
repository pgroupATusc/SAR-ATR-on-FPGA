//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <datatype.h>
#include <utils2.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

// XRT includes
#include "xrt/xrt_bo.h"
#include <experimental/xrt_xclbin.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef _ACC
#define _ACC

class accelerator{
private:
    xrt::device device;


    // accelerator copy 1
    xrt::kernel cp1_krnl_feagg; // feature aggregation kernel
    xrt::kernel cp1_krnl_trans; // feature transformation kernel
    xrt::kernel cp1_load_mm1; // data loader 1 for transformation kernel
    xrt::kernel cp1_load_mm2; // data loader 2 for transformation kernel
    xrt::kernel cp1_load_mm3; // data loader 3 for transformation kernel
    xrt::kernel cp1_store_mm1; // data storer 1 for transformation kernel
    xrt::kernel cp1_store_mm2; // data storer 2 for transformation kernel
    xrt::kernel cp1_store_mm3; // data storer 3 for transformation kernel
    xrt::kernel cp1_krnl_mlp; // mlp kernel
    
    // memory for gconv1
    int imagescale; 
    int num_vertex;
    int num_edge_block;
    int num_vertex_bank;
    int f_block_len_custom;
    int BM;
    xrt::bo cp1_boX_bank1;
    xrt::bo cp1_boX_bank2;
    xrt::bo cp1_boX_bank3;
    xrt::bo cp1_boA_p1;
    xrt::bo cp1_boA_p2;
    xrt::bo cp1_boAX_bank1;
    xrt::bo cp1_boAX_bank2;
    xrt::bo cp1_boAX_bank3;
    xrt::bo cp1_boW_self;
    xrt::bo cp1_boAX_Merge_bank1;
    xrt::bo cp1_boAX_Merge_bank2; 
    xrt::bo cp1_boAX_Merge_bank3; 
    xrt::bo cp1_boW_neighbor;
    xrt::bo cp1_boC_bank1;
    xrt::bo cp1_boC_bank2;
    xrt::bo cp1_boC_bank3;

    // memory for pooling 1
    int num_pool1_edges;
    int BM_pooling1;
    int num_vertex_pooling1;
    xrt::bo cp1_boA_pool1;
    xrt::bo cp1_boC_pooling1_bank1;
    xrt::bo cp1_boC_pooling1_bank2;
    xrt::bo cp1_boC_pooling1_bank3;

    // memory for gconv2
    int imagescale_gconv2; 
    int num_vertex_gconv2;
    int num_edge_block_gconv2;
    int num_vertex_bank_gconv2;
    int f_block_len_custom_gconv2;
    int BM_gconv2;
    xrt::bo cp1_boA_gconv2;
    xrt::bo cp1_boAX_gconv2_bank1;
    xrt::bo cp1_boAX_gconv2_bank2;
    xrt::bo cp1_boAX_gconv2_bank3;
    xrt::bo cp1_boW_gconv2_self;
    xrt::bo cp1_boAX_Merge_gconv2_bank1;
    xrt::bo cp1_boAX_Merge_gconv2_bank2;
    xrt::bo cp1_boAX_Merge_gconv2_bank3;
    xrt::bo cp1_boW_gconv2_neighbor;
    xrt::bo cp1_boC_gconv2_bank1;
    xrt::bo cp1_boC_gconv2_bank2;
    xrt::bo cp1_boC_gconv2_bank3;

    // memory for pooling 2
    int num_pool2_edges;
    int BM_pooling2;
    int num_vertex_pooling2;
    xrt::bo cp1_boA_pool2;
    xrt::bo cp1_boC_pooling2_bank1;
    xrt::bo cp1_boC_pooling2_bank2;
    xrt::bo cp1_boC_pooling2_bank3;

    // memory for gconv 3
    int imagescale_gconv3; 
    int num_vertex_gconv3;
    int num_edge_block_gconv3;
    int num_vertex_bank_gconv3;
    int f_block_len_custom_gconv3;
    int BM_gconv3;
    xrt::bo cp1_boA_gconv3;
    xrt::bo cp1_boAX_gconv3_bank1;
    xrt::bo cp1_boAX_gconv3_bank2;
    xrt::bo cp1_boAX_gconv3_bank3;
    xrt::bo cp1_boW_gconv3_self;
    xrt::bo cp1_boAX_Merge_gconv3_bank1;
    xrt::bo cp1_boAX_Merge_gconv3_bank2;
    xrt::bo cp1_boAX_Merge_gconv3_bank3;
    xrt::bo cp1_boW_gconv3_neighbor;
    xrt::bo cp1_boC_gconv3_bank1;
    xrt::bo cp1_boC_gconv3_bank2;
    xrt::bo cp1_boC_gconv3_bank3;

    // memory for pooling 3
    int num_pool3_edges;
    int BM_pooling3;
    int num_vertex_pooling3;
    xrt::bo cp1_boA_pool3;
    xrt::bo cp1_boC_pooling3_bank1;
    xrt::bo cp1_boC_pooling3_bank2;
    xrt::bo cp1_boC_pooling3_bank3;

    // memory for gconv 4
    int imagescale_gconv4; 
    int num_vertex_gconv4;
    int num_edge_block_gconv4;
    int num_vertex_bank_gconv4;
    int f_block_len_custom_gconv4;
    int BM_gconv4;
    xrt::bo cp1_boA_gconv4;
    xrt::bo cp1_boAX_gconv4_bank1;
    xrt::bo cp1_boAX_gconv4_bank2;
    xrt::bo cp1_boAX_gconv4_bank3;
    xrt::bo cp1_boW_gconv4_self;
    xrt::bo cp1_boAX_Merge_gconv4_bank1;
    xrt::bo cp1_boAX_Merge_gconv4_bank2;
    xrt::bo cp1_boAX_Merge_gconv4_bank3;
    xrt::bo cp1_boW_gconv4_neighbor;
    xrt::bo cp1_boC_gconv4_bank1;
    xrt::bo cp1_boC_gconv4_bank2;
    xrt::bo cp1_boC_gconv4_bank3;

    // memory for mlp
    xrt::bo cp1_weight_l1;
    xrt::bo cp1_weight_l2;
    xrt::bo cp1_result_holder;
    
    // input on host side
    v_float * cp1_boX_map_bank1;
    v_float * cp1_boX_map_bank2;
    v_float * cp1_boX_map_bank3;

    // define the accelerator 2 ==============================================================================================================================================//
     // accelerator copy 2
    xrt::kernel cp2_krnl_feagg; // feature aggregation kernel
    xrt::kernel cp2_krnl_trans; // feature transformation kernel
    xrt::kernel cp2_load_mm1; // data loader 1 for transformation kernel
    xrt::kernel cp2_load_mm2; // data loader 2 for transformation kernel
    xrt::kernel cp2_load_mm3; // data loader 3 for transformation kernel
    xrt::kernel cp2_store_mm1; // data storer 1 for transformation kernel
    xrt::kernel cp2_store_mm2; // data storer 2 for transformation kernel
    xrt::kernel cp2_store_mm3; // data storer 3 for transformation kernel
    xrt::kernel cp2_krnl_mlp; // mlp kernel
    
    // memory for gconv1 copy 2
    xrt::bo cp2_boX_bank1;
    xrt::bo cp2_boX_bank2;
    xrt::bo cp2_boX_bank3;
    xrt::bo cp2_boA_p1;
    xrt::bo cp2_boA_p2;
    xrt::bo cp2_boAX_bank1;
    xrt::bo cp2_boAX_bank2;
    xrt::bo cp2_boAX_bank3;
    xrt::bo cp2_boW_self;
    xrt::bo cp2_boAX_Merge_bank1;
    xrt::bo cp2_boAX_Merge_bank2; 
    xrt::bo cp2_boAX_Merge_bank3; 
    xrt::bo cp2_boW_neighbor;
    xrt::bo cp2_boC_bank1;
    xrt::bo cp2_boC_bank2;
    xrt::bo cp2_boC_bank3;

    // memory for pooling 1 copy 2
    xrt::bo cp2_boA_pool1;
    xrt::bo cp2_boC_pooling1_bank1;
    xrt::bo cp2_boC_pooling1_bank2;
    xrt::bo cp2_boC_pooling1_bank3;

    // memory for gconv2 copy 2
    xrt::bo cp2_boA_gconv2;
    xrt::bo cp2_boAX_gconv2_bank1;
    xrt::bo cp2_boAX_gconv2_bank2;
    xrt::bo cp2_boAX_gconv2_bank3;
    xrt::bo cp2_boW_gconv2_self;
    xrt::bo cp2_boAX_Merge_gconv2_bank1;
    xrt::bo cp2_boAX_Merge_gconv2_bank2;
    xrt::bo cp2_boAX_Merge_gconv2_bank3;
    xrt::bo cp2_boW_gconv2_neighbor;
    xrt::bo cp2_boC_gconv2_bank1;
    xrt::bo cp2_boC_gconv2_bank2;
    xrt::bo cp2_boC_gconv2_bank3;

    // memory for pooling 2 copy 2
    xrt::bo cp2_boA_pool2;
    xrt::bo cp2_boC_pooling2_bank1;
    xrt::bo cp2_boC_pooling2_bank2;
    xrt::bo cp2_boC_pooling2_bank3;

    // memory for gconv 3 copy 2
    xrt::bo cp2_boA_gconv3;
    xrt::bo cp2_boAX_gconv3_bank1;
    xrt::bo cp2_boAX_gconv3_bank2;
    xrt::bo cp2_boAX_gconv3_bank3;
    xrt::bo cp2_boW_gconv3_self;
    xrt::bo cp2_boAX_Merge_gconv3_bank1;
    xrt::bo cp2_boAX_Merge_gconv3_bank2;
    xrt::bo cp2_boAX_Merge_gconv3_bank3;
    xrt::bo cp2_boW_gconv3_neighbor;
    xrt::bo cp2_boC_gconv3_bank1;
    xrt::bo cp2_boC_gconv3_bank2;
    xrt::bo cp2_boC_gconv3_bank3;

    // memory for pooling 3 copy 2
    xrt::bo cp2_boA_pool3;
    xrt::bo cp2_boC_pooling3_bank1;
    xrt::bo cp2_boC_pooling3_bank2;
    xrt::bo cp2_boC_pooling3_bank3;

    // memory for gconv 4 copy 2
    xrt::bo cp2_boA_gconv4;
    xrt::bo cp2_boAX_gconv4_bank1;
    xrt::bo cp2_boAX_gconv4_bank2;
    xrt::bo cp2_boAX_gconv4_bank3;
    xrt::bo cp2_boW_gconv4_self;
    xrt::bo cp2_boAX_Merge_gconv4_bank1;
    xrt::bo cp2_boAX_Merge_gconv4_bank2;
    xrt::bo cp2_boAX_Merge_gconv4_bank3;
    xrt::bo cp2_boW_gconv4_neighbor;
    xrt::bo cp2_boC_gconv4_bank1;
    xrt::bo cp2_boC_gconv4_bank2;
    xrt::bo cp2_boC_gconv4_bank3;

    // memory for mlp copy 2
    xrt::bo cp2_weight_l1;
    xrt::bo cp2_weight_l2;
    xrt::bo cp2_result_holder;
    
    // input on host side copy 2
    v_float * cp2_boX_map_bank1;
    v_float * cp2_boX_map_bank2;
    v_float * cp2_boX_map_bank3;

public:
    accelerator(){}
    accelerator(xrt::device & device, xrt::uuid & uuid, int copy);
    void preparation();
    void loadweight();
    void inference();
    void loadinput(std::string filename);
    void checkresult(std::string filename);
};

#endif