#include <iostream>
#include <cstring>
#include <datatype.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>

// XRT includes
#include "xrt/xrt_bo.h"
#include <experimental/xrt_xclbin.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"


void load_input(
    v_float * input_map_bank1, 
    v_float * input_map_bank2, 
    v_float * input_map_bank3,
    int num_vertex
);

void load_input_general(
    std::string filename,
    v_float * input_map_bank1, 
    v_float * input_map_bank2, 
    v_float * input_map_bank3,
    int num_vertex
);

void create_edges_layer1(
    v_edge * boA_p1_map,
    v_edge * boA_p2_map
);

void load_gconv1(
    v_dt * boW_map_self, 
    v_dt * boW_neighbor_map
);

void layerone_load_gconv1(
    v_dt * boW_map_self, 
    v_dt * boW_neighbor_map
);

void verify_gconv1(
    v_dt* boC_map_bank1,
    v_dt* boC_map_bank2,
    v_dt* boC_map_bank3
);

void create_edges_pooling1(
    v_edge * boA_pool1_map
);


void verify_pooling1(
    v_float* boC_pooling1_bank1_map,
    v_float* boC_pooling1_bank2_map,
    v_float* boC_pooling1_bank3_map
);

void create_edges_gconv2(
    v_edge * boA_gconv2_map
);

void load_gconv2(
    v_dt* boW_gconv2_map_self, 
    v_dt* boW_gconv2_neighbor_map
);

void verify_gconv2(
    v_dt* boC_gconv2_map_bank1, 
    v_dt* boC_gconv2_map_bank2, 
    v_dt* boC_gconv2_map_bank3
);

void create_edges_pooling2(
    v_edge * boA_pool2_map
);

void verify_pooling2(
    v_float* boC_pooling2_bank1_map, 
    v_float* boC_pooling2_bank2_map, 
    v_float* boC_pooling2_bank3_map
);


void create_edges_gconv3(
    v_edge * boA_gconv3_map
);

void create_edges_gconv4(
    v_edge * boA_gconv4_map
);

void load_gconv3(
    v_dt* boW_gconv3_map_self, 
    v_dt* boW_gconv3_neighbor_map
);

void load_gconv4(
    v_dt* boW_gconv4_map_self, 
    v_dt* boW_gconv4_neighbor_map
);


void verify_gconv3(
    v_dt* boC_gconv3_map_bank1, 
    v_dt* boC_gconv3_map_bank2, 
    v_dt* boC_gconv3_map_bank3
);


void create_edges_pooling3(
    v_edge *boA_pool3_map
);

void verify_pooling3(
    v_float* boC_pooling3_bank1_map, 
    v_float* boC_pooling3_bank2_map, 
    v_float* boC_pooling3_bank3_map
);

void verify_gconv4(
    v_dt* boC_gconv4_map_bank1, 
    v_dt*boC_gconv4_map_bank2, 
    v_dt* boC_gconv4_map_bank3
);

void allocateMLP(
    v_dt* weight_l1_map, 
    v_dt* weight_l2_map
);