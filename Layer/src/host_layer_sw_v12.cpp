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
    auto load_mm1 = xrt::kernel(device, uuid, "read_AX_fromDDR_bankDouble:{read_AX_fromDDR_bank_1}", xrt::kernel::cu_access_mode::exclusive);
    auto load_mm2 = xrt::kernel(device, uuid, "read_AX_fromDDR_bankSingle:{read_AX_fromDDR_bank_2}", xrt::kernel::cu_access_mode::exclusive);
    auto load_mm3 = xrt::kernel(device, uuid, "read_AX_fromDDR_bankSingle:{read_AX_fromDDR_bank_3}", xrt::kernel::cu_access_mode::exclusive);
    auto store_mm1 = xrt::kernel(device, uuid, "store_output_matrix:{store_output_matrix_1}", xrt::kernel::cu_access_mode::exclusive);
    auto store_mm2 = xrt::kernel(device, uuid, "store_output_matrix:{store_output_matrix_2}", xrt::kernel::cu_access_mode::exclusive);
    auto store_mm3 = xrt::kernel(device, uuid, "store_output_matrix:{store_output_matrix_3}", xrt::kernel::cu_access_mode::exclusive);
    auto krnl_mlp = xrt::kernel(device, uuid, "mlp:{mlp_1}");
    auto krnl_layerone = xrt::kernel(device, uuid, "layerone:{layerone_1}");
    
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
    xrt::bo boW_self = xrt::bo(device, size_t(sizeof(float) * 2 * BLOCK_NUM * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boAX_Merge_bank1 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), load_mm1.group_id(0)); //
    xrt::bo boAX_Merge_bank2 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), load_mm2.group_id(0)); //
    xrt::bo boAX_Merge_bank3 = xrt::bo(device, size_t(sizeof(float) * BM * VDATA_SIZE * VDATA_SIZE), load_mm3.group_id(0)); //
    
    v_dt* boW_map_self = boW_self.map<v_dt*>();
    v_dt* boAX_Merge_map_bank1 = boAX_Merge_bank1.map<v_dt*>();
    v_dt* boAX_Merge_map_bank2 = boAX_Merge_bank2.map<v_dt*>();
    v_dt* boAX_Merge_map_bank3 = boAX_Merge_bank3.map<v_dt*>();
     // allocate space for neighbor transformation

    xrt::bo boW_neighbor = xrt::bo(device, size_t(sizeof(float) * 2 * BLOCK_NUM  * VDATA_SIZE), krnl_trans.group_id(0));
    xrt::bo boC_bank1 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), store_mm1.group_id(0));
    xrt::bo boC_bank2 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), store_mm2.group_id(0));
    xrt::bo boC_bank3 = xrt::bo(device, size_t(sizeof(float) * BM  * VDATA_SIZE * VDATA_SIZE), store_mm3.group_id(0));

    v_dt* boW_neighbor_map = boW_neighbor.map<v_dt*>();
    v_dt* boC_map_bank1 = boC_bank1.map<v_dt*>();
    v_dt* boC_map_bank2 = boC_bank2.map<v_dt*>();
    v_dt* boC_map_bank3 = boC_bank3.map<v_dt*>();

    layerone_load_gconv1(boW_map_self, boW_neighbor_map);

    boW_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boW_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //==========================================================================================================================
    // loading model weight
    //==========================================================================================================================

    auto run_load_weight_layerone_self = krnl_layerone(
        boW_self,
        BM, 
        1, // merge
        1, // ReLu
        0, // biasflag
        1, // mode
        0  // layerSelect
    ); //DATA_SIZE=size
    run_load_weight_layerone_self.wait();

    auto run_load_weight_layerone_neighbor = krnl_layerone(
        boW_neighbor,
        BM, 
        1, // merge
        1, // ReLu
        1, // biasflag
        1, // mode
        1  // layerSelect
    ); //DATA_SIZE=size
    run_load_weight_layerone_neighbor.wait();

    // auto load_gconv1_weight_self = krnl_trans(boW_self, BM, 0, 0, 0, 0, 1, 0); load_gconv1_weight_self.wait();
    // auto load_gconv1_weight_neigbhor = krnl_trans(boW_neighbor, BM, 1, 1, 1, 0, 1, 1); load_gconv1_weight_neigbhor.wait();
    // auto load_gconv2_weight_self = krnl_trans(boW_gconv2_self, BM, 0, 0, 0, 0, 1, 2); load_gconv2_weight_self.wait();
    // auto load_gconv2_weight_neigbhor = krnl_trans(boW_gconv2_neighbor, BM, 1, 1, 1, 0, 1, 3); load_gconv2_weight_neigbhor.wait();
    // auto load_gconv3_weight_self = krnl_trans(boW_gconv3_self, BM, 0, 0, 0, 0, 1, 4); load_gconv3_weight_self.wait();
    // auto load_gconv3_weight_neigbhor = krnl_trans(boW_gconv3_neighbor, BM, 1, 1, 1, 0, 1, 5); load_gconv3_weight_neigbhor.wait();
    // auto load_gconv4_weight_self = krnl_trans(boW_gconv4_self, BM, 0, 0, 0, 0, 1, 6); load_gconv4_weight_self.wait();
    // auto load_gconv4_weight_neigbhor = krnl_trans(boW_gconv4_neighbor, BM, 1, 1, 1, 0, 1, 7); load_gconv4_weight_neigbhor.wait();

    // auto load_weight_run = xrt::run(krnl_mlp); 
    // load_weight_run.set_arg(0, weight_l1);
    // load_weight_run.set_arg(1, weight_l2);
    // load_weight_run.set_arg(2, 1);
    // load_weight_run.set_arg(3, result_holder);
    // load_weight_run.start();
    // load_weight_run.wait();


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

    auto run_self_trans = krnl_layerone(
        boW_self,
        BM, 0, 0, 0,
        2, 0 // new parameter v11
        );
    auto run_self_trans_Load1 = load_mm1(boX_bank1, 128*128, 2);
    // auto run_self_trans_Load2 = load_mm2(boX_bank2, 128*128);
    // auto run_self_trans_Load3 = load_mm3(boX_bank3, 128*128);
    auto run_self_trans_Store1 = store_mm1(boAX_Merge_bank1, 128*128, 2);
    auto run_self_trans_Store2 = store_mm2(boAX_Merge_bank2, 128*128, 2);
    auto run_self_trans_Store3 = store_mm3(boAX_Merge_bank3, 128*128, 2);
    
    run_p2.wait();
    run_p1.wait();

    run_self_trans_Load1.wait();
    // run_self_trans_Load2.wait();
    // run_self_trans_Load3.wait();
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
        3, BM, 1, // v8 new parameters
        boAX_Merge_bank1, boAX_Merge_bank2, boAX_Merge_bank3 // v8 new parameters
    ); 

    auto run_neigh_trans = krnl_layerone(
        boW_neighbor,
        BM, 1, 1, 1,
        2, 1 // new parameter v11
        );

    auto run_neigh_trans_Load1 = load_mm1(boAX_bank1, 128*128, 2);
    // auto run_neigh_trans_Load2 = load_mm2(boAX_bank2, 128*128);
    // auto run_neigh_trans_Load3 = load_mm3(boAX_bank3, 128*128);
    auto run_neigh_trans_Store1 = store_mm1(boC_bank1, 128*128, 2);
    auto run_neigh_trans_Store2 = store_mm2(boC_bank2, 128*128, 2);
    auto run_neigh_trans_Store3 = store_mm3(boC_bank3, 128*128, 2);

    run_neigh_trans_Load1.wait();
    // run_neigh_trans_Load2.wait();
    // run_neigh_trans_Load3.wait();
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

    

    boAX_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_Merge_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_Merge_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAX_Merge_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC_bank3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // result_holder.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // float perdata = 0;

    // for(int i = 0; i < num_vertex; i++){
    //     for(int f = 0; f < 3; f++){
    //         for(int j = 0; j < VDATA_SIZE; j++){
    //             if(f == 0) perdata = boAX_map_bank1[i].data[j]; //boAX_Merge_map_bank1 boAX_map_bank1
    //             if(f == 1) perdata = boAX_map_bank2[i].data[j];
    //             if(f == 2) perdata = boAX_map_bank3[i].data[j];
    //             printf("perdata: %f\n", perdata);
    //         }
    //     }
    // }


    verify_gconv1(boC_map_bank1, boC_map_bank2, boC_map_bank3);
    // verify_pooling1(boC_pooling1_bank1_map, boC_pooling1_bank2_map, boC_pooling1_bank3_map);
    //verify_gconv2(boC_gconv2_map_bank1, boC_gconv2_map_bank2, boC_gconv2_map_bank3);
    //verify_pooling2(boC_pooling2_bank1_map, boC_pooling2_bank2_map, boC_pooling2_bank3_map);
    //verify_gconv3(boC_gconv3_map_bank1, boC_gconv3_map_bank2, boC_gconv3_map_bank3);
    //verify_pooling3(boC_pooling3_bank1_map, boC_pooling3_bank2_map, boC_pooling3_bank3_map);
    //verify_gconv4(boC_gconv4_map_bank1, boC_gconv4_map_bank2, boC_gconv4_map_bank3);
    // std::ifstream fin_After_fc2("../../Layer/data/intermediate/After_fc2.bin", std::ios::binary);
    // float fdata_f;
    // for(int i = 0; i < 10; i++){
    //     fin_After_fc2.read(reinterpret_cast<char*>(&fdata_f), sizeof(float));
    //     printf("golden %f, myresult %f\n", fdata_f, result_holder_map[0].data[i]);
    // }

}