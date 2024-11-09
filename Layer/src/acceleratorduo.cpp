#include <datatype.h>
#include <acceleratorduo.h>


accelerator::accelerator(xrt::device & mydevice, xrt::uuid & uuid, int copy){
    // parameters for gconv1
    device = mydevice;

    imagescale = 128; 
    num_vertex = imagescale * imagescale;
    num_edge_block = imagescale * 32 * 9;
    num_vertex_bank = 128*16;
    f_block_len_custom = 1;
    BM = 1026;

    // parameters for pooling 1
    num_pool1_edges = 8*64*4;
    BM_pooling1 = 516;
    num_vertex_pooling1 = 64*64;

    // parameters for gconv2
    imagescale_gconv2 = 64; 
    num_vertex_gconv2 = imagescale_gconv2 * imagescale_gconv2;
    num_edge_block_gconv2 = imagescale_gconv2 * 16 * 9;
    num_vertex_bank_gconv2 = 64*16;
    f_block_len_custom_gconv2 = 3;
    BM_gconv2 = 258;

    // parameters for pooling 2
    num_pool2_edges = 8*32*4;
    BM_pooling2 = 66;
    num_vertex_pooling2 = 32*32;

    // parameters for gconv3
    imagescale_gconv3 = 32; 
    num_vertex_gconv3 = imagescale_gconv3 * imagescale_gconv3;
    num_edge_block_gconv3 = imagescale_gconv3 * 8 * 9;
    num_vertex_bank_gconv3 = 32*8;
    f_block_len_custom_gconv3 = 3;
    BM_gconv3 = 66;

    // parameters for pooling 3
    num_pool3_edges = 4*16*4;
    BM_pooling3 = 18;
    num_vertex_pooling3 = 16*16;

    // parameters for gconv 4
    imagescale_gconv4 = 16; 
    num_vertex_gconv4 = imagescale_gconv4 * imagescale_gconv4;
    num_edge_block_gconv4 = imagescale_gconv4 * 4 * 9;
    num_vertex_bank_gconv4 = 16*4;
    f_block_len_custom_gconv4 = 3;
    BM_gconv4 = 18;


    cp1_krnl_feagg = xrt::kernel(device, uuid, "feagg_top:{copy1_feagg_top_1}", xrt::kernel::cu_access_mode::exclusive);
    cp1_krnl_trans = xrt::kernel(device, uuid, "mmult:{copy1_mmult_1}", xrt::kernel::cu_access_mode::exclusive);
    cp1_load_mm1 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{copy1_read_AX_fromDDR_bank_1}", xrt::kernel::cu_access_mode::exclusive);
    cp1_load_mm2 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{copy1_read_AX_fromDDR_bank_2}", xrt::kernel::cu_access_mode::exclusive);
    cp1_load_mm3 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{copy1_read_AX_fromDDR_bank_3}", xrt::kernel::cu_access_mode::exclusive);
    cp1_store_mm1 = xrt::kernel(device, uuid, "store_output_matrix:{copy1_store_output_matrix_1}", xrt::kernel::cu_access_mode::exclusive);
    cp1_store_mm2 = xrt::kernel(device, uuid, "store_output_matrix:{copy1_store_output_matrix_2}", xrt::kernel::cu_access_mode::exclusive);
    cp1_store_mm3 = xrt::kernel(device, uuid, "store_output_matrix:{copy1_store_output_matrix_3}", xrt::kernel::cu_access_mode::exclusive);
    cp1_krnl_mlp = xrt::kernel(device, uuid, "mlp:{copy1_mlp_1}", xrt::kernel::cu_access_mode::exclusive);

    cp2_krnl_feagg = xrt::kernel(device, uuid, "feagg_top:{copy2_feagg_top_1}", xrt::kernel::cu_access_mode::exclusive);
    cp2_krnl_trans = xrt::kernel(device, uuid, "mmult:{copy2_mmult_1}", xrt::kernel::cu_access_mode::exclusive);
    cp2_load_mm1 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{copy2_read_AX_fromDDR_bank_1}", xrt::kernel::cu_access_mode::exclusive);
    cp2_load_mm2 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{copy2_read_AX_fromDDR_bank_2}", xrt::kernel::cu_access_mode::exclusive);
    cp2_load_mm3 = xrt::kernel(device, uuid, "read_AX_fromDDR_bank:{copy2_read_AX_fromDDR_bank_3}", xrt::kernel::cu_access_mode::exclusive);
    cp2_store_mm1 = xrt::kernel(device, uuid, "store_output_matrix:{copy2_store_output_matrix_1}", xrt::kernel::cu_access_mode::exclusive);
    cp2_store_mm2 = xrt::kernel(device, uuid, "store_output_matrix:{copy2_store_output_matrix_2}", xrt::kernel::cu_access_mode::exclusive);
    cp2_store_mm3 = xrt::kernel(device, uuid, "store_output_matrix:{copy2_store_output_matrix_3}", xrt::kernel::cu_access_mode::exclusive);
    cp2_krnl_mlp = xrt::kernel(device, uuid, "mlp:{copy2_mlp_1}", xrt::kernel::cu_access_mode::exclusive);
    
}


void accelerator::preparation(){

    //  allocate memory space for gconv1 ========================================================================
    cp1_boX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp1_krnl_feagg.group_id(0)); // 2 //Match kernel arguments to RTL kernel
    cp1_boX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp1_krnl_feagg.group_id(1)); // 14 //Match kernel arguments to RTL kernel
    cp1_boX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp1_krnl_feagg.group_id(2)); // 30  // Match kernel arguments to RTL kernel
    cp1_boA_p1 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), cp1_krnl_feagg.group_id(3));
    cp1_boA_p2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), cp1_krnl_feagg.group_id(3));
    cp1_boAX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp1_krnl_feagg.group_id(4)); // 3
    cp1_boAX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp1_krnl_feagg.group_id(5)); // 15
    cp1_boAX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp1_krnl_feagg.group_id(6)); // 31

    v_edge * cp1_boA_p1_map = cp1_boA_p1.map<v_edge *>();
    v_edge * cp1_boA_p2_map = cp1_boA_p2.map<v_edge *>();
    create_edges_layer1(cp1_boA_p1_map, cp1_boA_p2_map);
    cp1_boA_p1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp1_boA_p2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp1_boW_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), cp1_krnl_trans.group_id(0));
    cp1_boAX_Merge_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM * VDATA_SIZE * VDATA_SIZE), cp1_load_mm1.group_id(0)); //
    cp1_boAX_Merge_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM * VDATA_SIZE * VDATA_SIZE), cp1_load_mm2.group_id(0)); //
    cp1_boAX_Merge_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM * VDATA_SIZE * VDATA_SIZE), cp1_load_mm3.group_id(0)); //
    cp1_boW_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), cp1_krnl_trans.group_id(0));
    cp1_boC_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm1.group_id(0));
    cp1_boC_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm2.group_id(0));
    cp1_boC_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm3.group_id(0));

    v_dt* cp1_boW_map_self = cp1_boW_self.map<v_dt*>();
    v_dt* cp1_boW_neighbor_map = cp1_boW_neighbor.map<v_dt*>();
    load_gconv1(cp1_boW_map_self, cp1_boW_neighbor_map);
    cp1_boW_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp1_boW_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for pooling1 ========================================================================    
    cp1_boA_pool1 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool1_edges), cp1_krnl_feagg.group_id(3));
    cp1_boC_pooling1_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp1_krnl_feagg.group_id(0));
    cp1_boC_pooling1_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp1_krnl_feagg.group_id(1));
    cp1_boC_pooling1_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp1_krnl_feagg.group_id(2));

    v_edge * cp1_boA_pool1_map = cp1_boA_pool1.map<v_edge*>();
    create_edges_pooling1(cp1_boA_pool1_map);
    cp1_boA_pool1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for gconv2 ========================================================================
    cp1_boA_gconv2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv2), cp1_krnl_feagg.group_id(3));
    cp1_boAX_gconv2_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp1_krnl_feagg.group_id(4));
    cp1_boAX_gconv2_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp1_krnl_feagg.group_id(5));
    cp1_boAX_gconv2_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp1_krnl_feagg.group_id(6));

    v_edge * cp1_boA_gconv2_map = cp1_boA_gconv2.map<v_edge *>();
    create_edges_gconv2(cp1_boA_gconv2_map);
    cp1_boA_gconv2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp1_boW_gconv2_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE),  cp1_krnl_trans.group_id(0));
    cp1_boAX_Merge_gconv2_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm1.group_id(0));
    cp1_boAX_Merge_gconv2_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm2.group_id(0));
    cp1_boAX_Merge_gconv2_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm3.group_id(0));
    cp1_boW_gconv2_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE),  cp1_krnl_trans.group_id(0));
    cp1_boC_gconv2_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm1.group_id(0));
    cp1_boC_gconv2_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm2.group_id(0));
    cp1_boC_gconv2_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm3.group_id(0));

    v_dt* cp1_boW_gconv2_map_self = cp1_boW_gconv2_self.map<v_dt*>();
    v_dt* cp1_boW_gconv2_neighbor_map = cp1_boW_gconv2_neighbor.map<v_dt*>();
    load_gconv2(cp1_boW_gconv2_map_self, cp1_boW_gconv2_neighbor_map);
    cp1_boW_gconv2_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp1_boW_gconv2_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for pooling2 ========================================================================
    cp1_boA_pool2 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool2_edges), cp1_krnl_feagg.group_id(3));
    cp1_boC_pooling2_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp1_krnl_feagg.group_id(0));
    cp1_boC_pooling2_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp1_krnl_feagg.group_id(1));
    cp1_boC_pooling2_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp1_krnl_feagg.group_id(2));

    v_edge * cp1_boA_pool2_map = cp1_boA_pool2.map<v_edge*>();
    create_edges_pooling2(cp1_boA_pool2_map);
    cp1_boA_pool2.sync(XCL_BO_SYNC_BO_TO_DEVICE);    

    //  allocate memory space for gconv3 ========================================================================
    cp1_boA_gconv3 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv3), cp1_krnl_feagg.group_id(3));
    cp1_boAX_gconv3_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp1_krnl_feagg.group_id(0));
    cp1_boAX_gconv3_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp1_krnl_feagg.group_id(1));
    cp1_boAX_gconv3_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp1_krnl_feagg.group_id(2));

    v_edge * cp1_boA_gconv3_map = cp1_boA_gconv3.map<v_edge *>();
    create_edges_gconv3(cp1_boA_gconv3_map);
    cp1_boA_gconv3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp1_boW_gconv3_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), cp1_krnl_trans.group_id(0));
    cp1_boAX_Merge_gconv3_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm1.group_id(0));
    cp1_boAX_Merge_gconv3_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm2.group_id(0));
    cp1_boAX_Merge_gconv3_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm3.group_id(0));
    cp1_boW_gconv3_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), cp1_krnl_trans.group_id(0));
    cp1_boC_gconv3_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm1.group_id(0));
    cp1_boC_gconv3_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm2.group_id(0));
    cp1_boC_gconv3_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm3.group_id(0));

    v_dt* cp1_boW_gconv3_map_self = cp1_boW_gconv3_self.map<v_dt*>();
    v_dt* cp1_boW_gconv3_neighbor_map = cp1_boW_gconv3_neighbor.map<v_dt*>();
    load_gconv3(cp1_boW_gconv3_map_self, cp1_boW_gconv3_neighbor_map);
    cp1_boW_gconv3_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp1_boW_gconv3_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);   

    //  allocate memory space for pooling3 ========================================================================
    cp1_boA_pool3 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool3_edges), cp1_krnl_feagg.group_id(3));
    cp1_boC_pooling3_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp1_krnl_feagg.group_id(0));
    cp1_boC_pooling3_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp1_krnl_feagg.group_id(1));
    cp1_boC_pooling3_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp1_krnl_feagg.group_id(2));

    v_edge * cp1_boA_pool3_map = cp1_boA_pool3.map<v_edge*>();
    create_edges_pooling3(cp1_boA_pool3_map);
    cp1_boA_pool3.sync(XCL_BO_SYNC_BO_TO_DEVICE);    

    //  allocate memory space for gconv4 ========================================================================
    cp1_boA_gconv4 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv4), cp1_krnl_feagg.group_id(3));
    cp1_boAX_gconv4_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp1_krnl_feagg.group_id(0));
    cp1_boAX_gconv4_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp1_krnl_feagg.group_id(1));
    cp1_boAX_gconv4_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp1_krnl_feagg.group_id(2));

    v_edge * cp1_boA_gconv4_map = cp1_boA_gconv4.map<v_edge *>();
    create_edges_gconv4(cp1_boA_gconv4_map);
    cp1_boA_gconv4.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp1_boW_gconv4_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), cp1_krnl_trans.group_id(0));
    cp1_boAX_Merge_gconv4_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm1.group_id(0));
    cp1_boAX_Merge_gconv4_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm2.group_id(0));
    cp1_boAX_Merge_gconv4_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), cp1_load_mm3.group_id(0));
    cp1_boW_gconv4_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), cp1_krnl_trans.group_id(0));
    cp1_boC_gconv4_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm1.group_id(0));
    cp1_boC_gconv4_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm2.group_id(0));
    cp1_boC_gconv4_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), cp1_store_mm3.group_id(0));

    v_dt* cp1_boW_gconv4_map_self = cp1_boW_gconv4_self.map<v_dt*>();
    v_dt* cp1_boW_gconv4_neighbor_map = cp1_boW_gconv4_neighbor.map<v_dt*>();
    load_gconv4(cp1_boW_gconv4_map_self, cp1_boW_gconv4_neighbor_map);
    cp1_boW_gconv4_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp1_boW_gconv4_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for mlp ========================================================================
    cp1_weight_l1 = xrt::bo(device, size_t(sizeof(v_dt) * (64 * 256 * 3 + 4)), cp1_krnl_mlp.group_id(0));
    cp1_weight_l2 = xrt::bo(device, size_t(sizeof(v_dt) * (64 + 1)), cp1_krnl_mlp.group_id(1));
    cp1_result_holder = xrt::bo(device, size_t(sizeof(v_dt)), cp1_krnl_mlp.group_id(3));

    v_dt * cp1_weight_l1_map = cp1_weight_l1.map<v_dt *>();
    v_dt * cp1_weight_l2_map = cp1_weight_l2.map<v_dt *>();
    allocateMLP(cp1_weight_l1_map, cp1_weight_l2_map);

    cp1_weight_l1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp1_weight_l2.sync(XCL_BO_SYNC_BO_TO_DEVICE);


    // preparation for accelerator 2 **************************************************************************************************************************//
    
    //  allocate memory space for gconv1 ========================================================================
    cp2_boX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp2_krnl_feagg.group_id(0)); // 2 //Match kernel arguments to RTL kernel
    cp2_boX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp2_krnl_feagg.group_id(1)); // 14 //Match kernel arguments to RTL kernel
    cp2_boX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp2_krnl_feagg.group_id(2)); // 30  // Match kernel arguments to RTL kernel
    cp2_boA_p1 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), cp2_krnl_feagg.group_id(3));
    cp2_boA_p2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block / 2), cp2_krnl_feagg.group_id(3));
    cp2_boAX_bank1 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp2_krnl_feagg.group_id(4)); // 3
    cp2_boAX_bank2 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp2_krnl_feagg.group_id(5)); // 15
    cp2_boAX_bank3 = xrt::bo(device, size_t(sizeof(v_float) * num_vertex), cp2_krnl_feagg.group_id(6)); // 31

    v_edge * cp2_boA_p1_map = cp2_boA_p1.map<v_edge *>();
    v_edge * cp2_boA_p2_map = cp2_boA_p2.map<v_edge *>();
    create_edges_layer1(cp2_boA_p1_map, cp2_boA_p2_map);
    cp2_boA_p1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp2_boA_p2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp2_boW_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), cp2_krnl_trans.group_id(0));
    cp2_boAX_Merge_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM * VDATA_SIZE * VDATA_SIZE), cp2_load_mm1.group_id(0)); //
    cp2_boAX_Merge_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM * VDATA_SIZE * VDATA_SIZE), cp2_load_mm2.group_id(0)); //
    cp2_boAX_Merge_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM * VDATA_SIZE * VDATA_SIZE), cp2_load_mm3.group_id(0)); //
    cp2_boW_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), cp2_krnl_trans.group_id(0));
    cp2_boC_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm1.group_id(0));
    cp2_boC_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm2.group_id(0));
    cp2_boC_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm3.group_id(0));

    v_dt* cp2_boW_map_self = cp2_boW_self.map<v_dt*>();
    v_dt* cp2_boW_neighbor_map = cp2_boW_neighbor.map<v_dt*>();
    load_gconv1(cp2_boW_map_self, cp2_boW_neighbor_map);
    cp2_boW_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp2_boW_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for pooling1 ========================================================================    
    cp2_boA_pool1 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool1_edges), cp2_krnl_feagg.group_id(3));
    cp2_boC_pooling1_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp2_krnl_feagg.group_id(0));
    cp2_boC_pooling1_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp2_krnl_feagg.group_id(1));
    cp2_boC_pooling1_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp2_krnl_feagg.group_id(2));

    v_edge * cp2_boA_pool1_map = cp2_boA_pool1.map<v_edge*>();
    create_edges_pooling1(cp2_boA_pool1_map);
    cp2_boA_pool1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for gconv2 ========================================================================
    cp2_boA_gconv2 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv2), cp2_krnl_feagg.group_id(3));
    cp2_boAX_gconv2_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp2_krnl_feagg.group_id(4));
    cp2_boAX_gconv2_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp2_krnl_feagg.group_id(5));
    cp2_boAX_gconv2_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 4128), cp2_krnl_feagg.group_id(6));

    v_edge * cp2_boA_gconv2_map = cp2_boA_gconv2.map<v_edge *>();
    create_edges_gconv2(cp2_boA_gconv2_map);
    cp2_boA_gconv2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp2_boW_gconv2_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE),  cp2_krnl_trans.group_id(0));
    cp2_boAX_Merge_gconv2_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm1.group_id(0));
    cp2_boAX_Merge_gconv2_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm2.group_id(0));
    cp2_boAX_Merge_gconv2_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm3.group_id(0));
    cp2_boW_gconv2_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE),  cp2_krnl_trans.group_id(0));
    cp2_boC_gconv2_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm1.group_id(0));
    cp2_boC_gconv2_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm2.group_id(0));
    cp2_boC_gconv2_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv2  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm3.group_id(0));

    v_dt* cp2_boW_gconv2_map_self = cp2_boW_gconv2_self.map<v_dt*>();
    v_dt* cp2_boW_gconv2_neighbor_map = cp2_boW_gconv2_neighbor.map<v_dt*>();
    load_gconv2(cp2_boW_gconv2_map_self, cp2_boW_gconv2_neighbor_map);
    cp2_boW_gconv2_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp2_boW_gconv2_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for pooling2 ========================================================================
    cp2_boA_pool2 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool2_edges), cp2_krnl_feagg.group_id(3));
    cp2_boC_pooling2_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp2_krnl_feagg.group_id(0));
    cp2_boC_pooling2_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp2_krnl_feagg.group_id(1));
    cp2_boC_pooling2_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp2_krnl_feagg.group_id(2));

    v_edge * cp2_boA_pool2_map = cp2_boA_pool2.map<v_edge*>();
    create_edges_pooling2(cp2_boA_pool2_map);
    cp2_boA_pool2.sync(XCL_BO_SYNC_BO_TO_DEVICE);    

    //  allocate memory space for gconv3 ========================================================================
    cp2_boA_gconv3 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv3), cp2_krnl_feagg.group_id(3));
    cp2_boAX_gconv3_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp2_krnl_feagg.group_id(0));
    cp2_boAX_gconv3_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp2_krnl_feagg.group_id(1));
    cp2_boAX_gconv3_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 1056), cp2_krnl_feagg.group_id(2));

    v_edge * cp2_boA_gconv3_map = cp2_boA_gconv3.map<v_edge *>();
    create_edges_gconv3(cp2_boA_gconv3_map);
    cp2_boA_gconv3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp2_boW_gconv3_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), cp2_krnl_trans.group_id(0));
    cp2_boAX_Merge_gconv3_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm1.group_id(0));
    cp2_boAX_Merge_gconv3_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm2.group_id(0));
    cp2_boAX_Merge_gconv3_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm3.group_id(0));
    cp2_boW_gconv3_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), cp2_krnl_trans.group_id(0));
    cp2_boC_gconv3_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm1.group_id(0));
    cp2_boC_gconv3_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm2.group_id(0));
    cp2_boC_gconv3_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv3  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm3.group_id(0));

    v_dt* cp2_boW_gconv3_map_self = cp2_boW_gconv3_self.map<v_dt*>();
    v_dt* cp2_boW_gconv3_neighbor_map = cp2_boW_gconv3_neighbor.map<v_dt*>();
    load_gconv3(cp2_boW_gconv3_map_self, cp2_boW_gconv3_neighbor_map);
    cp2_boW_gconv3_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp2_boW_gconv3_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);   

    //  allocate memory space for pooling3 ========================================================================
    cp2_boA_pool3 = xrt::bo(device, size_t(sizeof(v_edge) * num_pool3_edges), cp2_krnl_feagg.group_id(3));
    cp2_boC_pooling3_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp2_krnl_feagg.group_id(0));
    cp2_boC_pooling3_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp2_krnl_feagg.group_id(1));
    cp2_boC_pooling3_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp2_krnl_feagg.group_id(2));

    v_edge * cp2_boA_pool3_map = cp2_boA_pool3.map<v_edge*>();
    create_edges_pooling3(cp2_boA_pool3_map);
    cp2_boA_pool3.sync(XCL_BO_SYNC_BO_TO_DEVICE);    

    //  allocate memory space for gconv4 ========================================================================
    cp2_boA_gconv4 = xrt::bo(device, size_t(sizeof(v_edge) * num_edge_block_gconv4), cp2_krnl_feagg.group_id(3));
    cp2_boAX_gconv4_bank1 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp2_krnl_feagg.group_id(0));
    cp2_boAX_gconv4_bank2 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp2_krnl_feagg.group_id(1));
    cp2_boAX_gconv4_bank3 = xrt::bo(device, size_t(sizeof(v_float) * 288), cp2_krnl_feagg.group_id(2));

    v_edge * cp2_boA_gconv4_map = cp2_boA_gconv4.map<v_edge *>();
    create_edges_gconv4(cp2_boA_gconv4_map);
    cp2_boA_gconv4.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cp2_boW_gconv4_self = xrt::bo(device, size_t(sizeof(datatype) * BLOCK_NUM * BLOCK_NUM * VDATA_SIZE * VDATA_SIZE), cp2_krnl_trans.group_id(0));
    cp2_boAX_Merge_gconv4_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm1.group_id(0));
    cp2_boAX_Merge_gconv4_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm2.group_id(0));
    cp2_boAX_Merge_gconv4_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4 * VDATA_SIZE * VDATA_SIZE), cp2_load_mm3.group_id(0));
    cp2_boW_gconv4_neighbor = xrt::bo(device, size_t(sizeof(datatype) * (BLOCK_NUM * VDATA_SIZE  + 1)* BLOCK_NUM  * VDATA_SIZE), cp2_krnl_trans.group_id(0));
    cp2_boC_gconv4_bank1 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm1.group_id(0));
    cp2_boC_gconv4_bank2 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm2.group_id(0));
    cp2_boC_gconv4_bank3 = xrt::bo(device, size_t(sizeof(datatype) * BM_gconv4  * VDATA_SIZE * VDATA_SIZE), cp2_store_mm3.group_id(0));

    v_dt* cp2_boW_gconv4_map_self = cp2_boW_gconv4_self.map<v_dt*>();
    v_dt* cp2_boW_gconv4_neighbor_map = cp2_boW_gconv4_neighbor.map<v_dt*>();
    load_gconv4(cp2_boW_gconv4_map_self, cp2_boW_gconv4_neighbor_map);
    cp2_boW_gconv4_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp2_boW_gconv4_neighbor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    //  allocate memory space for mlp ========================================================================
    cp2_weight_l1 = xrt::bo(device, size_t(sizeof(v_dt) * (64 * 256 * 3 + 4)), cp2_krnl_mlp.group_id(0));
    cp2_weight_l2 = xrt::bo(device, size_t(sizeof(v_dt) * (64 + 1)), cp2_krnl_mlp.group_id(1));
    cp2_result_holder = xrt::bo(device, size_t(sizeof(v_dt)), cp2_krnl_mlp.group_id(3));

    v_dt * cp2_weight_l1_map = cp2_weight_l1.map<v_dt *>();
    v_dt * cp2_weight_l2_map = cp2_weight_l2.map<v_dt *>();
    allocateMLP(cp2_weight_l1_map, cp2_weight_l2_map);

    cp2_weight_l1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cp2_weight_l2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

}


void accelerator::loadweight(){

    // load weight for feature transformation module
    auto cp1_load_gconv1_weight_self = cp1_krnl_trans(cp1_boW_self, BM, 0, 0, 0, 0, 1, 0); cp1_load_gconv1_weight_self.wait();
    auto cp1_load_gconv1_weight_neigbhor = cp1_krnl_trans(cp1_boW_neighbor, BM, 1, 1, 1, 0, 1, 1); cp1_load_gconv1_weight_neigbhor.wait();
    auto cp1_load_gconv2_weight_self = cp1_krnl_trans(cp1_boW_gconv2_self, BM, 0, 0, 0, 0, 1, 2); cp1_load_gconv2_weight_self.wait();
    auto cp1_load_gconv2_weight_neigbhor = cp1_krnl_trans(cp1_boW_gconv2_neighbor, BM, 1, 1, 1, 0, 1, 3); cp1_load_gconv2_weight_neigbhor.wait();
    auto cp1_load_gconv3_weight_self = cp1_krnl_trans(cp1_boW_gconv3_self, BM, 0, 0, 0, 0, 1, 4); cp1_load_gconv3_weight_self.wait();
    auto cp1_load_gconv3_weight_neigbhor = cp1_krnl_trans(cp1_boW_gconv3_neighbor, BM, 1, 1, 1, 0, 1, 5); cp1_load_gconv3_weight_neigbhor.wait();
    auto cp1_load_gconv4_weight_self = cp1_krnl_trans(cp1_boW_gconv4_self, BM, 0, 0, 0, 0, 1, 6); cp1_load_gconv4_weight_self.wait();
    auto cp1_load_gconv4_weight_neigbhor = cp1_krnl_trans(cp1_boW_gconv4_neighbor, BM, 1, 1, 1, 0, 1, 7); cp1_load_gconv4_weight_neigbhor.wait();

    // load weight for MLP
    auto cp1_load_weight_run = xrt::run(cp1_krnl_mlp); 
    cp1_load_weight_run.set_arg(0, cp1_weight_l1);
    cp1_load_weight_run.set_arg(1, cp1_weight_l2);
    cp1_load_weight_run.set_arg(2, 1);
    cp1_load_weight_run.set_arg(3, cp1_result_holder);
    cp1_load_weight_run.start();
    cp1_load_weight_run.wait();

    // load weight for accelerator 2

    // load weight for feature transformation module
    auto cp2_load_gconv1_weight_self = cp2_krnl_trans(cp2_boW_self, BM, 0, 0, 0, 0, 1, 0); cp2_load_gconv1_weight_self.wait();
    auto cp2_load_gconv1_weight_neigbhor = cp2_krnl_trans(cp2_boW_neighbor, BM, 1, 1, 1, 0, 1, 1); cp2_load_gconv1_weight_neigbhor.wait();
    auto cp2_load_gconv2_weight_self = cp2_krnl_trans(cp2_boW_gconv2_self, BM, 0, 0, 0, 0, 1, 2); cp2_load_gconv2_weight_self.wait();
    auto cp2_load_gconv2_weight_neigbhor = cp2_krnl_trans(cp2_boW_gconv2_neighbor, BM, 1, 1, 1, 0, 1, 3); cp2_load_gconv2_weight_neigbhor.wait();
    auto cp2_load_gconv3_weight_self = cp2_krnl_trans(cp2_boW_gconv3_self, BM, 0, 0, 0, 0, 1, 4); cp2_load_gconv3_weight_self.wait();
    auto cp2_load_gconv3_weight_neigbhor = cp2_krnl_trans(cp2_boW_gconv3_neighbor, BM, 1, 1, 1, 0, 1, 5); cp2_load_gconv3_weight_neigbhor.wait();
    auto cp2_load_gconv4_weight_self = cp2_krnl_trans(cp2_boW_gconv4_self, BM, 0, 0, 0, 0, 1, 6); cp2_load_gconv4_weight_self.wait();
    auto cp2_load_gconv4_weight_neigbhor = cp2_krnl_trans(cp2_boW_gconv4_neighbor, BM, 1, 1, 1, 0, 1, 7); cp2_load_gconv4_weight_neigbhor.wait();

    // load weight for MLP
    auto cp2_load_weight_run = xrt::run(cp2_krnl_mlp); 
    cp2_load_weight_run.set_arg(0, cp2_weight_l1);
    cp2_load_weight_run.set_arg(1, cp2_weight_l2);
    cp2_load_weight_run.set_arg(2, 1);
    cp2_load_weight_run.set_arg(3, cp2_result_holder);
    cp2_load_weight_run.start();
    cp2_load_weight_run.wait();
}



void accelerator::inference(){
    // Execute gconv1 =====================================================================================
    // printf("Execute gconv1 =====================================================================================\n");
    auto cp1_run_p1 = cp1_krnl_feagg(
        cp1_boX_bank1, cp1_boX_bank2, cp1_boX_bank3, 
        cp1_boA_p1, 
        cp1_boAX_bank1, cp1_boAX_bank2, cp1_boAX_bank3, 
        num_vertex_bank, num_edge_block/2, 
        f_block_len_custom, imagescale, 
        1, 0, 0, 0, num_vertex_bank, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        );  
    
    auto cp1_run_p2 = cp1_krnl_feagg(
        cp1_boX_bank1, cp1_boX_bank2, cp1_boX_bank3, 
        cp1_boA_p2, 
        cp1_boAX_bank1, cp1_boAX_bank2, cp1_boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    ); 

    auto cp1_run_self_trans = cp1_krnl_trans(
        cp1_boW_self,
        BM, 0, 0, 0, 0,
        2, 0 // new parameter v11
        );
    auto cp1_run_self_trans_Load1 = cp1_load_mm1(cp1_boX_bank1, BM);
    auto cp1_run_self_trans_Load2 = cp1_load_mm2(cp1_boX_bank2, BM);
    auto cp1_run_self_trans_Load3 = cp1_load_mm3(cp1_boX_bank3, BM);
    auto cp1_run_self_trans_Store1 = cp1_store_mm1(cp1_boAX_Merge_bank1, BM);
    auto cp1_run_self_trans_Store2 = cp1_store_mm2(cp1_boAX_Merge_bank2, BM);
    auto cp1_run_self_trans_Store3 = cp1_store_mm3(cp1_boAX_Merge_bank3, BM);

    // auto cp2_run_p1 = cp2_krnl_feagg(
    //     cp2_boX_bank1, cp2_boX_bank2, cp2_boX_bank3, 
    //     cp2_boA_p1, 
    //     cp2_boAX_bank1, cp2_boAX_bank2, cp2_boAX_bank3, 
    //     num_vertex_bank, num_edge_block/2, 
    //     f_block_len_custom, imagescale, 
    //     1, 0, 0, 0, num_vertex_bank, 0,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     );  
    
    // auto cp2_run_p2 = cp2_krnl_feagg(
    //     cp2_boX_bank1, cp2_boX_bank2, cp2_boX_bank3, 
    //     cp2_boA_p2, 
    //     cp2_boAX_bank1, cp2_boAX_bank2, cp2_boAX_bank3, 
    //     num_vertex_bank, num_edge_block/2,
    //     f_block_len_custom, imagescale,
    //     0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    // ); 

    // auto cp2_run_self_trans = cp2_krnl_trans(
    //     cp2_boW_self,
    //     BM, 0, 0, 0, 0,
    //     2, 0 // new parameter v11
    //     );
    // auto cp2_run_self_trans_Load1 = cp2_load_mm1(cp2_boX_bank1, BM);
    // auto cp2_run_self_trans_Load2 = cp2_load_mm2(cp2_boX_bank2, BM);
    // auto cp2_run_self_trans_Load3 = cp2_load_mm3(cp2_boX_bank3, BM);
    // auto cp2_run_self_trans_Store1 = cp2_store_mm1(cp2_boAX_Merge_bank1, BM);
    // auto cp2_run_self_trans_Store2 = cp2_store_mm2(cp2_boAX_Merge_bank2, BM);
    // auto cp2_run_self_trans_Store3 = cp2_store_mm3(cp2_boAX_Merge_bank3, BM);
    
    cp1_run_p2.wait();
    cp1_run_p1.wait();
    cp1_run_self_trans_Load1.wait();
    cp1_run_self_trans_Load2.wait();
    cp1_run_self_trans_Load3.wait();
    cp1_run_self_trans_Store1.wait();
    cp1_run_self_trans_Store2.wait();
    cp1_run_self_trans_Store3.wait();
    cp1_run_self_trans.wait();

    // cp2_run_p2.wait();
    // cp2_run_p1.wait();
    // cp2_run_self_trans_Load1.wait();
    // cp2_run_self_trans_Load2.wait();
    // cp2_run_self_trans_Load3.wait();
    // cp2_run_self_trans_Store1.wait();
    // cp2_run_self_trans_Store2.wait();
    // cp2_run_self_trans_Store3.wait();
    // cp2_run_self_trans.wait();

    auto cp1_run_loaddata = cp1_krnl_feagg(
        cp1_boAX_bank1, cp1_boAX_bank2, cp1_boAX_bank3, 
        (v_edge *) 0, 
        (v_float *) 0, (v_float *) 0, (v_float *) 0, 
        0, 0,
        0, 0,
        0, 0, 0, 0, 0, 0,
        2, BM, 1, // v8 new parameters
        cp1_boAX_Merge_bank1, cp1_boAX_Merge_bank2, cp1_boAX_Merge_bank3 // v8 new parameters
    ); 

    auto cp1_run_neigh_trans = cp1_krnl_trans(
        cp1_boW_neighbor,
        BM, 1, 1, 1, 0, 
        2, 1 // new parameter v11
        );

    auto cp1_run_neigh_trans_Load1 = cp1_load_mm1(cp1_boAX_bank1, BM);
    auto cp1_run_neigh_trans_Load2 = cp1_load_mm2(cp1_boAX_bank2, BM);
    auto cp1_run_neigh_trans_Load3 = cp1_load_mm3(cp1_boAX_bank3, BM);
    auto cp1_run_neigh_trans_Store1 = cp1_store_mm1(cp1_boC_bank1, BM);
    auto cp1_run_neigh_trans_Store2 = cp1_store_mm2(cp1_boC_bank2, BM);
    auto cp1_run_neigh_trans_Store3 = cp1_store_mm3(cp1_boC_bank3, BM);

    // auto cp2_run_loaddata = cp2_krnl_feagg(
    //     cp2_boAX_bank1, cp2_boAX_bank2, cp2_boAX_bank3, 
    //     (v_edge *) 0, 
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0, 
    //     0, 0,
    //     0, 0,
    //     0, 0, 0, 0, 0, 0,
    //     2, BM, 1, // v8 new parameters
    //     cp2_boAX_Merge_bank1, cp2_boAX_Merge_bank2, cp2_boAX_Merge_bank3 // v8 new parameters
    // ); 

    // auto cp2_run_neigh_trans = cp2_krnl_trans(
    //     cp2_boW_neighbor,
    //     BM, 1, 1, 1, 0, 
    //     2, 1 // new parameter v11
    //     );

    // auto cp2_run_neigh_trans_Load1 = cp2_load_mm1(cp2_boAX_bank1, BM);
    // auto cp2_run_neigh_trans_Load2 = cp2_load_mm2(cp2_boAX_bank2, BM);
    // auto cp2_run_neigh_trans_Load3 = cp2_load_mm3(cp2_boAX_bank3, BM);
    // auto cp2_run_neigh_trans_Store1 = cp2_store_mm1(cp2_boC_bank1, BM);
    // auto cp2_run_neigh_trans_Store2 = cp2_store_mm2(cp2_boC_bank2, BM);
    // auto cp2_run_neigh_trans_Store3 = cp2_store_mm3(cp2_boC_bank3, BM);

    cp1_run_neigh_trans_Load1.wait();
    cp1_run_neigh_trans_Load2.wait();
    cp1_run_neigh_trans_Load3.wait();
    cp1_run_neigh_trans_Store1.wait();
    cp1_run_neigh_trans_Store2.wait();
    cp1_run_neigh_trans_Store3.wait();
    cp1_run_loaddata.wait();
    cp1_run_neigh_trans.wait();

    // cp2_run_neigh_trans_Load1.wait();
    // cp2_run_neigh_trans_Load2.wait();
    // cp2_run_neigh_trans_Load3.wait();
    // cp2_run_neigh_trans_Store1.wait();
    // cp2_run_neigh_trans_Store2.wait();
    // cp2_run_neigh_trans_Store3.wait();
    // cp2_run_loaddata.wait();
    // cp2_run_neigh_trans.wait();

    // Execute pooling1 =====================================================================================
    // printf("Execute pooling1 =====================================================================================\n");
    auto cp1_run_poo1_part1 = cp1_krnl_feagg(
        cp1_boC_bank1, cp1_boC_bank2, cp1_boC_bank3, cp1_boA_pool1, 
        cp1_boC_pooling1_bank1, cp1_boC_pooling1_bank2, cp1_boC_pooling1_bank3, 
        num_vertex_bank, num_pool1_edges, 
        3, imagescale, 
        1, 1, 0, 0, num_vertex_pooling1/8, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        );  
    auto cp1_run_poo1_part2 = cp1_krnl_feagg(
        cp1_boC_bank1, cp1_boC_bank2, cp1_boC_bank3, cp1_boA_pool1, 
        cp1_boC_pooling1_bank1, cp1_boC_pooling1_bank2, cp1_boC_pooling1_bank3, 
        num_vertex_bank, num_pool1_edges,
        3, imagescale,
        1, 1, num_vertex_bank*4, num_vertex_pooling1/2, num_vertex_pooling1/8, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        );  

    // auto cp2_run_poo1_part1 = cp2_krnl_feagg(
    //     cp2_boC_bank1, cp2_boC_bank2, cp2_boC_bank3, cp2_boA_pool1, 
    //     cp2_boC_pooling1_bank1, cp2_boC_pooling1_bank2, cp2_boC_pooling1_bank3, 
    //     num_vertex_bank, num_pool1_edges, 
    //     3, imagescale, 
    //     1, 1, 0, 0, num_vertex_pooling1/8, 1,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     );  
    // auto cp2_run_poo1_part2 = cp2_krnl_feagg(
    //     cp2_boC_bank1, cp2_boC_bank2, cp2_boC_bank3, cp2_boA_pool1, 
    //     cp2_boC_pooling1_bank1, cp2_boC_pooling1_bank2, cp2_boC_pooling1_bank3, 
    //     num_vertex_bank, num_pool1_edges,
    //     3, imagescale,
    //     1, 1, num_vertex_bank*4, num_vertex_pooling1/2, num_vertex_pooling1/8, 1,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     );  
    cp1_run_poo1_part1.wait();
    cp1_run_poo1_part2.wait();
    // cp2_run_poo1_part1.wait();
    // cp2_run_poo1_part2.wait();

    // Execute gconv2 ==========================================================================================
    // printf("Execute gconv2 =====================================================================================\n");
    auto cp1_run_gconv2_agg = cp1_krnl_feagg(
        cp1_boC_pooling1_bank1, cp1_boC_pooling1_bank2, cp1_boC_pooling1_bank3, cp1_boA_gconv2, 
        cp1_boAX_gconv2_bank1, cp1_boAX_gconv2_bank2, cp1_boAX_gconv2_bank3, 
        num_vertex_bank_gconv2, num_edge_block_gconv2, 
        f_block_len_custom_gconv2, imagescale_gconv2, 
        1, 1, 0, 0, num_vertex_bank_gconv2, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    
    auto cp1_run_self_trans_gconv2 = cp1_krnl_trans(
        cp1_boW_gconv2_self,
        BM_gconv2, 0, 0, 0, 0,
        2, 2 // new parameter v11
        );
    auto cp1_run_self_trans_Load1_gconv2 = cp1_load_mm1(cp1_boC_pooling1_bank1, BM_gconv2);
    auto cp1_run_self_trans_Load2_gconv2 = cp1_load_mm2(cp1_boC_pooling1_bank2, BM_gconv2);
    auto cp1_run_self_trans_Load3_gconv2 = cp1_load_mm3(cp1_boC_pooling1_bank3, BM_gconv2);
    auto cp1_run_self_trans_Store1_gconv2 = cp1_store_mm1(cp1_boAX_Merge_gconv2_bank1, BM_gconv2);
    auto cp1_run_self_trans_Store2_gconv2 = cp1_store_mm2(cp1_boAX_Merge_gconv2_bank2, BM_gconv2);
    auto cp1_run_self_trans_Store3_gconv2 = cp1_store_mm3(cp1_boAX_Merge_gconv2_bank3, BM_gconv2);


    // auto cp2_run_gconv2_agg = cp2_krnl_feagg(
    //     cp2_boC_pooling1_bank1, cp2_boC_pooling1_bank2, cp2_boC_pooling1_bank3, cp2_boA_gconv2, 
    //     cp2_boAX_gconv2_bank1, cp2_boAX_gconv2_bank2, cp2_boAX_gconv2_bank3, 
    //     num_vertex_bank_gconv2, num_edge_block_gconv2, 
    //     f_block_len_custom_gconv2, imagescale_gconv2, 
    //     1, 1, 0, 0, num_vertex_bank_gconv2, 0,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     ); 
    
    // auto cp2_run_self_trans_gconv2 = cp2_krnl_trans(
    //     cp2_boW_gconv2_self,
    //     BM_gconv2, 0, 0, 0, 0,
    //     2, 2 // new parameter v11
    //     );
    // auto cp2_run_self_trans_Load1_gconv2 = cp2_load_mm1(cp2_boC_pooling1_bank1, BM_gconv2);
    // auto cp2_run_self_trans_Load2_gconv2 = cp2_load_mm2(cp2_boC_pooling1_bank2, BM_gconv2);
    // auto cp2_run_self_trans_Load3_gconv2 = cp2_load_mm3(cp2_boC_pooling1_bank3, BM_gconv2);
    // auto cp2_run_self_trans_Store1_gconv2 = cp2_store_mm1(cp2_boAX_Merge_gconv2_bank1, BM_gconv2);
    // auto cp2_run_self_trans_Store2_gconv2 = cp2_store_mm2(cp2_boAX_Merge_gconv2_bank2, BM_gconv2);
    // auto cp2_run_self_trans_Store3_gconv2 = cp2_store_mm3(cp2_boAX_Merge_gconv2_bank3, BM_gconv2);
    
    cp1_run_self_trans_Load1_gconv2.wait();
    cp1_run_self_trans_Load2_gconv2.wait();
    cp1_run_self_trans_Load3_gconv2.wait();
    cp1_run_self_trans_Store1_gconv2.wait();
    cp1_run_self_trans_Store2_gconv2.wait();
    cp1_run_self_trans_Store3_gconv2.wait();
    cp1_run_gconv2_agg.wait();
    cp1_run_self_trans_gconv2.wait();

    // cp2_run_self_trans_Load1_gconv2.wait();
    // cp2_run_self_trans_Load2_gconv2.wait();
    // cp2_run_self_trans_Load3_gconv2.wait();
    // cp2_run_self_trans_Store1_gconv2.wait();
    // cp2_run_self_trans_Store2_gconv2.wait();
    // cp2_run_self_trans_Store3_gconv2.wait();
    // cp2_run_gconv2_agg.wait();
    // cp2_run_self_trans_gconv2.wait();

    auto cp1_run_loaddata_gconv2 = cp1_krnl_feagg(
        cp1_boX_bank1, cp1_boX_bank2, cp1_boX_bank3, 
        cp1_boA_p2, 
        cp1_boAX_bank1, cp1_boAX_bank2, cp1_boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        2, BM_gconv2, 1, // v8 new parameters
        cp1_boAX_Merge_gconv2_bank1, cp1_boAX_Merge_gconv2_bank2,  cp1_boAX_Merge_gconv2_bank3  // v8 new parameters
    ); 
    auto cp1_run_neigh_trans_gconv2 = cp1_krnl_trans(
        cp1_boW_gconv2_neighbor,
        BM_gconv2, 1, 1, 1, 0,
        2, 3 // new parameter v11
        );

    auto cp1_run_neigh_trans_Load1_gconv2 = cp1_load_mm1(cp1_boAX_gconv2_bank1, BM_gconv2);
    auto cp1_run_neigh_trans_Load2_gconv2 = cp1_load_mm2(cp1_boAX_gconv2_bank2, BM_gconv2);
    auto cp1_run_neigh_trans_Load3_gconv2 = cp1_load_mm3(cp1_boAX_gconv2_bank3, BM_gconv2);
    auto cp1_run_neigh_trans_Store1_gconv2 = cp1_store_mm1(cp1_boC_gconv2_bank1, BM_gconv2);
    auto cp1_run_neigh_trans_Store2_gconv2 = cp1_store_mm2(cp1_boC_gconv2_bank2, BM_gconv2);
    auto cp1_run_neigh_trans_Store3_gconv2 = cp1_store_mm3(cp1_boC_gconv2_bank3, BM_gconv2);

    // auto cp2_run_loaddata_gconv2 = cp2_krnl_feagg(
    //     cp2_boX_bank1, cp2_boX_bank2, cp2_boX_bank3, 
    //     cp2_boA_p2, 
    //     cp2_boAX_bank1, cp2_boAX_bank2, cp2_boAX_bank3, 
    //     num_vertex_bank, num_edge_block/2,
    //     f_block_len_custom, imagescale,
    //     0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
    //     2, BM_gconv2, 1, // v8 new parameters
    //     cp2_boAX_Merge_gconv2_bank1, cp2_boAX_Merge_gconv2_bank2,  cp2_boAX_Merge_gconv2_bank3  // v8 new parameters
    // ); 
    // auto cp2_run_neigh_trans_gconv2 = cp2_krnl_trans(
    //     cp2_boW_gconv2_neighbor,
    //     BM_gconv2, 1, 1, 1, 0,
    //     2, 3 // new parameter v11
    //     );

    // auto cp2_run_neigh_trans_Load1_gconv2 = cp2_load_mm1(cp2_boAX_gconv2_bank1, BM_gconv2);
    // auto cp2_run_neigh_trans_Load2_gconv2 = cp2_load_mm2(cp2_boAX_gconv2_bank2, BM_gconv2);
    // auto cp2_run_neigh_trans_Load3_gconv2 = cp2_load_mm3(cp2_boAX_gconv2_bank3, BM_gconv2);
    // auto cp2_run_neigh_trans_Store1_gconv2 = cp2_store_mm1(cp2_boC_gconv2_bank1, BM_gconv2);
    // auto cp2_run_neigh_trans_Store2_gconv2 = cp2_store_mm2(cp2_boC_gconv2_bank2, BM_gconv2);
    // auto cp2_run_neigh_trans_Store3_gconv2 = cp2_store_mm3(cp2_boC_gconv2_bank3, BM_gconv2);

    cp1_run_neigh_trans_Load1_gconv2.wait();
    cp1_run_neigh_trans_Load2_gconv2.wait();
    cp1_run_neigh_trans_Load3_gconv2.wait();
    cp1_run_neigh_trans_Store1_gconv2.wait();
    cp1_run_neigh_trans_Store2_gconv2.wait();
    cp1_run_neigh_trans_Store3_gconv2.wait();
    cp1_run_loaddata_gconv2.wait();
    cp1_run_neigh_trans_gconv2.wait();

    // cp2_run_neigh_trans_Load1_gconv2.wait();
    // cp2_run_neigh_trans_Load2_gconv2.wait();
    // cp2_run_neigh_trans_Load3_gconv2.wait();
    // cp2_run_neigh_trans_Store1_gconv2.wait();
    // cp2_run_neigh_trans_Store2_gconv2.wait();
    // cp2_run_neigh_trans_Store3_gconv2.wait();
    // cp2_run_loaddata_gconv2.wait();
    // cp2_run_neigh_trans_gconv2.wait();

    // Execute pooling2 =====================================================================================
    // printf("Execute pooling2 =====================================================================================\n");
    auto cp1_run_poo2 = cp1_krnl_feagg(
        cp1_boC_gconv2_bank1, cp1_boC_gconv2_bank2, cp1_boC_gconv2_bank3, cp1_boA_pool2, 
        cp1_boC_pooling2_bank1, cp1_boC_pooling2_bank2, cp1_boC_pooling2_bank3, 
        num_vertex_pooling1/4, num_pool2_edges, 
        3, imagescale_gconv2, 
        1, 1, 0, 0, num_vertex_pooling2/4, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    // auto cp2_run_poo2 = cp2_krnl_feagg(
    //     cp2_boC_gconv2_bank1, cp2_boC_gconv2_bank2, cp2_boC_gconv2_bank3, cp2_boA_pool2, 
    //     cp2_boC_pooling2_bank1, cp2_boC_pooling2_bank2, cp2_boC_pooling2_bank3, 
    //     num_vertex_pooling1/4, num_pool2_edges, 
    //     3, imagescale_gconv2, 
    //     1, 1, 0, 0, num_vertex_pooling2/4, 1,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     ); 
    cp1_run_poo2.wait();
    // cp2_run_poo2.wait();

    // Execute gconv3 =========================================================================================
    // printf("Execute gconv3 =====================================================================================\n");
    auto cp1_run_gconv3_agg = cp1_krnl_feagg(
        cp1_boC_pooling2_bank1, cp1_boC_pooling2_bank2, cp1_boC_pooling2_bank3, cp1_boA_gconv3, 
        cp1_boAX_gconv3_bank1, cp1_boAX_gconv3_bank2, cp1_boAX_gconv3_bank3, 
        num_vertex_bank_gconv3, num_edge_block_gconv3, 
        f_block_len_custom_gconv3, imagescale_gconv3, 
        1, 1, 0, 0, num_vertex_bank_gconv3, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    auto cp1_run_self_trans_gconv3 = cp1_krnl_trans(
        cp1_boW_gconv3_self, 
        BM_gconv3, 0, 0, 0, 0,
        2, 4 // new parameter v11
        );
    auto cp1_run_self_trans_Load1_gconv3 = cp1_load_mm1(cp1_boC_pooling2_bank1, BM_gconv3);
    auto cp1_run_self_trans_Load2_gconv3 = cp1_load_mm2(cp1_boC_pooling2_bank2, BM_gconv3);
    auto cp1_run_self_trans_Load3_gconv3 = cp1_load_mm3(cp1_boC_pooling2_bank3, BM_gconv3);
    auto cp1_run_self_trans_Store1_gconv3 = cp1_store_mm1(cp1_boAX_Merge_gconv3_bank1, BM_gconv3);
    auto cp1_run_self_trans_Store2_gconv3 = cp1_store_mm2(cp1_boAX_Merge_gconv3_bank2, BM_gconv3);
    auto cp1_run_self_trans_Store3_gconv3 = cp1_store_mm3(cp1_boAX_Merge_gconv3_bank3, BM_gconv3);

    // auto cp2_run_gconv3_agg = cp2_krnl_feagg(
    //     cp2_boC_pooling2_bank1, cp2_boC_pooling2_bank2, cp2_boC_pooling2_bank3, cp2_boA_gconv3, 
    //     cp2_boAX_gconv3_bank1, cp2_boAX_gconv3_bank2, cp2_boAX_gconv3_bank3, 
    //     num_vertex_bank_gconv3, num_edge_block_gconv3, 
    //     f_block_len_custom_gconv3, imagescale_gconv3, 
    //     1, 1, 0, 0, num_vertex_bank_gconv3, 0,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     ); 
    // auto cp2_run_self_trans_gconv3 = cp2_krnl_trans(
    //     cp2_boW_gconv3_self, 
    //     BM_gconv3, 0, 0, 0, 0,
    //     2, 4 // new parameter v11
    //     );
    // auto cp2_run_self_trans_Load1_gconv3 = cp2_load_mm1(cp2_boC_pooling2_bank1, BM_gconv3);
    // auto cp2_run_self_trans_Load2_gconv3 = cp2_load_mm2(cp2_boC_pooling2_bank2, BM_gconv3);
    // auto cp2_run_self_trans_Load3_gconv3 = cp2_load_mm3(cp2_boC_pooling2_bank3, BM_gconv3);
    // auto cp2_run_self_trans_Store1_gconv3 = cp2_store_mm1(cp2_boAX_Merge_gconv3_bank1, BM_gconv3);
    // auto cp2_run_self_trans_Store2_gconv3 = cp2_store_mm2(cp2_boAX_Merge_gconv3_bank2, BM_gconv3);
    // auto cp2_run_self_trans_Store3_gconv3 = cp2_store_mm3(cp2_boAX_Merge_gconv3_bank3, BM_gconv3);

    cp1_run_self_trans_Load1_gconv3.wait();
    cp1_run_self_trans_Load2_gconv3.wait();
    cp1_run_self_trans_Load3_gconv3.wait();
    cp1_run_self_trans_Store1_gconv3.wait();
    cp1_run_self_trans_Store2_gconv3.wait();
    cp1_run_self_trans_Store3_gconv3.wait();
    cp1_run_gconv3_agg.wait();
    cp1_run_self_trans_gconv3.wait();
    
    // cp2_run_self_trans_Load1_gconv3.wait();
    // cp2_run_self_trans_Load2_gconv3.wait();
    // cp2_run_self_trans_Load3_gconv3.wait();
    // cp2_run_self_trans_Store1_gconv3.wait();
    // cp2_run_self_trans_Store2_gconv3.wait();
    // cp2_run_self_trans_Store3_gconv3.wait();
    // cp2_run_gconv3_agg.wait();
    // cp2_run_self_trans_gconv3.wait();

    auto cp1_run_loaddata_gconv3 = cp1_krnl_feagg(
        cp1_boX_bank1, cp1_boX_bank2, cp1_boX_bank3, 
        cp1_boA_p2, 
        cp1_boAX_bank1, cp1_boAX_bank2, cp1_boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        2, BM_gconv3, 1, // v8 new parameters
        cp1_boAX_Merge_gconv3_bank1, cp1_boAX_Merge_gconv3_bank2,  cp1_boAX_Merge_gconv3_bank3  // v8 new parameters
    ); 
    auto cp1_run_neigh_trans_gconv3 = cp1_krnl_trans(
        cp1_boW_gconv3_neighbor,
        BM_gconv3, 1, 1, 1, 0,
        2, 5 // new parameter v11
        );

    auto cp1_run_neigh_trans_Load1_gconv3 = cp1_load_mm1(cp1_boAX_gconv3_bank1, BM_gconv3);
    auto cp1_run_neigh_trans_Load2_gconv3 = cp1_load_mm2(cp1_boAX_gconv3_bank2, BM_gconv3);
    auto cp1_run_neigh_trans_Load3_gconv3 = cp1_load_mm3(cp1_boAX_gconv3_bank3, BM_gconv3);
    auto cp1_run_neigh_trans_Store1_gconv3 = cp1_store_mm1(cp1_boC_gconv3_bank1, BM_gconv3);
    auto cp1_run_neigh_trans_Store2_gconv3 = cp1_store_mm2(cp1_boC_gconv3_bank2, BM_gconv3);
    auto cp1_run_neigh_trans_Store3_gconv3 = cp1_store_mm3(cp1_boC_gconv3_bank3, BM_gconv3);

    // auto cp2_run_loaddata_gconv3 = cp2_krnl_feagg(
    //     cp2_boX_bank1, cp2_boX_bank2, cp2_boX_bank3, 
    //     cp2_boA_p2, 
    //     cp2_boAX_bank1, cp2_boAX_bank2, cp2_boAX_bank3, 
    //     num_vertex_bank, num_edge_block/2,
    //     f_block_len_custom, imagescale,
    //     0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
    //     2, BM_gconv3, 1, // v8 new parameters
    //     cp2_boAX_Merge_gconv3_bank1, cp2_boAX_Merge_gconv3_bank2,  cp2_boAX_Merge_gconv3_bank3  // v8 new parameters
    // ); 
    // auto cp2_run_neigh_trans_gconv3 = cp2_krnl_trans(
    //     cp2_boW_gconv3_neighbor,
    //     BM_gconv3, 1, 1, 1, 0,
    //     2, 5 // new parameter v11
    //     );

    // auto cp2_run_neigh_trans_Load1_gconv3 = cp2_load_mm1(cp2_boAX_gconv3_bank1, BM_gconv3);
    // auto cp2_run_neigh_trans_Load2_gconv3 = cp2_load_mm2(cp2_boAX_gconv3_bank2, BM_gconv3);
    // auto cp2_run_neigh_trans_Load3_gconv3 = cp2_load_mm3(cp2_boAX_gconv3_bank3, BM_gconv3);
    // auto cp2_run_neigh_trans_Store1_gconv3 = cp2_store_mm1(cp2_boC_gconv3_bank1, BM_gconv3);
    // auto cp2_run_neigh_trans_Store2_gconv3 = cp2_store_mm2(cp2_boC_gconv3_bank2, BM_gconv3);
    // auto cp2_run_neigh_trans_Store3_gconv3 = cp2_store_mm3(cp2_boC_gconv3_bank3, BM_gconv3);

    cp1_run_neigh_trans_Load1_gconv3.wait();
    cp1_run_neigh_trans_Load2_gconv3.wait();
    cp1_run_neigh_trans_Load3_gconv3.wait();
    cp1_run_neigh_trans_Store1_gconv3.wait();
    cp1_run_neigh_trans_Store2_gconv3.wait();
    cp1_run_neigh_trans_Store3_gconv3.wait();
    cp1_run_loaddata_gconv3.wait();
    cp1_run_neigh_trans_gconv3.wait();

    // cp2_run_neigh_trans_Load1_gconv3.wait();
    // cp2_run_neigh_trans_Load2_gconv3.wait();
    // cp2_run_neigh_trans_Load3_gconv3.wait();
    // cp2_run_neigh_trans_Store1_gconv3.wait();
    // cp2_run_neigh_trans_Store2_gconv3.wait();
    // cp2_run_neigh_trans_Store3_gconv3.wait();
    // cp2_run_loaddata_gconv3.wait();
    // cp2_run_neigh_trans_gconv3.wait();


    // Execute pooling3 =====================================================================================
    // printf("Execute pooling3 =====================================================================================\n");
    auto cp1_run_poo3 = cp1_krnl_feagg(
        cp1_boC_gconv3_bank1, cp1_boC_gconv3_bank2, cp1_boC_gconv3_bank3, cp1_boA_pool3, 
        cp1_boC_pooling3_bank1, cp1_boC_pooling3_bank2, cp1_boC_pooling3_bank3, 
        num_vertex_pooling2/4, num_pool3_edges, 
        3, imagescale_gconv3, 
        1, 1, 0, 0, num_vertex_pooling3/4, 1,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    // auto cp2_run_poo3 = cp2_krnl_feagg(
    //     cp2_boC_gconv3_bank1, cp2_boC_gconv3_bank2, cp2_boC_gconv3_bank3, cp2_boA_pool3, 
    //     cp2_boC_pooling3_bank1, cp2_boC_pooling3_bank2, cp2_boC_pooling3_bank3, 
    //     num_vertex_pooling2/4, num_pool3_edges, 
    //     3, imagescale_gconv3, 
    //     1, 1, 0, 0, num_vertex_pooling3/4, 1,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     ); 
    cp1_run_poo3.wait();
    // cp2_run_poo3.wait();

    // Execute gconv4 =========================================================================================
    // printf("Execute gconv4 =====================================================================================\n");
    auto cp1_run_gconv4_agg = cp1_krnl_feagg(
        cp1_boC_pooling3_bank1, cp1_boC_pooling3_bank2, cp1_boC_pooling3_bank3, cp1_boA_gconv4, 
        cp1_boAX_gconv4_bank1, cp1_boAX_gconv4_bank2, cp1_boAX_gconv4_bank3, 
        num_vertex_bank_gconv4, num_edge_block_gconv4, 
        f_block_len_custom_gconv4, imagescale_gconv4, 
        1, 1, 0, 0, num_vertex_bank_gconv4, 0,
        1, 0, 0, // v8 new parameters
        (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
        ); 
    auto cp1_run_self_trans_gconv4 = cp1_krnl_trans(
        cp1_boW_gconv4_self, 
        BM_gconv4, 0, 0, 0, 0,
        2, 6 // new parameter v11
        );

    auto cp1_run_self_trans_Load1_gconv4 = cp1_load_mm1(cp1_boC_pooling3_bank1, BM_gconv4);
    auto cp1_run_self_trans_Load2_gconv4 = cp1_load_mm2(cp1_boC_pooling3_bank2, BM_gconv4);
    auto cp1_run_self_trans_Load3_gconv4 = cp1_load_mm3(cp1_boC_pooling3_bank3, BM_gconv4);
    auto cp1_run_self_trans_Store1_gconv4 = cp1_store_mm1(cp1_boAX_Merge_gconv4_bank1, BM_gconv4);
    auto cp1_run_self_trans_Store2_gconv4 = cp1_store_mm2(cp1_boAX_Merge_gconv4_bank2, BM_gconv4);
    auto cp1_run_self_trans_Store3_gconv4 = cp1_store_mm3(cp1_boAX_Merge_gconv4_bank3, BM_gconv4);

    // auto cp2_run_gconv4_agg = cp2_krnl_feagg(
    //     cp2_boC_pooling3_bank1, cp2_boC_pooling3_bank2, cp2_boC_pooling3_bank3, cp2_boA_gconv4, 
    //     cp2_boAX_gconv4_bank1, cp2_boAX_gconv4_bank2, cp2_boAX_gconv4_bank3, 
    //     num_vertex_bank_gconv4, num_edge_block_gconv4, 
    //     f_block_len_custom_gconv4, imagescale_gconv4, 
    //     1, 1, 0, 0, num_vertex_bank_gconv4, 0,
    //     1, 0, 0, // v8 new parameters
    //     (v_float *) 0, (v_float *) 0, (v_float *) 0 // v8 new parameters
    //     ); 
    // auto cp2_run_self_trans_gconv4 = cp2_krnl_trans(
    //     cp2_boW_gconv4_self, 
    //     BM_gconv4, 0, 0, 0, 0,
    //     2, 6 // new parameter v11
    //     );

    // auto cp2_run_self_trans_Load1_gconv4 = cp2_load_mm1(cp2_boC_pooling3_bank1, BM_gconv4);
    // auto cp2_run_self_trans_Load2_gconv4 = cp2_load_mm2(cp2_boC_pooling3_bank2, BM_gconv4);
    // auto cp2_run_self_trans_Load3_gconv4 = cp2_load_mm3(cp2_boC_pooling3_bank3, BM_gconv4);
    // auto cp2_run_self_trans_Store1_gconv4 = cp2_store_mm1(cp2_boAX_Merge_gconv4_bank1, BM_gconv4);
    // auto cp2_run_self_trans_Store2_gconv4 = cp2_store_mm2(cp2_boAX_Merge_gconv4_bank2, BM_gconv4);
    // auto cp2_run_self_trans_Store3_gconv4 = cp2_store_mm3(cp2_boAX_Merge_gconv4_bank3, BM_gconv4);

    cp1_run_self_trans_Load1_gconv4.wait();
    cp1_run_self_trans_Load2_gconv4.wait();
    cp1_run_self_trans_Load3_gconv4.wait();
    cp1_run_self_trans_Store1_gconv4.wait();
    cp1_run_self_trans_Store2_gconv4.wait();
    cp1_run_self_trans_Store3_gconv4.wait();
    cp1_run_gconv4_agg.wait();
    cp1_run_self_trans_gconv4.wait();

    // cp2_run_self_trans_Load1_gconv4.wait();
    // cp2_run_self_trans_Load2_gconv4.wait();
    // cp2_run_self_trans_Load3_gconv4.wait();
    // cp2_run_self_trans_Store1_gconv4.wait();
    // cp2_run_self_trans_Store2_gconv4.wait();
    // cp2_run_self_trans_Store3_gconv4.wait();
    // cp2_run_gconv4_agg.wait();
    // cp2_run_self_trans_gconv4.wait();
    
    auto cp1_run_loaddata_gconv4 = cp1_krnl_feagg(
        cp1_boX_bank1, cp1_boX_bank2, cp1_boX_bank3, 
        cp1_boA_p2, 
        cp1_boAX_bank1, cp1_boAX_bank2, cp1_boAX_bank3, 
        num_vertex_bank, num_edge_block/2,
        f_block_len_custom, imagescale,
        0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
        2, BM_gconv4, 1, // v8 new parameters
        cp1_boAX_Merge_gconv4_bank1, cp1_boAX_Merge_gconv4_bank2,  cp1_boAX_Merge_gconv4_bank3 // v8 new parameters
    ); 
    auto cp1_run_neigh_trans_gconv4 = cp1_krnl_trans(
        cp1_boW_gconv4_neighbor,
        BM_gconv4, 1, 1, 1, 1,
        2, 7 // new parameter v11
        );

    auto cp1_run_neigh_trans_Load1_gconv4 = cp1_load_mm1(cp1_boAX_gconv4_bank1, BM_gconv4);
    auto cp1_run_neigh_trans_Load2_gconv4 = cp1_load_mm2(cp1_boAX_gconv4_bank2, BM_gconv4);
    auto cp1_run_neigh_trans_Load3_gconv4 = cp1_load_mm3(cp1_boAX_gconv4_bank3, BM_gconv4);
    auto cp1_mlp_run = xrt::run(cp1_krnl_mlp); 
    cp1_mlp_run.set_arg(0, cp1_weight_l1);
    cp1_mlp_run.set_arg(1, cp1_weight_l2);
    cp1_mlp_run.set_arg(2, 0);
    cp1_mlp_run.set_arg(3, cp1_result_holder);
    cp1_mlp_run.start(); 

    // auto cp2_run_loaddata_gconv4 = cp2_krnl_feagg(
    //     cp2_boX_bank1, cp2_boX_bank2, cp2_boX_bank3, 
    //     cp2_boA_p2, 
    //     cp2_boAX_bank1, cp2_boAX_bank2, cp2_boAX_bank3, 
    //     num_vertex_bank, num_edge_block/2,
    //     f_block_len_custom, imagescale,
    //     0, 1, num_vertex_bank*4, num_vertex_bank*4, num_vertex_bank, 0,
    //     2, BM_gconv4, 1, // v8 new parameters
    //     cp2_boAX_Merge_gconv4_bank1, cp2_boAX_Merge_gconv4_bank2,  cp2_boAX_Merge_gconv4_bank3 // v8 new parameters
    // ); 
    // auto cp2_run_neigh_trans_gconv4 = cp2_krnl_trans(
    //     cp2_boW_gconv4_neighbor,
    //     BM_gconv4, 1, 1, 1, 1,
    //     2, 7 // new parameter v11
    //     );

    // auto cp2_run_neigh_trans_Load1_gconv4 = cp2_load_mm1(cp2_boAX_gconv4_bank1, BM_gconv4);
    // auto cp2_run_neigh_trans_Load2_gconv4 = cp2_load_mm2(cp2_boAX_gconv4_bank2, BM_gconv4);
    // auto cp2_run_neigh_trans_Load3_gconv4 = cp2_load_mm3(cp2_boAX_gconv4_bank3, BM_gconv4);
    // auto cp2_mlp_run = xrt::run(cp2_krnl_mlp); 
    // cp2_mlp_run.set_arg(0, cp2_weight_l1);
    // cp2_mlp_run.set_arg(1, cp2_weight_l2);
    // cp2_mlp_run.set_arg(2, 0);
    // cp2_mlp_run.set_arg(3, cp2_result_holder);
    // cp2_mlp_run.start(); 

    cp1_run_neigh_trans_Load1_gconv4.wait();
    cp1_run_neigh_trans_Load2_gconv4.wait();
    cp1_run_neigh_trans_Load3_gconv4.wait();
    cp1_run_loaddata_gconv4.wait();
    cp1_run_neigh_trans_gconv4.wait();
    cp1_mlp_run.wait();
    
    // cp2_run_neigh_trans_Load1_gconv4.wait();
    // cp2_run_neigh_trans_Load2_gconv4.wait();
    // cp2_run_neigh_trans_Load3_gconv4.wait();
    // cp2_run_loaddata_gconv4.wait();
    // cp2_run_neigh_trans_gconv4.wait();
    // cp2_mlp_run.wait();
}

void accelerator::loadinput(std::string filename){
    v_float * cp1_boX_map_bank1 = cp1_boX_bank1.map<v_float *>();
    v_float * cp1_boX_map_bank2 = cp1_boX_bank2.map<v_float *>();
    v_float * cp1_boX_map_bank3 = cp1_boX_bank3.map<v_float *>();
    load_input_general(filename, cp1_boX_map_bank1, cp1_boX_map_bank2, cp1_boX_map_bank3, num_vertex);
    cp1_boX_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // boX_bank2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // boX_bank3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    v_float * cp2_boX_map_bank1 = cp2_boX_bank1.map<v_float *>();
    v_float * cp2_boX_map_bank2 = cp2_boX_bank2.map<v_float *>();
    v_float * cp2_boX_map_bank3 = cp2_boX_bank3.map<v_float *>();
    load_input_general(filename, cp2_boX_map_bank1, cp2_boX_map_bank2, cp2_boX_map_bank3, num_vertex);
    cp2_boX_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}


void accelerator::checkresult(std::string filename){
    cp1_result_holder.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    v_dt * cp1_result_holder_map = cp1_result_holder.map<v_dt *>();
    
    int cp1_maxi = 0;
    float cp1_resulti  = (float) cp1_result_holder_map->data[0];

    for(int i = 1; i < 10; i++){
        printf("data1[%d]: %f\n", i, (float) cp1_result_holder_map->data[i]);
        if((float) cp1_result_holder_map->data[i] > cp1_resulti){
            cp1_resulti = (float) cp1_result_holder_map->data[i];
            cp1_maxi = i;
        }
    }

    cp2_result_holder.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    v_dt * cp2_result_holder_map = cp2_result_holder.map<v_dt *>();
    
    int cp2_maxi = 0;
    float cp2_resulti  = (float) cp2_result_holder_map->data[0];

    for(int i = 1; i < 10; i++){
        printf("data2[%d]: %f\n", i, (float) cp2_result_holder_map->data[i]);
        if((float) cp2_result_holder_map->data[i] > cp2_resulti){
            cp2_resulti = (float) cp2_result_holder_map->data[i];
            cp2_maxi = i;
        }
    }


    // std::ifstream fin_After_fc2("../../Layer/data/intermediate/After_fc2.bin", std::ios::binary);
    std::ifstream fin_After_fc2(filename, std::ios::binary);
    float fdata_f;
    fin_After_fc2.read(reinterpret_cast<char*>(&fdata_f), sizeof(float));

    printf("Prediction label %d, ground truth label, %d\n", cp1_maxi, (int) fdata_f);
    printf("Prediction label %d, ground truth label, %d\n", cp2_maxi, (int) fdata_f);

    // for(int i = 0; i < 10; i++){
    //     fin_After_fc2.read(reinterpret_cast<char*>(&fdata_f), sizeof(float));
    //     printf("golden %f, myresult %f\n", fdata_f, (float) result_holder_map[0].data[i]);
    // }

}