#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"


#include <datatype.h>
#include <feaggv7.hpp>
#include <mmultv7.hpp>


using namespace std;

extern "C"{



void feaggController(
    v_dt * X_bank1,
    v_dt * X_bank2,
    v_dt * X_bank3,
    v_edge * A,
    v_dt * AX_bank1,
    v_dt * AX_bank2,
    v_dt * AX_bank3,

    int numberOfPass,

    int num_vertex_bank_first,
    int num_edge_block_first,
    int feature_block_len_first,
    int imagedimension_first,
    int ignorehead_first,
    int ignoretail_first,
    int xoffset_first,
    int axoffset_first,
    int num_vertex_bank_store_first,
    int maxpooling_first,

    int num_vertex_bank_second, 
    int num_edge_block_second,
    int feature_block_len_second,
    int imagedimension_second,
    int ignorehead_second,
    int ignoretail_second,
    int xoffset_second,
    int axoffset_second,
    int num_vertex_bank_store_second,
    int maxpooling_second
){
    if(numberOfPass == 1){
        feagg_top(
            X_bank1, X_bank2, X_bank3, A,
            AX_bank1, AX_bank2, AX_bank3,
            num_vertex_bank_first, num_edge_block_first, feature_block_len_first,
            imagedimension_first, ignorehead_first, ignoretail_first, xoffset_first,
            axoffset_first, num_vertex_bank_store_first, maxpooling_first
        );
    }
    else if (numberOfPass == 2){
        feagg_top(
            X_bank1, X_bank2, X_bank3, A,
            AX_bank1, AX_bank2, AX_bank3,
            num_vertex_bank_first, num_edge_block_first, feature_block_len_first,
            imagedimension_first, ignorehead_first, ignoretail_first, xoffset_first,
            axoffset_first, num_vertex_bank_store_first, maxpooling_first
        );
        feagg_top(
            X_bank1, X_bank2, X_bank3, A,
            AX_bank1, AX_bank2, AX_bank3,
            num_vertex_bank_second, num_edge_block_second, feature_block_len_second,
            imagedimension_second, ignorehead_second, ignoretail_second, xoffset_second,
            axoffset_second, num_vertex_bank_store_second, maxpooling_second
        );
    }
}


void feaggPlusSelfTrans(
    v_dt * X_bank1,
    v_dt * X_bank2,
    v_dt * X_bank3,
    v_edge * A,
    v_dt * AX_bank1_FA,
    v_dt * AX_bank2_FA,
    v_dt * AX_bank3_FA,

    int numberOfPass,

    int num_vertex_bank_first,
    int num_edge_block_first,
    int feature_block_len_first,
    int imagedimension_first,
    int ignorehead_first,
    int ignoretail_first,
    int xoffset_first,
    int axoffset_first,
    int num_vertex_bank_store_first,
    int maxpooling_first,

    int num_vertex_bank_second, 
    int num_edge_block_second,
    int feature_block_len_second,
    int imagedimension_second,
    int ignorehead_second,
    int ignoretail_second,
    int xoffset_second,
    int axoffset_second,
    int num_vertex_bank_store_second,
    int maxpooling_second,

    const v_dt * AX_bank1_FT,             // Read-Only Matrix AX_bank1
    const v_dt * AX_bank2_FT,             // Read-Only Matrix AX_bank2
    const v_dt * AX_bank3_FT,             // Read-Only Matrix AX_bank3
    const v_dt * W,              // Read-Only Matrix W
    v_dt *c_bank1,                    // Output Result c = AX x W + AX_Merge in bank1
    v_dt *c_bank2,                    // Output Result c = AX x W + AX_Merge in bank2
    v_dt *c_bank3,                    // Output Result c = AX x W + AX_Merge in bank3

    int block_row_num,           //amount of block row
    int merge,
    int ReLu,
    int biasflag
){
    #pragma HLS dataflow
    feaggController(
        X_bank1, X_bank2, X_bank3, A,
        AX_bank1_FA, AX_bank2_FA, AX_bank3_FA,
        numberOfPass,

        num_vertex_bank_first, num_edge_block_first, feature_block_len_first,
        imagedimension_first, ignorehead_first, ignoretail_first,
        xoffset_first, axoffset_first, num_vertex_bank_store_first,
        maxpooling_first,

        num_vertex_bank_second, num_edge_block_second, feature_block_len_second,
        imagedimension_second, ignorehead_second, ignoretail_second,
        xoffset_second, axoffset_second, num_vertex_bank_store_second,
        maxpooling_second
    );
    mmult(
        AX_bank1_FT, AX_bank2_FT, AX_bank3_FT,
        (v_dt *) 0, (v_dt *) 0, (v_dt *) 0, W, 
        c_bank1, c_bank2, c_bank3, block_row_num, 0, ReLu, biasflag
    );
}


void gnntop(
    // interface and control signal for feature aggregation module
    v_dt * X_bank1,
    v_dt * X_bank2,
    v_dt * X_bank3,
    v_edge * A,
    v_dt * AX_bank1_FA,
    v_dt * AX_bank2_FA,
    v_dt * AX_bank3_FA,

    int numberOfPass,
    int mode,

    int num_vertex_bank_first,
    int num_edge_block_first,
    int feature_block_len_first,
    int imagedimension_first,
    int ignorehead_first,
    int ignoretail_first,
    int xoffset_first,
    int axoffset_first,
    int num_vertex_bank_store_first,
    int maxpooling_first,

    int num_vertex_bank_second, 
    int num_edge_block_second,
    int feature_block_len_second,
    int imagedimension_second,
    int ignorehead_second,
    int ignoretail_second,
    int xoffset_second,
    int axoffset_second,
    int num_vertex_bank_store_second,
    int maxpooling_second,

    // interface and control signal for feature transformation module

    const v_dt * AX_bank1_FT,             // Read-Only Matrix AX_bank1
    const v_dt * AX_bank2_FT,             // Read-Only Matrix AX_bank2
    const v_dt * AX_bank3_FT,             // Read-Only Matrix AX_bank3
    const v_dt * AX_Merge_bank1,       // Read-Only AX_Merge
    const v_dt * AX_Merge_bank2,       // Read-Only AX_Merge
    const v_dt * AX_Merge_bank3,       // Read-Only AX_Merge
    const v_dt * W,              // Read-Only Matrix W
    v_dt *c_bank1,                    // Output Result c = AX x W + AX_Merge in bank1
    v_dt *c_bank2,                    // Output Result c = AX x W + AX_Merge in bank2
    v_dt *c_bank3,                    // Output Result c = AX x W + AX_Merge in bank3


    int block_row_num,           //amount of block row
    int merge,
    int ReLu,
    int biasflag

){
    // interface aggregate
    #pragma HLS aggregate variable = X_bank1
    #pragma HLS aggregate variable = X_bank2
    #pragma HLS aggregate variable = X_bank3
    #pragma HLS aggregate variable = A
    #pragma HLS aggregate variable = AX_bank1_FA
    #pragma HLS aggregate variable = AX_bank2_FA
    #pragma HLS aggregate variable = AX_bank3_FA

    #pragma HLS aggregate variable = AX_bank1_FT
    #pragma HLS aggregate variable = AX_bank2_FT
    #pragma HLS aggregate variable = AX_bank3_FT
    #pragma HLS aggregate variable = AX_Merge_bank1
    #pragma HLS aggregate variable = AX_Merge_bank2
    #pragma HLS aggregate variable = AX_Merge_bank3
    #pragma HLS aggregate variable = W
    #pragma HLS aggregate variable = c_bank1
    #pragma HLS aggregate variable = c_bank2
    #pragma HLS aggregate variable = c_bank3

    // master interface

    #pragma HLS INTERFACE m_axi port = X_bank1 offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = X_bank2 offset = slave bundle = gmem1
    #pragma HLS INTERFACE m_axi port = X_bank3 offset = slave bundle = gmem2
    #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem3       
    #pragma HLS INTERFACE m_axi port = AX_bank1_FA offset = slave bundle = gmem4
    #pragma HLS INTERFACE m_axi port = AX_bank2_FA offset = slave bundle = gmem5
    #pragma HLS INTERFACE m_axi port = AX_bank3_FA offset = slave bundle = gmem6

    #pragma HLS INTERFACE m_axi port = AX_bank1_FT offset = slave bundle = gmem7
    #pragma HLS INTERFACE m_axi port = AX_bank2_FT offset = slave bundle = gmem8
    #pragma HLS INTERFACE m_axi port = AX_bank3_FT offset = slave bundle = gmem9
    #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem10
    #pragma HLS INTERFACE m_axi port = AX_Merge_bank1 offset = slave bundle = gmem4
    #pragma HLS INTERFACE m_axi port = AX_Merge_bank2 offset = slave bundle = gmem5
    #pragma HLS INTERFACE m_axi port = AX_Merge_bank3 offset = slave bundle = gmem6
    #pragma HLS INTERFACE m_axi port = c_bank1 offset = slave bundle = gmem11
    #pragma HLS INTERFACE m_axi port = c_bank2 offset = slave bundle = gmem12
    #pragma HLS INTERFACE m_axi port = c_bank3 offset = slave bundle = gmem13

    // define AXI lite port

    #pragma HLS INTERFACE s_axilite port = X_bank1 bundle = control
    #pragma HLS INTERFACE s_axilite port = X_bank2 bundle = control
    #pragma HLS INTERFACE s_axilite port = X_bank3 bundle = control
    #pragma HLS INTERFACE s_axilite port = A bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank1_FA bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank2_FA bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank3_FA bundle = control

    #pragma HLS INTERFACE s_axilite port = numberOfPass bundle=control
    #pragma HLS INTERFACE s_axilite port = mode bundle=control

    #pragma HLS INTERFACE s_axilite port = num_vertex_bank_first bundle=control
    #pragma HLS INTERFACE s_axilite port = num_edge_block_first bundle=control
    #pragma HLS INTERFACE s_axilite port = feature_block_len_first bundle=control
    #pragma HLS INTERFACE s_axilite port = imagedimension_first bundle=control
    #pragma HLS INTERFACE s_axilite port = ignorehead_first bundle=control
    #pragma HLS INTERFACE s_axilite port = ignoretail_first bundle=control
    #pragma HLS INTERFACE s_axilite port = xoffset_first bundle=control
    #pragma HLS INTERFACE s_axilite port = axoffset_first bundle=control
    #pragma HLS INTERFACE s_axilite port = num_vertex_bank_store_first bundle=control
    #pragma HLS INTERFACE s_axilite port = maxpooling_first bundle=control

    #pragma HLS INTERFACE s_axilite port = num_vertex_bank_second bundle=control
    #pragma HLS INTERFACE s_axilite port = num_edge_block_second bundle=control
    #pragma HLS INTERFACE s_axilite port = feature_block_len_second bundle=control
    #pragma HLS INTERFACE s_axilite port = imagedimension_second bundle=control
    #pragma HLS INTERFACE s_axilite port = ignorehead_second bundle=control
    #pragma HLS INTERFACE s_axilite port = ignoretail_second bundle=control
    #pragma HLS INTERFACE s_axilite port = xoffset_second bundle=control
    #pragma HLS INTERFACE s_axilite port = axoffset_second bundle=control
    #pragma HLS INTERFACE s_axilite port = num_vertex_bank_store_second bundle=control
    #pragma HLS INTERFACE s_axilite port = maxpooling_second bundle=control

    #pragma HLS INTERFACE s_axilite port = AX_bank1_FT bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank2_FT bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank3_FT bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_Merge_bank1 bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_Merge_bank2 bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_Merge_bank3 bundle = control
    #pragma HLS INTERFACE s_axilite port = W bundle = control
    #pragma HLS INTERFACE s_axilite port = c_bank1 bundle = control
    #pragma HLS INTERFACE s_axilite port = c_bank2 bundle = control
    #pragma HLS INTERFACE s_axilite port = c_bank3 bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control
    #pragma HLS INTERFACE s_axilite port = merge bundle = control
    #pragma HLS INTERFACE s_axilite port = ReLu bundle = control
    #pragma HLS INTERFACE s_axilite port = biasflag bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control


    if(mode == 0){ // perform aggregate and self transformation
        feaggPlusSelfTrans(
            X_bank1, X_bank2, X_bank3, A,
            AX_bank1_FA, AX_bank2_FA, AX_bank3_FA, numberOfPass,

            num_vertex_bank_first, num_edge_block_first, feature_block_len_first,
            imagedimension_first, ignorehead_first, ignoretail_first, xoffset_first, 
            axoffset_first, num_vertex_bank_store_first, maxpooling_first,

            num_vertex_bank_second, num_edge_block_second, feature_block_len_second, imagedimension_second,
            ignorehead_second, ignoretail_second, xoffset_second, axoffset_second, 
            num_vertex_bank_store_second, maxpooling_second,

            AX_bank1_FT, AX_bank2_FT, AX_bank3_FT, 
            W, c_bank1, c_bank2, c_bank3, 

            block_row_num, merge, ReLu, biasflag
        );
    }
    else if(mode == 1){  // perform neighbor transformation
        mmult(
            AX_bank1_FT, AX_bank2_FT, AX_bank3_FT,   
            AX_Merge_bank1, AX_Merge_bank2, AX_Merge_bank3, 
            W, c_bank1, c_bank2, c_bank3, 
            block_row_num, 1, ReLu, biasflag
        );
    }
}



}