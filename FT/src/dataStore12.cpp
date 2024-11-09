#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include <datatype.h>

extern "C"{

void store_output_matrix(
    v_dt * c, 
    int size,
    int mode,
    hls::stream<ap_axiu<512, 0, 0, 0>> & R_stream,
    hls::stream<ap_axiu<512, 0, 0, 0>> & fromlayerone
){
    #pragma HLS INTERFACE mode=axis register both port=R_stream
    #pragma HLS INTERFACE mode=axis register both port=fromlayerone
    #pragma HLS aggregate variable = c
    
    #pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem0

    #pragma HLS INTERFACE s_axilite port = c bundle = control
    #pragma HLS INTERFACE s_axilite port = size bundle = control
    #pragma HLS INTERFACE s_axilite port = mode bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control

    for(int n = 0; n < size; n++){
        #pragma HLS loop_flatten
        #pragma HLS PIPELINE II=1 rewind
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn

        ap_axiu<512, 0, 0, 0> mydata;
        
        if(mode == 1) R_stream >> mydata;
        else if(mode == 2) fromlayerone >> mydata;

        for(int k = 0; k < 16; k++){
            ap_uint<32> idata = mydata.data.range((k + 1)*32 - 1, k*32);
            tmpIn.data[k] = *reinterpret_cast<float *>(&idata);
        }

        c[n] = tmpIn;
        
    }
}


}