#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include <datatype.h>

extern "C"{

void read_AX_fromDDR_bank(
    const v_dt *AX, 
    int block_row_num,
    hls::stream<ap_axiu<256, 0, 0, 0>> & AX_DDR
){
    #pragma HLS INTERFACE mode=axis register both port=AX_DDR
    #pragma HLS aggregate variable = AX

    #pragma HLS INTERFACE m_axi port = AX offset = slave bundle = gmem0

    #pragma HLS INTERFACE s_axilite port = AX bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control

    for(int n = 0; n < block_row_num; n ++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){ // load data from DDR/HBM
            #pragma HLS PIPELINE II=1 rewind
            #pragma HLS loop_flatten
            int index =  n * VDATA_SIZE + ii;
            v_dt tmpIn;
            tmpIn = AX[index];

            ap_axiu<256, 0, 0, 0> mydata;
            ap_uint<256> datapack;
            for(int k = 0; k < 16; k++){
                datapack.range((k + 1)*16 - 1, k*16) = *reinterpret_cast<ap_uint<16> *>(&tmpIn.data[k]);
            }
            mydata.data = datapack;

            AX_DDR << mydata;
        }
    }

}




}