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
    int block_row_num,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & R_stream
){
    #pragma HLS INTERFACE mode=axis register both port=R_stream
    #pragma HLS aggregate variable = c
    
    #pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem0

    #pragma HLS INTERFACE s_axilite port = c bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control

    for(int n = 0; n < block_row_num; n++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            #pragma HLS loop_flatten
            #pragma HLS PIPELINE II=1 rewind
            v_dt tmpIn;
            #pragma HLS aggregate variable = tmpIn
            int index = n * VDATA_SIZE + ii;

            ap_axiu<sizeof(datatype)*8  * 16, 0, 0, 0> mydata;
            R_stream >> mydata;

            for(int k = 0; k < 16; k++){
                ap_uint<sizeof(datatype)*8 > idata = mydata.data.range((k + 1)*sizeof(datatype)*8  - 1, k*sizeof(datatype)*8 );
                tmpIn.data[k] = *reinterpret_cast<datatype *>(&idata);
            }

            c[index] = tmpIn;
        }
    }
}


}