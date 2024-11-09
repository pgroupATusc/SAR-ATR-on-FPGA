#include <stdio.h>
#include <iostream>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "ap_int.h"


// #include <datatype.h>

using namespace std;

extern "C"{

#define VDATA_SIZE 16
struct v_dt {
    float data[VDATA_SIZE];
};


void dataloader(
     const v_dt * input_port,
     hls::stream<ap_axiu<512, 0, 0, 0>> & C_stream
){
    #pragma HLS aggregate variable = input_port

    #pragma HLS INTERFACE mode=axis port=C_stream
    #pragma HLS INTERFACE mode=m_axi port = input_port offset = slave bundle = gmem0

    #pragma HLS INTERFACE s_axilite port = input_port bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    for(int i = 0; i < 256; i++){
        #pragma HLS PIPELINE II=1
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn

        tmpIn = input_port[i];
        ap_axiu<512, 0, 0, 0> mydata;
        ap_uint<512> datapack;

        for(int k = 0; k < 16; k++){
            datapack.range((k + 1)*32 - 1, k*32) = *reinterpret_cast<ap_uint<32> *>(&tmpIn.data[k]);
        }
        mydata.data = datapack;

        C_stream << mydata;
    }

}

}