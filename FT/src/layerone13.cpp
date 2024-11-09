#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>


#include <datatype13.h>



using namespace std;

extern "C"{


void layerone_read_block_matrix(
    hls::stream<ap_axiu<256, 0, 0, 0>> & AX_DDR_bank1, 
    hls::stream<datatype> &AX_stream,
    int block_row_num
){
    for(int n = 0; n < 128 * 128; n++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS loop_flatten
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn
        ap_axiu<256, 0, 0, 0> mydata;
        AX_DDR_bank1 >> mydata;
        ap_uint<16> idata = mydata.data.range(16 - 1, 0);
        datatype value = *reinterpret_cast<datatype *>(&idata);
        AX_stream <<  value;
    }
}



void layerone_MergeResult(
    datatype localW[BLOCK_NUM][VDATA_SIZE],
    hls::stream<datatype> &AX_stream,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank2,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank3,
    hls::stream<v_dt> & F_stream1,
    hls::stream<v_dt> & F_stream2,
    hls::stream<v_dt> & F_stream3,
    int block_row_num,
    int merge
){
    for(int n = 0; n < 128*128; n++){
        #pragma HLS loop_flatten
        #pragma HLS PIPELINE II=3 rewind

        datatype pixelvalue; 

        v_dt tmpIn1, tmpIn2, tmpIn3, tmp_input, tmpOut1, tmpOut2, tmpOut3;
        // #pragma HLS aggregate variable = tmpIn1
        // #pragma HLS aggregate variable = tmpIn2
        // #pragma HLS aggregate variable = tmpIn3
        // #pragma HLS aggregate variable = tmp_input
        // #pragma HLS aggregate variable = tmpOut1
        // #pragma HLS aggregate variable = tmpOut2
        // #pragma HLS aggregate variable = tmpOut3
        if(merge == 1){
            ap_axiu<256, 0, 0, 0> mydata1;  AX_Merge_Stream_bank1 >> mydata1;
            ap_axiu<256, 0, 0, 0> mydata2;  AX_Merge_Stream_bank2 >> mydata2;
            ap_axiu<256, 0, 0, 0> mydata3;  AX_Merge_Stream_bank3 >> mydata3;

            for(int k = 0; k < 16; k++){
                ap_uint<16> idata = mydata1.data.range((k + 1)*16 - 1, k*16);
                tmpIn1.data[k] = *reinterpret_cast<datatype *>(&idata);
            }
            for(int k = 0; k < 16; k++){
                ap_uint<16> idata = mydata2.data.range((k + 1)*16 - 1, k*16);
                tmpIn2.data[k] = *reinterpret_cast<datatype *>(&idata);
            }
            for(int k = 0; k < 16; k++){
                ap_uint<16> idata = mydata3.data.range((k + 1)*16 - 1, k*16);
                tmpIn3.data[k] = *reinterpret_cast<datatype *>(&idata);
            }
        }

        AX_stream >> pixelvalue;

        for(int jj = 0; jj < VDATA_SIZE; jj++){
            tmpOut1.data[jj] = pixelvalue * localW[0][jj];
            tmpOut2.data[jj] = pixelvalue * localW[1][jj];
            tmpOut3.data[jj] = pixelvalue * localW[2][jj];
            if(merge == 1){
                tmpOut1.data[jj] += tmpIn1.data[jj];
                tmpOut2.data[jj] += tmpIn2.data[jj];
                tmpOut3.data[jj] += tmpIn3.data[jj];
            }

            // printf("tmpOut1[%d][%d]: %.12f\n", n, jj, (float) tmpOut1.data[jj]);
            // printf("tmpOut2[%d][%d]: %.12f\n", n, jj, (float) tmpOut2.data[jj]);
            // printf("tmpOut3[%d][%d]: %.12f\n", n, jj, (float) tmpOut3.data[jj]);
        }
        F_stream1 << tmpOut1; F_stream2 << tmpOut2; F_stream3 << tmpOut3;
    }

}


void layerone_ApplyActivationBias(
    hls::stream<v_dt> & F_stream1,
    hls::stream<v_dt> & F_stream2,
    hls::stream<v_dt> & F_stream3,
    hls::stream<ap_axiu<256, 0, 0, 0>> & R_stream_bank1,
    hls::stream<ap_axiu<256, 0, 0, 0>> & R_stream_bank2,
    hls::stream<ap_axiu<256, 0, 0, 0>> & R_stream_bank3,
    int block_row_num,
    datatype bias[BLOCK_NUM][VDATA_SIZE],
    int ReLu,
    int biasflag
){
    for(int n = 0; n < 128*128; n++){
        #pragma HLS loop_flatten
        #pragma HLS PIPELINE II=3 rewind
        v_dt tmpIn1, tmpIn2, tmpIn3;
        // #pragma HLS aggregate variable = tmpIn1
        // #pragma HLS aggregate variable = tmpIn2
        // #pragma HLS aggregate variable = tmpIn3

        tmpIn1 = F_stream1.read(); tmpIn2 = F_stream2.read(); tmpIn3 = F_stream3.read();

        for(int jj = 0; jj < VDATA_SIZE; jj++){
            if(biasflag == 1){
                tmpIn1.data[jj] = tmpIn1.data[jj] + bias[0][jj];
                tmpIn2.data[jj] = tmpIn2.data[jj] + bias[1][jj];
                tmpIn3.data[jj] = tmpIn3.data[jj] + bias[2][jj];
            }
            if(ReLu == 1 && tmpIn1.data[jj] <= 0){ tmpIn1.data[jj] = 0;}
            if(ReLu == 1 && tmpIn2.data[jj] <= 0){ tmpIn2.data[jj] = 0;}
            if(ReLu == 1 && tmpIn3.data[jj] <= 0){ tmpIn3.data[jj] = 0;}

            // printf("tmpOut1[%d][%d]: %.12f\n", n, jj, (float) tmpIn1.data[jj]);
            // printf("tmpOut2[%d][%d]: %.12f\n", n, jj, (float) tmpIn2.data[jj]);
            // printf("tmpOut3[%d][%d]: %.12f\n", n, jj, (float) tmpIn3.data[jj]);
        }

        ap_axiu<256, 0, 0, 0> mydata1, mydata2, mydata3;
        ap_uint<256> datapack1, datapack2, datapack3;
        for(int k = 0; k < 16; k++){
            datapack1.range((k + 1)*16 - 1, k*16) = *reinterpret_cast<ap_uint<16> *>(&tmpIn1.data[k]);
            datapack2.range((k + 1)*16 - 1, k*16) = *reinterpret_cast<ap_uint<16> *>(&tmpIn2.data[k]);
            datapack3.range((k + 1)*16 - 1, k*16) = *reinterpret_cast<ap_uint<16> *>(&tmpIn3.data[k]);
        }
        mydata1.data = datapack1; mydata2.data = datapack2;  mydata3.data = datapack3;
        R_stream_bank1 << mydata1; R_stream_bank2 << mydata2; R_stream_bank3 << mydata3;
    }
}


void layerone_loadweight(
    const v_dt *W, 
    datatype localW[BLOCK_NUM][VDATA_SIZE],
    datatype bias[BLOCK_NUM][VDATA_SIZE],
    int biasflag
){
    int j = 0;
    for(int counter = 0; counter < BLOCK_NUM + BLOCK_NUM; counter ++){
        #pragma HLS loop_flatten
        #pragma HLS PIPELINE II=1 rewind

        if(j == BLOCK_NUM){j = 0;}
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn
        tmpIn = W[counter];

        if(counter < BLOCK_NUM){
            for (int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                localW[j][jj] = tmpIn.data[jj];
                // printf("weight[%d][%d]: %.12f\n", j, jj, (float) localW[j][jj]);
            } 
        }          
        else{
            for (int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                if(biasflag == 1) bias[j][jj] = tmpIn.data[jj];
                else bias[j][jj] = 0;
                // printf("bias[%d][%d]: %.12f\n", j, jj, (float) bias[j][jj]);
            }  
        }    
        j++;
    }

}



void layerone_startExecute(
    int block_row_num,           //amount of block row
    int merge,
    int ReLu,
    int biasflag,
    int layerSelect,
    datatype localW[2][BLOCK_NUM][VDATA_SIZE],
    datatype bias[2][BLOCK_NUM][VDATA_SIZE],
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank2,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank3,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_DDR_stream1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  R_stream1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  R_stream2,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  R_stream3
){
    #pragma HLS dataflow  
        
    static hls::stream<datatype> AX_stream;
    #pragma HLS STREAM variable=AX_stream depth=64 type=fifo

    static hls::stream<v_dt> F_stream[3];
    #pragma HLS STREAM variable=F_stream depth=16 type=fifo

    // printf("here 1\n");
    layerone_read_block_matrix(AX_DDR_stream1, AX_stream, block_row_num);
    layerone_MergeResult(localW[layerSelect], AX_stream, AX_Merge_Stream_bank1, AX_Merge_Stream_bank2, AX_Merge_Stream_bank3, F_stream[0], F_stream[1], F_stream[2], block_row_num, merge);
    layerone_ApplyActivationBias(F_stream[0], F_stream[1], F_stream[2], R_stream1, R_stream2, R_stream3, block_row_num, bias[layerSelect], ReLu, biasflag);
}



void layerone(
    const v_dt * W,              // Read-Only Matrix W
    int block_row_num,           //amount of block row
    int merge,
    int ReLu,
    int biasflag,
    int mode,
    int layerSelect,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank2,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_Merge_Stream_bank3,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  AX_DDR_stream1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  R_stream1,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  R_stream2,
    hls::stream<ap_axiu<256, 0, 0, 0>> &  R_stream3
){
    // define the inte
    #pragma HLS aggregate variable = W

    #pragma HLS INTERFACE mode=axis register both port=AX_Merge_Stream_bank1
    #pragma HLS INTERFACE mode=axis register both port=AX_Merge_Stream_bank2
    #pragma HLS INTERFACE mode=axis register both port=AX_Merge_Stream_bank3
    #pragma HLS INTERFACE mode=axis register both port=AX_DDR_stream1
    #pragma HLS INTERFACE mode=axis register both port=R_stream1
    #pragma HLS INTERFACE mode=axis register both port=R_stream2
    #pragma HLS INTERFACE mode=axis register both port=R_stream3

    #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem3

    #pragma HLS INTERFACE s_axilite port = W bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control
    #pragma HLS INTERFACE s_axilite port = merge bundle = control
    #pragma HLS INTERFACE s_axilite port = ReLu bundle = control
    #pragma HLS INTERFACE s_axilite port = biasflag bundle = control
    #pragma HLS INTERFACE s_axilite port = mode bundle = control
    #pragma HLS INTERFACE s_axilite port = layerSelect bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control

    static datatype localW[2][BLOCK_NUM][VDATA_SIZE];
    #pragma HLS bind_storage variable=localW  impl=bram latency=1
    #pragma HLS ARRAY_PARTITION variable=localW dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=localW dim=3 complete

    static datatype bias[2][BLOCK_NUM][VDATA_SIZE];
    #pragma HLS bind_storage variable=bias  impl=bram latency=1
    // #pragma HLS ARRAY_PARTITION variable=bias dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=bias dim=3 complete

    if(mode == 1){
        layerone_loadweight(W, localW[layerSelect], bias[layerSelect], biasflag);
    }
    else if(mode == 2){
        layerone_startExecute(
            block_row_num, merge, ReLu, biasflag, layerSelect, localW, bias,
            AX_Merge_Stream_bank1, AX_Merge_Stream_bank2, AX_Merge_Stream_bank3,
            AX_DDR_stream1, R_stream1, R_stream2, R_stream3
        );
    }
}

}