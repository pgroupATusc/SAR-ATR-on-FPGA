#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include <datatype.h>


extern "C"{

// #define VDATA_SIZE 16
// struct v_dt {
//     datatype data[VDATA_SIZE];
// };

void loadweightL1(
    const v_dt * weight_port,
    v_dt_mlp weight[64][256][3], 
    float biasl1[64]
){
    int i = 0;
    int j = 0;
    int k = 0;
    int count = 0;

    for(int count = 0; count < 64 * 256 * 3 + 4; count++){
        #pragma HLS loop_flatten
        #pragma HLS PIPELINE II=1 rewind
        
        if(k == 3){
            j++; k = 0;
        }
        if(j == 256){
            i++; j = 0;
        }
        v_dt tmpIn = weight_port[count];

        if(count < 64 * 256 * 3){
            for(int hh = 0; hh < 16; hh++){   
                weight[i][j][k].data[hh] = (float) tmpIn.data[hh];
            }
        }
        else{
            int diff  = count - 64 * 256 * 3;
            for(int kk = 0; kk < 16; kk++){
                biasl1[diff*16 + kk] = (float) tmpIn.data[kk];
            }
        }
        k++;
    }
}

void loadweightL2(
    const v_dt * weight_portL2, 
    v_dt_mlp weightL2[64],
    v_dt_mlp & biasl2
){
    for(int i = 0; i < 65; i++){
        #pragma HLS PIPELINE II=1 rewind
        v_dt tmpIn = weight_portL2[i];
        #pragma HLS aggregate variable = tmpIn
        if(i < 64){
            for(int j = 0; j < 16; j++){
                weightL2[i].data[j] = (float) tmpIn.data[j];
            }
        }
        else{
            for(int j = 0; j < 16; j++){
                biasl2.data[j] = (float) tmpIn.data[j];
            }
        }
    }

}

void loaddata(
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & C_stream1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & C_stream2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & C_stream3,
    v_dt_mlp featurevector[256][3]
){
    for(int i = 0; i < 256; i++){
        #pragma HLS PIPELINE II=3

        ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0> mydata;
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn

        C_stream1 >> mydata;
        for(int k = 0; k < 16; k++){
            ap_uint<sizeof(datatype)*8 > idata = mydata.data.range((k + 1)*sizeof(datatype)*8  - 1, k*sizeof(datatype)*8 );
            tmpIn.data[k] = *reinterpret_cast<datatype *>(&idata);
            featurevector[i][0].data[k] = (float) tmpIn.data[k];
        }
        C_stream2 >> mydata;
        for(int k = 0; k < 16; k++){
            ap_uint<sizeof(datatype)*8 > idata = mydata.data.range((k + 1)*sizeof(datatype)*8  - 1, k*sizeof(datatype)*8 );
            tmpIn.data[k] = *reinterpret_cast<datatype *>(&idata);
            featurevector[i][1].data[k] = (float) tmpIn.data[k];
        }
        C_stream3 >> mydata;
        for(int k = 0; k < 16; k++){
            ap_uint<sizeof(datatype)*8 > idata = mydata.data.range((k + 1)*sizeof(datatype)*8  - 1, k*sizeof(datatype)*8 );
            tmpIn.data[k] = *reinterpret_cast<datatype *>(&idata);
            featurevector[i][2].data[k] = (float) tmpIn.data[k];
        }
    }

    for(int i = 0; i < 288 - 256; i++){
        #pragma HLS PIPELINE II=3
        ap_axiu<sizeof(datatype)*8  * 16, 0, 0, 0> mydata;
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn

        C_stream1 >> mydata;
        C_stream2 >> mydata;
        C_stream3 >> mydata;
    }
}


void computeL1(
    v_dt_mlp featurevector[256][3],
    v_dt_mlp weight[64][256][3],
    float biasl1[64],
    float results[64]
){

    for(int k = 0; k < 64; k++){
        #pragma HLS PIPELINE II=1 rewind
        results[k] = biasl1[k]; 
    }


    for(int i = 0; i < 256; i++){
        for(int j = 0; j < 3; j ++){
            for(int k = 0; k < 64; k++){
                #pragma HLS PIPELINE II=1 rewind
                #pragma HLS dependence variable=results type=inter false

                v_dt_mlp tmpIn1 = featurevector[i][j];

                v_dt_mlp tmpIn2 = weight[k][i][j];

                float stage1_p1 = tmpIn1.data[0]* tmpIn2.data[0]  + tmpIn1.data[1] * tmpIn2.data[1];
                float stage1_p2 = tmpIn1.data[2]* tmpIn2.data[2]  + tmpIn1.data[3] * tmpIn2.data[3];
                float stage1_p3 = tmpIn1.data[4]* tmpIn2.data[4]  + tmpIn1.data[5] * tmpIn2.data[5];
                float stage1_p4 = tmpIn1.data[6]* tmpIn2.data[6]  + tmpIn1.data[7] * tmpIn2.data[7];
                float stage1_p5 = tmpIn1.data[8]* tmpIn2.data[8]  + tmpIn1.data[9] * tmpIn2.data[9];
                float stage1_p6 = tmpIn1.data[10]*tmpIn2.data[10] + tmpIn1.data[11]*tmpIn2.data[11];
                float stage1_p7 = tmpIn1.data[12]*tmpIn2.data[12] + tmpIn1.data[13]*tmpIn2.data[13];
                float stage1_p8 = tmpIn1.data[14]*tmpIn2.data[14] + tmpIn1.data[15]*tmpIn2.data[15];

                float stage2_p1 = stage1_p1 + stage1_p2;
                float stage2_p2 = stage1_p3 + stage1_p4;
                float stage2_p3 = stage1_p5 + stage1_p6;
                float stage2_p4 = stage1_p7 + stage1_p8;

                float stage3_p1 = stage2_p1 + stage2_p2;
                float stage3_p2 = stage2_p3 + stage2_p4;

                float tmpresult = stage3_p1 + stage3_p2;
                
                results[k] += tmpresult;
            }
        }
    }

    for(int k = 0; k < 64; k++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS dependence variable=results type=inter false
        results[k] = results[k] < (float) 0.0?  (float) 0.0: results[k]; 
    }
}



void computeL2(
    float results[64], 
    v_dt_mlp weightL2[64], 
    v_dt_mlp & biasl2,
    v_dt_mlp & resultL2
){
    // for(int i = 0; i < 16; i++){
    //     #pragma HLS unroll
    //     resultL2.data[i] = 0;
    // }
    resultL2 = biasl2;


    for(int i = 0; i < 64; i++){
        #pragma HLS PIPELINE II=1
        float coeff = results[i];
        v_dt_mlp tmpIn = weightL2[i];
        for(int j = 0; j < 10; j++){
            resultL2.data[j] += coeff * tmpIn.data[j];
        }
    }

}


void storeResult(
    v_dt * result_port, 
    v_dt_mlp  & resultL2
){
    v_dt final_result;
    for(int i = 0; i < 16; i++){
        final_result.data[i] = (datatype) resultL2.data[i];
    }
    result_port[0] = final_result;
}



void mlp(
    const v_dt * weight_port,
    const v_dt * weight_portL2,
    int loadweight,
    v_dt * result_port,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & C_stream1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & C_stream2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & C_stream3
){
    #pragma HLS aggregate variable = weight_port
    #pragma HLS aggregate variable = weight_portL2
    #pragma HLS aggregate variable = result_port



    #pragma HLS INTERFACE mode=m_axi port = weight_port offset = slave bundle = gmem0
    #pragma HLS INTERFACE mode=m_axi port = weight_portL2 offset = slave bundle = gmem0
    #pragma HLS INTERFACE mode=m_axi port = result_port offset = slave bundle = gmem0
    #pragma HLS INTERFACE mode=axis port=C_stream1
    #pragma HLS INTERFACE mode=axis port=C_stream2
    #pragma HLS INTERFACE mode=axis port=C_stream3

    #pragma HLS INTERFACE s_axilite port = weight_port bundle = control
    #pragma HLS INTERFACE s_axilite port = weight_portL2 bundle = control
    #pragma HLS INTERFACE s_axilite port = result_port bundle = control
    #pragma HLS INTERFACE s_axilite port = loadweight bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control


    v_dt_mlp featurevector[256][3];
    #pragma HLS aggregate variable=featurevector
    #pragma HLS bind_storage variable=featurevector type=RAM_T2P impl=BRAM latency=1

    static v_dt_mlp weight[64][256][3];
    #pragma HLS bind_storage variable=weight type=RAM_T2P impl=URAM latency=1
    #pragma HLS aggregate variable=weight
    // #pragma HLS array_partition variable=weight type=cyclic factor=2  dim=1


    static v_dt_mlp weightL2[64];
    #pragma HLS bind_storage variable=weightL2 type=RAM_S2P impl=LUTRAM latency=1

    static float biasl1[64];
    #pragma HLS bind_storage variable=biasl1 type=RAM_S2P impl=LUTRAM latency=1

    float results[64];
    #pragma HLS bind_storage variable=results type=RAM_S2P impl=LUTRAM latency=1

    static v_dt_mlp biasl2;
    #pragma HLS bind_storage variable=biasl2 type=RAM_S2P impl=LUTRAM latency=1
    #pragma HLS aggregate variable = biasl2

    v_dt_mlp resultL2;
    #pragma HLS bind_storage variable=resultL2 type=RAM_S2P impl=LUTRAM latency=1

    if(loadweight == 1){
        loadweightL1(weight_port, weight, biasl1);
        loadweightL2(weight_portL2, weightL2, biasl2);
    }
    else{
        loaddata(C_stream1, C_stream2, C_stream3, featurevector);
        computeL1(featurevector, weight, biasl1, results);
        computeL2(results, weightL2, biasl2, resultL2);
        storeResult(result_port, resultL2);
    }
}





}