#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"


#include <datatype.h>


using namespace std;

extern "C"{

void read_AX_fromDDR(const v_dt *AX, hls::stream<v_dt> & AX_DDR, int block_row_num){
    for(int n = 0; n < block_row_num; n ++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){ // load data from DDR/HBM
            for(int j = 0 ; j < BLOCK_NUM; j++){
                #pragma HLS PIPELINE II=1 rewind
                #pragma HLS loop_flatten
                int index =  (n * VDATA_SIZE + ii) * BLOCK_NUM + j;
                v_dt tmpIn;
                tmpIn = AX[index];
                AX_DDR << tmpIn;
            }
        }
    }

}

void read_block_matrix(hls::stream<v_dt> & AX_DDR, hls::stream<v_dt> &AX_stream, int block_row_num){

    // define the buffer to cache the input vertices
    float localAXblock[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
    #pragma HLS resource variable=localAXblock core=RAM_2P_URAM
    #pragma HLS ARRAY_PARTITION variable=localAXblock dim=4 complete
    
    for(int n = 0; n < block_row_num; n += BUFFER_BLOCK){
        for(int nn = 0; nn < BUFFER_BLOCK; nn++){
            for(int ii = 0; ii < VDATA_SIZE; ii++){ // load data from DDR/HBM
                for(int j = 0 ; j < BLOCK_NUM; j++){
                    #pragma HLS PIPELINE II=1 rewind
                    #pragma HLS loop_flatten
                    v_dt tmpIn;
                    #pragma HLS aggregate variable = tmpIn

                    tmpIn = AX_DDR.read();
                    for(int jj = 0; jj < VDATA_SIZE; jj++){
                        localAXblock[nn][j][ii][jj] = tmpIn.data[jj];
                    }
                    
                }
            }
        }
        
        for(int j = 0 ; j < BLOCK_NUM; j++){  // send the data to the computation
            for(int nn = 0; nn < BUFFER_BLOCK; nn++){
                for(int ii = 0; ii < VDATA_SIZE; ii++){
                    #pragma HLS PIPELINE II=1 rewind
                    #pragma HLS loop_flatten
                    v_dt tmpIn;
                    #pragma HLS aggregate variable = tmpIn
                    for(int jj = 0; jj < VDATA_SIZE; jj++){
                        tmpIn.data[jj] = localAXblock[nn][j][ii][jj];
                    }
                    AX_stream << tmpIn;
                }
            }
        }
    }
}


void LoadUnit(
    hls::stream<v_dt> &AX_stream,
    float localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE]
){
    for(int nn = 0; nn < BUFFER_BLOCK; nn++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){ // read a data block
            #pragma HLS PIPELINE II=1 rewind
            #pragma HLS loop_flatten

            v_dt tmpIn;
            #pragma HLS aggregate variable = tmpIn
            tmpIn = AX_stream.read();
            for(int jj = 0; jj < VDATA_SIZE; jj++){
                localAX[nn][ii][jj] = tmpIn.data[jj];
                // printf("%.0f ", localAX[ii][jj]);
            }
            // printf("\n");
        }
    }
}

void ComputeUnit(
    int j,
    float localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    float localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE],
    const float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
){
    int k = 0;
    int kk = 0;
    int nn = 0;
    for(int loop = 0; loop < VDATA_SIZE * BLOCK_NUM * BUFFER_BLOCK; loop++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS dependence variable=localC type=inter false

        if(nn == BUFFER_BLOCK){
            k++;
            nn = 0;
        }
        if(k == BLOCK_NUM){
            kk++;
            k = 0;
        }
        for(int jj = 0; jj < VDATA_SIZE; jj++){
            for(int ii = 0; ii < VDATA_SIZE; ii++){
                float tmp = localC[nn][k][ii][jj];
                if(j== 0 && kk == 0){
                    tmp = localAX[nn][ii][kk] * localW[j][k][kk][jj];
                }
                else{
                    tmp += localAX[nn][ii][kk] * localW[j][k][kk][jj];
                }
                localC[nn][k][ii][jj] = tmp;
            }
        }
        nn++;
    }
}

void LoadandCompute(
    hls::stream<v_dt> &AX_stream,
    float localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    const float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
){
    for(int j = 0 ; j < BLOCK_NUM; j++){
        #pragma HLS dataflow 

        float localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE];
        //#pragma HLS resource variable=localAX core=RAM_2P_LUTRAM
        #pragma HLS bind_storage variable=localAX type=RAM_2P impl=LUTRAM
        #pragma HLS ARRAY_PARTITION variable=localAX dim=2 complete
        #pragma HLS ARRAY_PARTITION variable=localAX dim=3 complete
        #pragma HLS STREAM variable=localAX type=pipo
        // printf("\nload AX within hardware --------------------------------- \n");
        // printf("---------------------------------------------------------- \n");
        LoadUnit(AX_stream, localAX);
        // for(int kk = 0; kk < VDATA_SIZE; kk++){
        //     for(int k = 0; k < BLOCK_NUM; k++){  // computation
        ComputeUnit(j, localC, localAX, localW);
    }
}

void StoreC(
    hls::stream<v_dt> & C_stream,
    float localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
){
    for(int nn = 0; nn < BUFFER_BLOCK; nn++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int j = 0; j < BLOCK_NUM; j++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind
                v_dt tmpIn;
                #pragma HLS aggregate variable = tmpIn
                for(int jj = 0; jj < VDATA_SIZE; jj++){
                    tmpIn.data[jj] = localC[nn][j][ii][jj];
                }
                C_stream << tmpIn;
            }
        }
    }
}


void compute(hls::stream<v_dt> &AX_stream, hls::stream<v_dt> & C_stream, const float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE], int block_row_num){

    for(int n = 0; n < block_row_num; n++){ // iterate the blocks
        #pragma HLS dataflow 

        float localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
        //#pragma HLS resource variable=localC core=RAM_2P_URAM
        #pragma HLS bind_storage variable=localC type=RAM_2P impl=bram
        #pragma HLS ARRAY_PARTITION variable=localC dim=3 complete
        #pragma HLS ARRAY_PARTITION variable=localC dim=4 complete
        #pragma HLS STREAM variable=localC type=pipo

        LoadandCompute(AX_stream, localC, localW);
        StoreC(C_stream, localC);
    }   
                
}


void MergeResult(
    const v_dt * AX_Merge,  
    hls::stream<v_dt> & C_stream,  
    hls::stream<v_dt> & F_stream,
    int block_row_num,
    int merge
){
    if(merge == 1){
        for(int n = 0; n < block_row_num; n++){
            for(int ii = 0; ii < VDATA_SIZE; ii++){
                for(int j = 0; j < BLOCK_NUM; j++){
                    #pragma HLS loop_flatten
                    #pragma HLS PIPELINE II=1 rewind
                    v_dt tmpIn1;
                    #pragma HLS aggregate variable = tmpIn1
                    v_dt tmpIn2;
                    #pragma HLS aggregate variable = tmpIn2
                    v_dt tmpIn3;
                    #pragma HLS aggregate variable = tmpIn3
                    int index = (n * VDATA_SIZE + ii) * BLOCK_NUM + j;
                    tmpIn1 = AX_Merge[index];
                    tmpIn2 = C_stream.read();
                    for(int jj = 0; jj < VDATA_SIZE; jj++){
                        tmpIn3.data[jj] = tmpIn1.data[jj] + tmpIn2.data[jj];
                    }
                    F_stream << tmpIn3;
                }
            }
        }
    }
    else{
        for(int n = 0; n < block_row_num; n++){
            for(int ii = 0; ii < VDATA_SIZE; ii++){
                for(int j = 0; j < BLOCK_NUM; j++){
                    #pragma HLS loop_flatten
                    #pragma HLS PIPELINE II=1 rewind
                    v_dt tmpIn1;
                    #pragma HLS aggregate variable = tmpIn1
                    tmpIn1 = C_stream.read();
                    F_stream << tmpIn1;
                }
            }
        }
    }
}

// void ApplyActivation(
//     hls::stream<v_dt> & F_stream,
//     hls::stream<v_dt> & R_stream,
//     int block_row_num,
//     int ReLu
// ){
//     if(ReLu == 1){
//         for(int n = 0; n < block_row_num; n++){
//             for(int ii = 0; ii < VDATA_SIZE; ii++){
//                 for(int j = 0; j < BLOCK_NUM; j++){
//                     #pragma HLS loop_flatten
//                     #pragma HLS PIPELINE II=1 rewind
//                     v_dt tmpIn1;
//                     #pragma HLS aggregate variable = tmpIn1
//                     tmpIn1 = F_stream.read();

//                     for(int jj = 0; jj < VDATA_SIZE; jj++){
//                         if(tmpIn1.data[jj] <= 0.0){
//                             tmpIn1.data[jj] = 0.0;
//                             printf("Applied Relu\n");
//                         }
//                     }
//                     R_stream << tmpIn1;
//                 }
//             }
//         }
//     }
//     else{
//         for(int n = 0; n < block_row_num; n++){
//             for(int ii = 0; ii < VDATA_SIZE; ii++){
//                 for(int j = 0; j < BLOCK_NUM; j++){
//                     #pragma HLS loop_flatten
//                     #pragma HLS PIPELINE II=1 rewind
//                     v_dt tmpIn1;
//                     #pragma HLS aggregate variable = tmpIn1
//                     tmpIn1 = F_stream.read();
//                     R_stream << tmpIn1;
//                 }
//             }
//         }
//     }
// }



void store_output_matrix(
    hls::stream<v_dt> & F_stream,  
    v_dt * c, 
    int block_row_num,
    float bias[BLOCK_NUM][VDATA_SIZE],
    int ReLu
){
    for(int n = 0; n < block_row_num; n++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int j = 0; j < BLOCK_NUM; j++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind
                v_dt tmpIn;
                #pragma HLS aggregate variable = tmpIn
                int index = (n * VDATA_SIZE + ii) * BLOCK_NUM + j;
                tmpIn = F_stream.read();
                for(int jj = 0; jj < VDATA_SIZE; jj++){
                    tmpIn.data[jj] = tmpIn.data[jj] + bias[j][jj];
                    if(ReLu == 1 && tmpIn.data[jj] <= 0){
                        tmpIn.data[jj] = 0;
                    }
                }
                c[index] = tmpIn;
            }
        }
    }
}


void loadweight(
    const v_dt *W, 
    float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    float bias[BLOCK_NUM][VDATA_SIZE]
){
    // printf("\n\nload weight --------------------------------------\n");
    for (int i = 0; i < BLOCK_NUM; i = i + 1){   
        for (int ii = 0; ii < VDATA_SIZE; ii = ii + 1){  // The first two for loops define the row numbers
            for (int j = 0; j < BLOCK_NUM; j = j + 1){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind
                v_dt tmpIn;
                #pragma HLS aggregate variable = tmpIn
                int index = (i * VDATA_SIZE + ii) * BLOCK_NUM + j;
                tmpIn = W[index];
                for (int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                    localW[i][j][ii][jj] = tmpIn.data[jj];
                    // printf("%.0f ", localW[i][j][ii][jj]);
                }               
            }
            // printf("\n");   
        }
    }
    // printf("finish load weight --------------------------------------\n\n");

    // load bias
    for(int i = 0; i < BLOCK_NUM; i++){
        int index = BLOCK_NUM*BLOCK_NUM*VDATA_SIZE + i;
        v_dt tmpIn = W[index];
        for (int ii = 0; ii < VDATA_SIZE; ii = ii + 1){
            bias[i][ii] = tmpIn.data[ii];
            // bias[i][ii] = 0;
            // printf("%.0f ", localW[i][j][ii][jj]);
        }  
    }
    // finish loading bias

}


void mmult(
    const v_dt * AX,             // Read-Only Matrix AX
    const v_dt * AX_Merge,       // Read-Only AX_Merge
    const v_dt * W,              // Read-Only Matrix W
    v_dt *c,                    // Output Result c = AX x W + AX_Merge
    int block_row_num,           //amount of block row
    int merge,
    int ReLu
)
{
    // define the inte
    #pragma HLS aggregate variable = AX
    #pragma HLS aggregate variable = AX_Merge
    #pragma HLS aggregate variable = W
    #pragma HLS aggregate variable = c
    #pragma HLS INTERFACE m_axi port = AX offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem1
    #pragma HLS INTERFACE m_axi port = AX_Merge offset = slave bundle = gmem2          
    #pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem3
    #pragma HLS INTERFACE s_axilite port = AX bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_Merge bundle = control
    #pragma HLS INTERFACE s_axilite port = W bundle = control
    #pragma HLS INTERFACE s_axilite port = c bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control
    #pragma HLS INTERFACE s_axilite port = merge bundle = control
    #pragma HLS INTERFACE s_axilite port = ReLu bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    #pragma HLS dataflow  
        
    static hls::stream<v_dt> AX_stream;
    #pragma HLS STREAM variable=AX_stream depth=64 type=fifo

    static hls::stream<v_dt> AX_DDR;
    #pragma HLS STREAM variable=AX_DDR depth=64 type=fifo

    static hls::stream<v_dt> C_stream;
    #pragma HLS STREAM variable=C_stream depth=16 type=fifo

    static hls::stream<v_dt> F_stream;
    #pragma HLS STREAM variable=C_stream depth=16 type=fifo

    // static hls::stream<v_dt> R_stream;
    // #pragma HLS STREAM variable=C_stream depth=16 type=fifo
        
    float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
    //#pragma HLS resource variable=localW core=RAM_2P_URAM 
    #pragma HLS bind_storage variable=localW type=RAM_2P impl=bram
    #pragma HLS ARRAY_PARTITION variable=localW dim=4 complete
    
    float bias[BLOCK_NUM][VDATA_SIZE];
    #pragma HLS aggregate variable = bias
               
    // const int block_row_num = 18;

    // read GCN weight
    loadweight(W, localW, bias);

    read_AX_fromDDR(AX, AX_DDR, block_row_num);

    read_block_matrix(AX_DDR, AX_stream,  block_row_num);

    compute(AX_stream, C_stream, localW, block_row_num/6);

    MergeResult(AX_Merge, C_stream, F_stream, block_row_num, merge);

    store_output_matrix(F_stream, c, block_row_num, bias, ReLu);
}
}
