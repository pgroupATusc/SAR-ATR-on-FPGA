#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"


#include <datatype.h>


using namespace std;

extern "C"{

void read_block_matrix(const v_dt *AX, hls::stream<v_dt> &AX_stream, int block_row_num){

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

                    int index =  ((n + nn) * VDATA_SIZE + ii) * BLOCK_NUM + j;
                    tmpIn = AX[index];
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
                    int index =  ((n + nn) * VDATA_SIZE + ii) * BLOCK_NUM + j;
                    
                    tmpIn = AX[index];
                    for(int jj = 0; jj < VDATA_SIZE; jj++){
                        tmpIn.data[jj] = localAXblock[nn][j][ii][jj];
                    }
                    AX_stream << tmpIn;
                }
            }
        }
    }
}





void compute(hls::stream<v_dt> &AX_stream, hls::stream<v_dt> & C_stream, const float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE], int block_row_num){
    
    float localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE];
    //#pragma HLS resource variable=localAX core=RAM_2P_LUTRAM
    #pragma HLS bind_storage variable=localAX type=RAM_2P impl=LUTRAM
    #pragma HLS ARRAY_PARTITION variable=localAX dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=localAX dim=3 complete


    float localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
    //#pragma HLS resource variable=localC core=RAM_2P_URAM
    #pragma HLS bind_storage variable=localC type=RAM_2P impl=bram
    #pragma HLS ARRAY_PARTITION variable=localC dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=localC dim=4 complete

    for(int n = 0; n < block_row_num; n += BUFFER_BLOCK){ // iterate the blocks
        for(int j = 0 ; j < BLOCK_NUM; j++){
            // printf("\nload AX within hardware --------------------------------- \n");
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
            // printf("---------------------------------------------------------- \n");

            // for(int kk = 0; kk < VDATA_SIZE; kk++){
            //     for(int k = 0; k < BLOCK_NUM; k++){  // computation
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
                
}

void store_output_matrix(hls::stream<v_dt> & C_stream,  v_dt * c, int block_row_num){
    for(int n = 0; n < block_row_num; n++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int j = 0; j < BLOCK_NUM; j++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind
                v_dt tmpIn;
                #pragma HLS aggregate variable = tmpIn
                int index = (n * VDATA_SIZE + ii) * BLOCK_NUM + j;
                tmpIn = C_stream.read();
                c[index] = tmpIn;
            }
        }
    }
}


void loadweight(const v_dt *W, float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE])
{
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

}


void mmult(
    const v_dt *AX,             // Read-Only Matrix AX
    const v_dt *W,              // Read-Only Matrix W
    v_dt *c,                    // Output Result
    int block_row_num           //amount of block row
)
{
    // define the inte
    #pragma HLS aggregate variable = AX
    #pragma HLS aggregate variable = W
    #pragma HLS aggregate variable = c
    #pragma HLS INTERFACE m_axi port = AX offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem1       
    #pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem2
    #pragma HLS INTERFACE s_axilite port = AX bundle = control
    #pragma HLS INTERFACE s_axilite port = W bundle = control
    #pragma HLS INTERFACE s_axilite port = c bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    #pragma HLS dataflow  
        
    static hls::stream<v_dt> AX_stream;
    #pragma HLS STREAM variable=AX_stream depth=64

    static hls::stream<v_dt> C_stream;
    #pragma HLS STREAM variable=AX_stream depth=128
        
    float localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
    //#pragma HLS resource variable=localW core=RAM_2P_URAM 
    #pragma HLS bind_storage variable=localW type=RAM_2P impl=bram
    #pragma HLS ARRAY_PARTITION variable=localW dim=4 complete
    
               
    // read GCN weight
    loadweight(W, localW);

    read_block_matrix(AX, AX_stream,  block_row_num);

    compute(AX_stream, C_stream, localW, block_row_num);

    store_output_matrix(C_stream, c, block_row_num);
}
}