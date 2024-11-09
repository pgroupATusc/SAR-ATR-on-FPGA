#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"


#include <datatype.h>


using namespace std;

extern "C"{



// define the function to load X
void loadX(
    const v_float * X, 
    v_float InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_vertex_bank,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail
){
    // load X from the DDR memory
    for(int i = 0; i < NUM_FEA_BANK; i++){  
        for(int ii = 0; ii < num_vertex_bank + imagedimension * 2; ii++){  // need to load two extra redundant lines
            for(int f = 0; f < feature_block_len; f++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind
                int index = (i * num_vertex_bank + ii - imagedimension) * feature_block_len + f;
                v_float iv = X[index];

                if(i == 0 && ii < imagedimension && ignorehead == 1){  // initialize the data for the first redundant row 
                    for(int k = 0; k < VDATA_SIZE; k++){
                        InVFeature[i][ii][f].data[k] = 0;
                    }
                }
                else if(i == (NUM_FEA_BANK - 1) && ii >= num_vertex_bank + imagedimension && ignoretail == 1){
                    for(int k = 0; k < VDATA_SIZE; k++){
                        InVFeature[i][ii][f].data[k] = 0;
                    }
                }
                else{
                    InVFeature[i][ii][f] = iv;
                }
            }
        }
    }

}

// initialize the data in InitializeAX
void InitializeAX(
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_vertex_bank, 
    int feature_block_len
){
    // initialize the data in AX
    for(int i = 0; i < NUM_FEA_BANK; i++){  
        for(int ii = 0; ii < num_vertex_bank; ii++){
            for(int f = 0; f < feature_block_len; f++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind

                int index = (i * num_vertex_bank + ii) * feature_block_len + f;
                for(int ff = 0; ff < VDATA_SIZE; ff++){
                    OutVFeature[i][ii][f].data[ff] = 0;
                }
            }
        }
    }
}

void workinit(const v_float * X, 
    v_float InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN],
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN],
    int num_vertex_bank,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail
){
    #pragma HLS dataflow
    loadX(X, InVFeature, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    InitializeAX(OutVFeature, num_vertex_bank, feature_block_len);
}



// define the function to load A
void loadA(
    const v_edge * A,
    hls::stream<edge> A_stream[NUM_FEA_BANK], 
    int num_edge_block
){
    for(int i = 0; i < num_edge_block; i++){
        #pragma HLS PIPELINE II=1 rewind
        v_edge tmpv_edge = A[i];
        for(int j = 0; j < NUM_FEA_BANK; j++){
            edge iedge = tmpv_edge.data[j];
            A_stream[j] << iedge;
        }
    }
}

// define the computation unit
void Compute(
    v_float InVFeature[NUM_VER_PER_BANK][F_BLOCK_LEN], 
    hls::stream<edge> & A_stream, 
    v_float OutVFeature[NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_edge_block,
    int imagedimension
){
    for(int i = 0; i < num_edge_block; i++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS dependence variable=OutVFeature type=inter false

        edge iedge = A_stream.read();
        int src = iedge.src + imagedimension;
        int dst = iedge.dst;
        float value = iedge.value;
        int flag = iedge.flag;
        for(int j = 0; j < F_BLOCK_LEN; j++){
            v_float mysrcvf = InVFeature[src][j];
            v_float mydstvf = OutVFeature[dst][j];
            for(int jj = 0; jj < VDATA_SIZE; jj++){
                if(flag == 1){  // (flag == 1) means this is a valid edge, we will skip the invalid edge
                    mydstvf.data[jj] += value * mysrcvf.data[jj];
                }
            }
            OutVFeature[dst][j] = mydstvf;
        }
    }

}

void working(
    const v_edge * A,
    v_float InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN],
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN],
    int num_edge_block,
    int imagedimension
){
    #pragma HLS dataflow
    
    static hls::stream<edge> A_stream[NUM_FEA_BANK];
    #pragma HLS STREAM variable=A_stream depth=128
    #pragma HLS ARRAY_PARTITION variable=A_stream dim=1 complete

    loadA(A, A_stream, num_edge_block);

    for(int i = 0; i < NUM_FEA_BANK; i++){
        #pragma HLS unroll
        Compute(InVFeature[i], A_stream[i], OutVFeature[i], num_edge_block, imagedimension);
    }
}



// define the function to store AX
void StoreAX(
    v_float * AX, 
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_vertex_bank, 
    int feature_block_len
){
    for(int i = 0; i < NUM_FEA_BANK; i++){
        for(int ii = 0; ii < num_vertex_bank; ii++){
            for(int f = 0; f < feature_block_len; f++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind

                int index = (i * num_vertex_bank + ii) * feature_block_len + f;
                AX[index] = OutVFeature[i][ii][f];
            }
        }
    }
}



void feagg_top(
    const v_float * X,
    const v_edge * A,
    v_float * AX,
    int num_vertex_bank,
    int num_edge_block,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail,
    int xoffset,
    int axoffset
){
    // define the interface
    #pragma HLS aggregate variable = X
    #pragma HLS aggregate variable = A
    #pragma HLS aggregate variable = AX
    #pragma HLS INTERFACE m_axi port = X offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem1       
    #pragma HLS INTERFACE m_axi port = AX offset = slave bundle = gmem2

    #pragma HLS INTERFACE s_axilite port = X bundle = control
    #pragma HLS INTERFACE s_axilite port = A bundle = control
    #pragma HLS INTERFACE s_axilite port = AX bundle = control
    #pragma HLS INTERFACE s_axilite port = num_vertex_bank bundle = control
    #pragma HLS INTERFACE s_axilite port = num_edge_block bundle = control
    #pragma HLS INTERFACE s_axilite port = feature_block_len bundle = control
    #pragma HLS INTERFACE s_axilite port = imagedimension bundle = control
    #pragma HLS INTERFACE s_axilite port = ignorehead bundle = control
    #pragma HLS INTERFACE s_axilite port = ignoretail bundle = control
     #pragma HLS INTERFACE s_axilite port = xoffset bundle = control
    #pragma HLS INTERFACE s_axilite port = axoffset bundle = control

    #pragma HLS INTERFACE s_axilite port = return bundle = control

    


    // define the buffer to store the vertex features
    v_float InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN];
    #pragma HLS resource variable=InVFeature core=RAM_2P_URAM
    #pragma HLS aggregate variable=InVFeature
    #pragma HLS ARRAY_PARTITION variable=InVFeature dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=InVFeature dim=3 complete


    // define the buffer to store the results
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN];
    #pragma HLS resource variable=OutVFeature core=RAM_2P_URAM
    #pragma HLS aggregate variable=OutVFeature
    #pragma HLS ARRAY_PARTITION variable=OutVFeature dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=OutVFeature dim=3 complete



    // loadX and InitializeAX can be executed in parallel under dataflow optimization

    const v_float * newX = X + xoffset * feature_block_len;
    v_float * newAX = AX + axoffset * feature_block_len;

    workinit(newX, InVFeature, OutVFeature, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    working(A, InVFeature, OutVFeature, num_edge_block, imagedimension);
    StoreAX(newAX, OutVFeature, num_vertex_bank, feature_block_len);
}


}

