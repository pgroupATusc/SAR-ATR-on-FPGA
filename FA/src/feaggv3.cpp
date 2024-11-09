#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"


#include <datatype.h>


using namespace std;

extern "C"{



// define the function to load X
void loadX(
    v_float * X, 
    int f,
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
            // #pragma HLS loop_flatten
            #pragma HLS PIPELINE II=1 rewind
            int index = i * num_vertex_bank + ii - imagedimension;
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

// initialize the data in InitializeAX
void InitializeAX(
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_vertex_bank, 
    int feature_block_len
){
    // initialize the data in AX
    for(int i = 0; i < NUM_FEA_BANK; i++){  
        for(int ii = 0; ii < num_vertex_bank; ii++){
            #pragma HLS loop_flatten
            #pragma HLS PIPELINE II=1 rewind
            
            for(int f = 0; f < feature_block_len; f++){
                int index = (i * num_vertex_bank + ii) * feature_block_len + f;
                for(int ff = 0; ff < VDATA_SIZE; ff++){
                    OutVFeature[i][ii][f].data[ff] = 0;
                }
            }
        }
    }
}

void workinit(
    v_float * X_bank1, 
    v_float * X_bank2,
    v_float * X_bank3,  
    v_float InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN],
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN],
    int num_vertex_bank,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail
){
    #pragma HLS dataflow
    v_float * X_array[3];
    X_array[0] = X_bank1;
    X_array[1] = X_bank2;
    X_array[2] = X_bank3;

    for(int i = 0; i < F_BLOCK_LEN; i++){
        #pragma HLS unroll
        loadX(X_array[i], i, InVFeature, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    }

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


void StoreAX_bank(
    v_float * AX,
    int f,
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_vertex_bank
){
    for(int i = 0; i < NUM_FEA_BANK; i++){
        for(int ii = 0; ii < num_vertex_bank; ii++){
            #pragma HLS loop_flatten
            #pragma HLS PIPELINE II=1 rewind

            int index = i * num_vertex_bank + ii;
            AX[index] = OutVFeature[i][ii][f];
        }
    }
}



// define the function to store AX
void StoreAX(
    v_float * AX[],
    v_float OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK][F_BLOCK_LEN], 
    int num_vertex_bank, 
    int feature_block_len
){
    #pragma HLS dataflow

    for(int i = 0; i < F_BLOCK_LEN; i++){
        #pragma HLS unroll
        StoreAX_bank(AX[i], i, OutVFeature, num_vertex_bank);
    }
}



void feagg_top(
    v_float * X_bank1,
    v_float * X_bank2,
    v_float * X_bank3,
    const v_edge * A,
    v_float * AX_bank1,
    v_float * AX_bank2,
    v_float * AX_bank3,
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
    #pragma HLS aggregate variable = X_bank1
    #pragma HLS aggregate variable = X_bank2
    #pragma HLS aggregate variable = X_bank3
    #pragma HLS aggregate variable = A
    #pragma HLS aggregate variable = AX_bank1
    #pragma HLS aggregate variable = AX_bank2
    #pragma HLS aggregate variable = AX_bank3
    #pragma HLS INTERFACE m_axi port = X_bank1 offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = X_bank2 offset = slave bundle = gmem1
    #pragma HLS INTERFACE m_axi port = X_bank3 offset = slave bundle = gmem2
    #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem3       
    #pragma HLS INTERFACE m_axi port = AX_bank1 offset = slave bundle = gmem4
    #pragma HLS INTERFACE m_axi port = AX_bank2 offset = slave bundle = gmem5
    #pragma HLS INTERFACE m_axi port = AX_bank3 offset = slave bundle = gmem6


    #pragma HLS INTERFACE s_axilite port = X_bank1 bundle = control
    #pragma HLS INTERFACE s_axilite port = X_bank2 bundle = control
    #pragma HLS INTERFACE s_axilite port = X_bank3 bundle = control
    #pragma HLS INTERFACE s_axilite port = A bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank1 bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank2 bundle = control
    #pragma HLS INTERFACE s_axilite port = AX_bank3 bundle = control
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

    v_float * newX_bank1 = X_bank1 + xoffset;
    v_float * newX_bank2 = X_bank2 + xoffset;
    v_float * newX_bank3 = X_bank3 + xoffset;
    v_float * newAX_array[3];
    newAX_array[0] = AX_bank1 + axoffset;
    newAX_array[1] = AX_bank2 + axoffset;
    newAX_array[2] = AX_bank3 + axoffset;

    workinit(newX_bank1, newX_bank2, newX_bank3, InVFeature, OutVFeature, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    working(A, InVFeature, OutVFeature, num_edge_block, imagedimension);
    StoreAX(newAX_array, OutVFeature, num_vertex_bank, feature_block_len);
}


}

