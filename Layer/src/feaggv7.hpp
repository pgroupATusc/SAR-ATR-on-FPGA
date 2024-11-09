#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"


#include <datatype.h>


using namespace std;

extern "C"{



// define the function to load X
void loadX(
    v_dt * X, 
    v_dt InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK], 
    int num_vertex_bank,
    int fid,
    int imagedimension,
    int ignorehead,
    int ignoretail
){
    #pragma HLS FUNCTION_INSTANTIATE variable=X
    #pragma HLS FUNCTION_INSTANTIATE variable=InVFeature
    #pragma HLS FUNCTION_INSTANTIATE variable=fid
    //load X from the DDR memory

    for(int i = 0; i < NUM_FEA_BANK; i++){  
        for(int ii = 0; ii < num_vertex_bank + imagedimension * 2; ii++){  // need to load two extra redundant lines

            #pragma HLS PIPELINE II=1
            int index = i * num_vertex_bank + ii - imagedimension;

            if(ignorehead == 1 && index < 0){
                continue;
            } 

            v_dt iv = X[index];

            if(i == 0 && ii < imagedimension && ignorehead == 1){  // initialize the data for the first redundant row 
                for(int k = 0; k < VDATA_SIZE; k++){
                    InVFeature[i][ii].data[k] = 0;
                }
            }
            else if(i == (NUM_FEA_BANK - 1) && ii >= num_vertex_bank + imagedimension && ignoretail == 1){
                for(int k = 0; k < VDATA_SIZE; k++){
                    InVFeature[i][ii].data[k] = 0;
                }
            }
            else{
                InVFeature[i][ii] = iv;
            }
        }
    }

}

// initialize the data in InitializeAX
void InitializeAX(
    v_dt OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK], 
    int fid,
    int num_vertex_bank
){
    #pragma HLS FUNCTION_INSTANTIATE variable=fid

    for(int i = 0; i < NUM_FEA_BANK; i++){  
        for(int ii = 0; ii < num_vertex_bank; ii++){
            #pragma HLS PIPELINE II=1
                int index = i * num_vertex_bank + ii;
                for(int ff = 0; ff < VDATA_SIZE; ff++){
                    OutVFeature[i][ii].data[ff] = 0;
                }
        }
    }
}


void workinit(
    v_dt * X, 
    v_dt InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK],
    v_dt OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK],
    int fid,
    int num_vertex_bank,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail
){
    #pragma HLS FUNCTION_INSTANTIATE variable=fid
    #pragma HLS dataflow

    loadX(X, InVFeature, num_vertex_bank, fid, imagedimension, ignorehead, ignoretail);
    InitializeAX(OutVFeature, fid, num_vertex_bank);
}



// define the function to load A
void loadA(
    v_edge * A,
    hls::stream<v_edge> & A_stream1, 
    hls::stream<v_edge> & A_stream2,
    hls::stream<v_edge> & A_stream3,  
    int num_edge_block
){
    #pragma HLS FUNCTION_INSTANTIATE variable=A
    for(int i = 0; i < num_edge_block; i++){
        #pragma HLS PIPELINE II=1 rewind
        v_edge tmpv_edge = A[i];
        A_stream1 << tmpv_edge;
        A_stream2 << tmpv_edge;
        A_stream3 << tmpv_edge;
    }
}

// define the computation unit
void Compute(
    v_dt InVFeature[NUM_VER_PER_BANK], 
    hls::stream<edge> & A_stream, 
    v_dt OutVFeature[NUM_VER_PER_BANK], 
    int num_edge_block,
    int imagedimension,
    int maxpooling
){
    #pragma HLS FUNCTION_INSTANTIATE variable=A_stream
    #pragma HLS FUNCTION_INSTANTIATE variable=InVFeature
    #pragma HLS FUNCTION_INSTANTIATE variable=OutVFeature

    for(int i = 0; i < num_edge_block; i++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS dependence variable=OutVFeature type=inter false

        edge iedge = A_stream.read();
        int src = iedge.src + imagedimension;
        int dst = iedge.dst;
        float value = iedge.value;
        int flag = iedge.flag;
        
        v_dt mysrcvf = InVFeature[src];
        v_dt mydstvf = OutVFeature[dst];
        for(int jj = 0; jj < VDATA_SIZE; jj++){
            if(flag == 1){  // (flag == 1) means this is a valid edge, we will skip the invalid edge
                if(maxpooling == 1){
                    mydstvf.data[jj] = max(mydstvf.data[jj], mysrcvf.data[jj]);
                }
                else{
                    mydstvf.data[jj] += value * mysrcvf.data[jj];
                }
            }
        }
        OutVFeature[dst] = mydstvf;
        
    }

}


// define the function to load A
void loadA_ind(
    hls::stream<v_edge> & A,
    hls::stream<edge> A_stream[NUM_FEA_BANK], 
    int num_edge_block
){
    #pragma HLS FUNCTION_INSTANTIATE variable=A
    #pragma HLS FUNCTION_INSTANTIATE variable=A_stream


    for(int i = 0; i < num_edge_block; i++){
        #pragma HLS PIPELINE II=1
        v_edge tmpv_edge = A.read();
        for(int j = 0; j < NUM_FEA_BANK; j++){
            edge iedge = tmpv_edge.data[j];
            A_stream[j] << iedge;
        }
    }
}


void working(
    hls::stream<v_edge> & A,
    v_dt InVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK],
    v_dt OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK],
    int num_edge_block,
    int imagedimension,
    int maxpooling
){
    #pragma HLS dataflow
    #pragma HLS FUNCTION_INSTANTIATE variable=A
    #pragma HLS FUNCTION_INSTANTIATE variable=InVFeature
    #pragma HLS FUNCTION_INSTANTIATE variable=OutVFeature
    
    hls::stream<edge> A_stream[NUM_FEA_BANK];
    #pragma HLS STREAM variable=A_stream depth=128
    #pragma HLS ARRAY_PARTITION variable=A_stream dim=1 complete

    loadA_ind(A, A_stream, num_edge_block);

    for(int i = 0; i < NUM_FEA_BANK; i++){
        #pragma HLS unroll
        Compute(InVFeature[i], A_stream[i], OutVFeature[i], num_edge_block, imagedimension, maxpooling);
    }
}



// define the function to store AX

void StoreAX(
    v_dt * AX, 
    v_dt OutVFeature[NUM_FEA_BANK][NUM_VER_PER_BANK], 
    int num_vertex_bank, 
    int feature_block_len
){
    #pragma HLS FUNCTION_INSTANTIATE variable=AX
    #pragma HLS FUNCTION_INSTANTIATE variable=OutVFeature

    for(int i = 0; i < NUM_FEA_BANK; i++){
        for(int ii = 0; ii < num_vertex_bank; ii++){
            #pragma HLS loop_flatten
            #pragma HLS PIPELINE II=1 rewind
            int index = i * num_vertex_bank + ii;
            AX[index] = OutVFeature[i][ii];
        }
    }
}



void feagg1(
    v_dt * X,
    hls::stream<v_edge> & A,
    int fid,
    v_dt * AX,
    int num_vertex_bank,
    int num_edge_block,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail,
    int xoffset,
    int axoffset,
    int num_vertex_bank_store,
    int maxpooling
){
    // define the buffer to store the vertex features
    v_dt InVFeature1[NUM_FEA_BANK][NUM_VER_PER_BANK];
    #pragma HLS resource variable=InVFeature1 core=RAM_2P_URAM
    #pragma HLS aggregate variable=InVFeature1
    #pragma HLS ARRAY_PARTITION variable=InVFeature1 dim=1 complete

    // define the buffer to store the results
    v_dt OutVFeature1[NUM_FEA_BANK][NUM_VER_PER_BANK];
    #pragma HLS resource variable=OutVFeature1 core=RAM_2P_URAM
    #pragma HLS aggregate variable=OutVFeature1
    #pragma HLS ARRAY_PARTITION variable=OutVFeature1 dim=1 complete

    v_dt * newX = X + xoffset;
    v_dt * newAX = AX + axoffset;

    workinit(newX, InVFeature1, OutVFeature1, fid, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    working(A, InVFeature1, OutVFeature1, num_edge_block, imagedimension, maxpooling);
    StoreAX(newAX, OutVFeature1, num_vertex_bank_store, feature_block_len);
}


void feagg2(
    v_dt * X,
    hls::stream<v_edge> & A,
    int fid,
    v_dt * AX,
    int num_vertex_bank,
    int num_edge_block,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail,
    int xoffset,
    int axoffset,
    int num_vertex_bank_store,
    int maxpooling
){
    // define the buffer to store the vertex features
    v_dt InVFeature2[NUM_FEA_BANK][NUM_VER_PER_BANK];
    #pragma HLS resource variable=InVFeature2 core=RAM_2P_URAM
    #pragma HLS aggregate variable=InVFeature2
    #pragma HLS ARRAY_PARTITION variable=InVFeature2 dim=1 complete

    // define the buffer to store the results
    v_dt OutVFeature2[NUM_FEA_BANK][NUM_VER_PER_BANK];
    #pragma HLS resource variable=OutVFeature2 core=RAM_2P_URAM
    #pragma HLS aggregate variable=OutVFeature2
    #pragma HLS ARRAY_PARTITION variable=OutVFeature2 dim=1 complete

    v_dt * newX = X + xoffset;
    v_dt * newAX = AX + axoffset;

    workinit(newX, InVFeature2, OutVFeature2, fid, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    working(A, InVFeature2, OutVFeature2, num_edge_block, imagedimension, maxpooling);
    StoreAX(newAX, OutVFeature2, num_vertex_bank_store, feature_block_len);
}


void feagg3(
    v_dt * X,
    hls::stream<v_edge> & A,
    int fid,
    v_dt * AX,
    int num_vertex_bank,
    int num_edge_block,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail,
    int xoffset,
    int axoffset,
    int num_vertex_bank_store,
    int maxpooling
){
    // define the buffer to store the vertex features
    v_dt InVFeature3[NUM_FEA_BANK][NUM_VER_PER_BANK];
    #pragma HLS resource variable=InVFeature3 core=RAM_2P_URAM
    #pragma HLS aggregate variable=InVFeature3
    #pragma HLS ARRAY_PARTITION variable=InVFeature3 dim=1 complete

    // define the buffer to store the results
    v_dt OutVFeature3[NUM_FEA_BANK][NUM_VER_PER_BANK];
    #pragma HLS resource variable=OutVFeature3 core=RAM_2P_URAM
    #pragma HLS aggregate variable=OutVFeature3
    #pragma HLS ARRAY_PARTITION variable=OutVFeature3 dim=1 complete

    v_dt * newX = X + xoffset;
    v_dt * newAX = AX + axoffset;

    workinit(newX, InVFeature3, OutVFeature3, fid, num_vertex_bank, feature_block_len, imagedimension, ignorehead, ignoretail);
    working(A, InVFeature3, OutVFeature3, num_edge_block, imagedimension, maxpooling);
    StoreAX(newAX, OutVFeature3, num_vertex_bank_store, feature_block_len);
}


void feagg_top(
    v_dt * X_bank1,
    v_dt * X_bank2,
    v_dt * X_bank3,
    v_edge * A,
    v_dt * AX_bank1,
    v_dt * AX_bank2,
    v_dt * AX_bank3,
    int num_vertex_bank, 
    int num_edge_block,
    int feature_block_len,
    int imagedimension,
    int ignorehead,
    int ignoretail,
    int xoffset,
    int axoffset,
    int num_vertex_bank_store,
    int maxpooling
){
    // #pragma HLS aggregate variable = X_bank1
    // #pragma HLS aggregate variable = X_bank2
    // #pragma HLS aggregate variable = X_bank3
    // #pragma HLS aggregate variable = A
    // #pragma HLS aggregate variable = AX_bank1
    // #pragma HLS aggregate variable = AX_bank2
    // #pragma HLS aggregate variable = AX_bank3
    // #pragma HLS INTERFACE m_axi port = X_bank1 offset = slave bundle = gmem0
    // #pragma HLS INTERFACE m_axi port = X_bank2 offset = slave bundle = gmem1
    // #pragma HLS INTERFACE m_axi port = X_bank3 offset = slave bundle = gmem2
    // #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem3       
    // #pragma HLS INTERFACE m_axi port = AX_bank1 offset = slave bundle = gmem4
    // #pragma HLS INTERFACE m_axi port = AX_bank2 offset = slave bundle = gmem5
    // #pragma HLS INTERFACE m_axi port = AX_bank3 offset = slave bundle = gmem6


    // #pragma HLS INTERFACE s_axilite port = X_bank1 bundle = control
    // #pragma HLS INTERFACE s_axilite port = X_bank2 bundle = control
    // #pragma HLS INTERFACE s_axilite port = X_bank3 bundle = control
    // #pragma HLS INTERFACE s_axilite port = A bundle = control
    // #pragma HLS INTERFACE s_axilite port = AX_bank1 bundle = control
    // #pragma HLS INTERFACE s_axilite port = AX_bank2 bundle = control
    // #pragma HLS INTERFACE s_axilite port = AX_bank3 bundle = control
    // #pragma HLS INTERFACE s_axilite port = num_vertex_bank bundle = control
    // #pragma HLS INTERFACE s_axilite port = num_edge_block bundle = control
    // #pragma HLS INTERFACE s_axilite port = feature_block_len bundle = control
    // #pragma HLS INTERFACE s_axilite port = imagedimension bundle = control
    // #pragma HLS INTERFACE s_axilite port = ignorehead bundle = control
    // #pragma HLS INTERFACE s_axilite port = ignoretail bundle = control
    // #pragma HLS INTERFACE s_axilite port = xoffset bundle = control
    // #pragma HLS INTERFACE s_axilite port = axoffset bundle = control
    // #pragma HLS INTERFACE s_axilite port = num_vertex_bank_store bundle = control
    // #pragma HLS INTERFACE s_axilite port = maxpooling bundle = control

    // #pragma HLS INTERFACE s_axilite port = return bundle = control

    #pragma HLS dataflow


    static hls::stream<v_edge> A_stream1;
    #pragma HLS STREAM variable=A_stream1 depth=128

    static hls::stream<v_edge> A_stream2;
    #pragma HLS STREAM variable=A_stream2 depth=128

    static hls::stream<v_edge> A_stream3;
    #pragma HLS STREAM variable=A_stream3 depth=128

    loadA(A, A_stream1, A_stream2, A_stream3, num_edge_block);

    // for(int i = 0; i < F_BLOCK_LEN; i++){
    feagg1(X_bank1, A_stream1, 0, 
        AX_bank1, num_vertex_bank, num_edge_block, 
        feature_block_len, imagedimension, 
        ignorehead, ignoretail, xoffset, axoffset,
        num_vertex_bank_store, maxpooling);
    feagg2(X_bank2, A_stream2, 1,
        AX_bank2, num_vertex_bank, num_edge_block, 
        feature_block_len, imagedimension, 
        ignorehead, ignoretail, xoffset, axoffset,
        num_vertex_bank_store, maxpooling);
    feagg3(X_bank3, A_stream3, 2, 
        AX_bank3, num_vertex_bank, num_edge_block, 
        feature_block_len, imagedimension, 
        ignorehead, ignoretail, xoffset, axoffset,
        num_vertex_bank_store, maxpooling);

    // }

}

}