#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include <datatype.h>


using namespace std;

extern "C"{


void read_block_matrix(
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & AX_DDR_bank1, 
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & AX_DDR_bank2, 
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & AX_DDR_bank3, 
    hls::stream<v_dt> &AX_stream,
    int block_row_num,
    hls::stream<datatype> &RM_stream,
    int layerSelect
){

    // define the buffer to cache the input vertices
    datatype localAXblock[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
    #pragma HLS bind_storage variable=localAXblock type=RAM_T2P impl=URAM latency=1
    #pragma HLS ARRAY_PARTITION variable=localAXblock dim=4 complete



    for(int n = 0; n < block_row_num; n += BUFFER_BLOCK){
        for(int nn = 0; nn < BUFFER_BLOCK; nn++){
            for(int ii = 0; ii < VDATA_SIZE; ii++){ // load data from DDR/HBM
                for(int k = 0; k < 3; k++){
                    #pragma HLS PIPELINE II=1 rewind
                    #pragma HLS loop_flatten
                    v_dt tmpIn;
                    #pragma HLS aggregate variable = tmpIn
                    ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0> mydata;

                    if(k == 0) AX_DDR_bank1 >> mydata;
                    else if(k == 1) AX_DDR_bank2 >> mydata;
                    else if(k == 2) AX_DDR_bank3 >> mydata;

                    for(int jj = 0; jj < 16; jj++){
                        ap_uint<sizeof(datatype)*8 > idata = mydata.data.range((jj + 1)*sizeof(datatype)*8  - 1, jj*sizeof(datatype)*8 );
                        if(layerSelect >= 2) {localAXblock[nn][k][ii][jj] = *reinterpret_cast<datatype *>(&idata);}         // layer 3, 4, 5, 6, 7, 8
                        else if(layerSelect < 2 && jj == 0 && k == 0) {RM_stream << *reinterpret_cast<datatype *>(&idata);};    // layer 1 and 2
                    }
                }
            }
        }

        if(layerSelect >= 2){
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
    
}


void LoadUnit(
    hls::stream<v_dt> &AX_stream,
    datatype localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE]
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
    datatype localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    datatype localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE],
    const datatype localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
){
    int k = 0;
    int kk = 0;
    int nn = 0;
    for(int loop = 0; loop < VDATA_SIZE * BLOCK_NUM * BUFFER_BLOCK; loop++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS dependence variable=localC type=inter false
        #pragma HLS latency max=17

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
                datatype tmp = localC[nn][k][ii][jj];
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
    datatype localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    const datatype localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
){
    for(int j = 0 ; j < BLOCK_NUM; j++){
        #pragma HLS dataflow 

        datatype localAX[BUFFER_BLOCK][VDATA_SIZE][VDATA_SIZE];
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
    datatype localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
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


void compute(hls::stream<v_dt> &AX_stream, hls::stream<v_dt> & C_stream, const datatype localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE], int block_row_num, int layerSelect){

    if(layerSelect >= 2){
        for(int n = 0; n < block_row_num; n++){ // iterate the blocks
            #pragma HLS dataflow 

            datatype localC[BUFFER_BLOCK][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
            //#pragma HLS resource variable=localC core=RAM_2P_URAM
            #pragma HLS bind_storage variable=localC type=RAM_T2P impl=bram latency=1
            #pragma HLS ARRAY_PARTITION variable=localC dim=3 complete
            #pragma HLS ARRAY_PARTITION variable=localC dim=4 complete
            #pragma HLS STREAM variable=localC type=pipo

            LoadandCompute(AX_stream, localC, localW);
            StoreC(C_stream, localC);
        }   
    }
                
}




void MergeResult(
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank3,
    hls::stream<v_dt> & C_stream,  
    hls::stream<v_dt> & F_stream,
    int block_row_num,
    int merge,
    hls::stream<datatype> & RM_stream,
    int layerSelect,
    datatype localW[BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE]
){
    // if(merge == 1){
    datatype pixevalue[3];
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

                // printf("here in MergeResult %d, %d, %d\n", n, ii, j);

                ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0> mydata;

                if(merge == 1){
                    if (j == 0){AX_Merge_Stream_bank1 >> mydata;}
                    else if (j == 1){ AX_Merge_Stream_bank2 >> mydata;}
                    else{AX_Merge_Stream_bank3 >> mydata;}

                    for(int k = 0; k < 16; k++){
                        ap_uint<sizeof(datatype)*8> idata = mydata.data.range((k + 1)*sizeof(datatype)*8  - 1, k*sizeof(datatype)*8 );
                        tmpIn1.data[k] = *reinterpret_cast<datatype *>(&idata);
                    }
                }

                if(layerSelect >= 2) {tmpIn2 = C_stream.read();}
                else if(layerSelect < 2){
                    if(j == 0){ RM_stream >> pixevalue[j];}
                    for(int jj = 0; jj < VDATA_SIZE; jj++){
                        tmpIn2.data[jj] = pixevalue[0] * localW[0][j][0][jj];
                    }
                }

                for(int jj = 0; jj < VDATA_SIZE; jj++){
                    if(merge == 1) tmpIn3.data[jj] = tmpIn1.data[jj] + tmpIn2.data[jj];
                    else tmpIn3.data[jj] = tmpIn2.data[jj];
                }
                F_stream << tmpIn3;
            }
        }
    }
    // }
    // else{
    //     datatype pixevalue = 0;
    //     for(int n = 0; n < block_row_num; n++){
    //         for(int ii = 0; ii < VDATA_SIZE; ii++){
    //             for(int j = 0; j < BLOCK_NUM; j++){
    //                 #pragma HLS loop_flatten
    //                 #pragma HLS PIPELINE II=1 rewind
    //                 v_dt tmpIn1;
    //                 #pragma HLS aggregate variable = tmpIn1
    //                 tmpIn1 = C_stream.read();
                


    //                 F_stream << tmpIn1;
    //             }
    //         }
    //     }
    // }
}




void ApplyActivationBias(
    hls::stream<v_dt> & F_stream,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & R_stream_bank1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & R_stream_bank2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> & R_stream_bank3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp3,
    int block_row_num,
    datatype bias[BLOCK_NUM][VDATA_SIZE],
    int ReLu,
    int biasflag,
    int tomlpflag
){
    for(int n = 0; n < block_row_num; n++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int j = 0; j < BLOCK_NUM; j++){
                #pragma HLS loop_flatten
                #pragma HLS PIPELINE II=1 rewind
                v_dt tmpIn;
                // #pragma HLS aggregate variable = tmpIn
                tmpIn = F_stream.read();

                for(int jj = 0; jj < VDATA_SIZE; jj++){
                    if(biasflag == 1){
                        tmpIn.data[jj] = tmpIn.data[jj] + bias[j][jj];
                    }
                    if(ReLu == 1 && tmpIn.data[jj] <= 0){
                        tmpIn.data[jj] = 0;
                    }
                }

                ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0> mydata;
                ap_uint<sizeof(datatype) *8 * 16> datapack;
                for(int k = 0; k < 16; k++){
                    datapack.range((k + 1)*sizeof(datatype)*8  - 1, k*sizeof(datatype)*8 ) = *reinterpret_cast<ap_uint<sizeof(datatype)*8 > *>(&tmpIn.data[k]);
                }
                mydata.data = datapack;

                if(j == 0){
                    if(tomlpflag != 1) R_stream_bank1 << mydata;
                    else To_mlp1 << mydata;
                }
                else if(j == 1){
                    if(tomlpflag != 1) R_stream_bank2 << mydata;
                    else To_mlp2 << mydata;
                }
                else{
                    if(tomlpflag != 1) R_stream_bank3 << mydata;
                    else To_mlp3 << mydata;
                }
            }
        }
    }
    

}




void loadweight(
    const v_dt *W, 
    datatype localW[2][8][BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    datatype bias[BLOCK_NUM][VDATA_SIZE],
    int biasflag,
    int layerSelect
){

    // loadmyweight(W, localW);
    
    // biasinit(bias);

    // loadmybias(W, bias, biasflag);
    int i = 0;
    int ii = 0;
    int j = 0;

    for(int counter = 0; counter < BLOCK_NUM * VDATA_SIZE * BLOCK_NUM + BLOCK_NUM; counter ++){
        #pragma HLS loop_flatten
        #pragma HLS PIPELINE II=1 rewind
        if(j == BLOCK_NUM){
            ii++;
            j = 0;
        }
        if(ii == VDATA_SIZE){
            i++;
            ii = 0;
        }
        int index = counter;
        v_dt tmpIn;
        #pragma HLS aggregate variable = tmpIn
        tmpIn = W[index];

        if(counter < BLOCK_NUM * VDATA_SIZE * BLOCK_NUM){
            for (int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                localW[0][layerSelect][i][j][ii][jj] = tmpIn.data[jj];
                localW[1][layerSelect][i][j][ii][jj] = tmpIn.data[jj];
                // printf("%.0f ", localW[i][j][ii][jj]);
            } 
        }          
        else{
            for (int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                if(biasflag == 1) bias[j][jj] = tmpIn.data[jj];
                else bias[j][jj] = 0;
            }  
        }    

        j++;
    }

}


// void loadAllweight(){

// }


void startExecute(
    const v_dt * W,              // Read-Only Matrix W
    int block_row_num,           //amount of block row
    int merge,
    int ReLu,
    int biasflag,
    int tomlpflag,
    int mode,
    int layerSelect,
    datatype localW[2][8][BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE],
    datatype bias[8][BLOCK_NUM][VDATA_SIZE],
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_DDR_stream1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_DDR_stream2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_DDR_stream3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  R_stream1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  R_stream2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  R_stream3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp3
){
    #pragma HLS dataflow  
        
    static hls::stream<v_dt> AX_stream;
    #pragma HLS STREAM variable=AX_stream depth=64 type=fifo

    static hls::stream<v_dt> C_stream;
    #pragma HLS STREAM variable=C_stream depth=16 type=fifo

    static hls::stream<v_dt> F_stream;
    #pragma HLS STREAM variable=F_stream depth=16 type=fifo

    static hls::stream<datatype> RM_stream;
    #pragma HLS STREAM variable=RM_stream depth=16 type=fifo

    read_block_matrix(AX_DDR_stream1, AX_DDR_stream2, AX_DDR_stream3, AX_stream,  block_row_num, RM_stream, layerSelect);
    compute(AX_stream, C_stream, localW[0][layerSelect], block_row_num/6, layerSelect);
    MergeResult(AX_Merge_Stream_bank1, AX_Merge_Stream_bank2, AX_Merge_Stream_bank3, C_stream, F_stream, block_row_num, merge, RM_stream, layerSelect, localW[1][layerSelect]);
    ApplyActivationBias(F_stream, R_stream1, R_stream2, R_stream3, To_mlp1, To_mlp2, To_mlp3, block_row_num, bias[layerSelect], ReLu, biasflag, tomlpflag);
}



void mmult(
    const v_dt * W,              // Read-Only Matrix W
    int block_row_num,           //amount of block row
    int merge,
    int ReLu,
    int biasflag,
    int tomlpflag,
    int mode,
    int layerSelect,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_Merge_Stream_bank3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_DDR_stream1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_DDR_stream2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  AX_DDR_stream3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  R_stream1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  R_stream2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  R_stream3,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp1,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp2,
    hls::stream<ap_axiu<sizeof(datatype) *8 * 16, 0, 0, 0>> &  To_mlp3
)
{
    // define the inte
    #pragma HLS aggregate variable = W

    #pragma HLS INTERFACE mode=axis register both port=AX_Merge_Stream_bank1
    #pragma HLS INTERFACE mode=axis register both port=AX_Merge_Stream_bank2
    #pragma HLS INTERFACE mode=axis register both port=AX_Merge_Stream_bank3
    #pragma HLS INTERFACE mode=axis register both port=AX_DDR_stream1
    #pragma HLS INTERFACE mode=axis register both port=AX_DDR_stream2
    #pragma HLS INTERFACE mode=axis register both port=AX_DDR_stream3
    #pragma HLS INTERFACE mode=axis register both port=R_stream1
    #pragma HLS INTERFACE mode=axis register both port=R_stream2
    #pragma HLS INTERFACE mode=axis register both port=R_stream3
    #pragma HLS INTERFACE mode=axis register both port=To_mlp1
    #pragma HLS INTERFACE mode=axis register both port=To_mlp2
    #pragma HLS INTERFACE mode=axis register both port=To_mlp3


    #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem3

    #pragma HLS INTERFACE s_axilite port = W bundle = control
    #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control
    #pragma HLS INTERFACE s_axilite port = merge bundle = control
    #pragma HLS INTERFACE s_axilite port = ReLu bundle = control
    #pragma HLS INTERFACE s_axilite port = biasflag bundle = control
    #pragma HLS INTERFACE s_axilite port = tomlpflag bundle = control
    #pragma HLS INTERFACE s_axilite port = mode bundle = control
    #pragma HLS INTERFACE s_axilite port = layerSelect bundle = control


    #pragma HLS INTERFACE s_axilite port = return bundle = control

  
    static datatype localW[2][8][BLOCK_NUM][BLOCK_NUM][VDATA_SIZE][VDATA_SIZE];
    //#pragma HLS resource variable=localW core=RAM_2P_URAM 
    #pragma HLS bind_storage variable=localW type=RAM_T2P impl=URAM latency=1
    #pragma HLS ARRAY_PARTITION variable=localW dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=localW dim=6 complete
    
    static datatype bias[8][BLOCK_NUM][VDATA_SIZE];
    #pragma HLS bind_storage variable=bias type=RAM_T2P impl=URAM latency=1
    #pragma HLS ARRAY_PARTITION variable=bias dim=3 complete
    
    if(mode == 1){
        loadweight(W, localW, bias[layerSelect], biasflag, layerSelect);
    }
    else if(mode == 2){
        startExecute(
            W, block_row_num, merge, ReLu, biasflag, tomlpflag, mode, layerSelect,
            localW, bias, 
            AX_Merge_Stream_bank1, AX_Merge_Stream_bank2, AX_Merge_Stream_bank3,
            AX_DDR_stream1, AX_DDR_stream2, AX_DDR_stream3,
            R_stream1, R_stream2, R_stream3,
            To_mlp1, To_mlp2, To_mlp3
        );
    }
}


}
