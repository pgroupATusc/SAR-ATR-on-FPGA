#include <iostream>
#include <cstring>
#include <datatype.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>


// XRT includes
#include "xrt/xrt_bo.h"
#include <experimental/xrt_xclbin.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"


void load_input(
    v_float * input_map_bank1, 
    v_float * input_map_bank2, 
    v_float * input_map_bank3,
    int num_vertex
){
    std::cout<< "Reading the input.bin\n";

    float fdata;
    std::ifstream fin("../data/intermediate/input.bin", std::ios::binary);

    printf("initialize the data in boX_map\n");
    for(int i = 0; i < num_vertex; i++){
        for(int f = 0; f < 3; f++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(f == 0){
                    if(j == 0){
                        fin.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                        input_map_bank1[i].data[j] = fdata;
                    }
                    else{
                        input_map_bank1[i].data[j] = 0;
                    }
                }
                if(f == 1) input_map_bank2[i].data[j] = 0;
                if(f == 2) input_map_bank3[i].data[j] = 0;
            }
        }
    }
    printf("finish initializing the data in boX_map\n");
}



void load_gconv1(
    v_dt * boW_map_self, 
    v_dt * boW_neighbor_map
){
    float fdata;
    std::ifstream fin_weight("../data/weights/gconv1.lin_r.weight.bin", std::ios::binary);
    
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM * VDATA_SIZE ; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){  
            for(int j = 0; j < VDATA_SIZE; j++)
            {
                if(i == 0){
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_map_self[i*BLOCK_NUM + kk].data[j] = fdata;
                    printf("boW_map_self %f\n", fdata);
                }
                else{
                    boW_map_self[i*BLOCK_NUM + kk].data[j] = 0;
                }
            }
        }
    }


    std::ifstream fin_weight_neighbor("../data/weights/gconv1.lin_l.weight.bin", std::ios::binary);
    std::ifstream fin_bias_neighbor("../data/weights/gconv1.lin_l.bias.bin", std::ios::binary);

    printf("initailze the data in W\n");
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM * VDATA_SIZE + 1; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){  
            for(int j = 0; j < VDATA_SIZE; j++)
            {
                if(i == 0){  // load weights
                    fin_weight_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
                }
                else if(i == BLOCK_NUM * VDATA_SIZE){ // load bias
                    fin_bias_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
                }
                else{
                    boW_neighbor_map[i*BLOCK_NUM + kk].data[j] = 0;
                }
                // printf("%.0f ", boW_map[i].data[j]);
            }
        }
        // if( i % BLOCK_NUM == BLOCK_NUM - 1) printf("\n");
    }
}


void layerone_load_gconv1(
    v_dt * boW_map_self, 
    v_dt * boW_neighbor_map
){
    float fdata;
    std::ifstream fin_weight_self("../data/weights/gconv1.lin_r.weight.bin", std::ios::binary);

    for(int kk = 0; kk < BLOCK_NUM; kk++){  
        for(int j = 0; j < VDATA_SIZE; j++)
        {
            fin_weight_self.read(reinterpret_cast<char*>(&fdata), sizeof(float));
            boW_map_self[kk].data[j] = fdata;
            printf("boW_map_self[%d][%d]: %f\n", kk, j, boW_map_self[kk].data[j]);

        }
    }
    
    std::ifstream fin_weight("../data/weights/gconv1.lin_l.weight.bin", std::ios::binary);
    std::ifstream fin_bias("../data/weights/gconv1.lin_l.bias.bin", std::ios::binary);
    
    for(int i = 0; i < 1 + 1; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){  
            for(int j = 0; j < VDATA_SIZE; j++)
            {
                if(i == 0){
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
                    printf("boW_neighbor_map[%d][%d]: %.12f\n", i*BLOCK_NUM + kk, j, boW_neighbor_map[i*BLOCK_NUM + kk].data[j]);
                }
                else if(i == 1){
                    fin_bias.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
                    printf("boW_neighbor_map[%d][%d]: %.12f\n", i*BLOCK_NUM + kk, j, boW_neighbor_map[i*BLOCK_NUM + kk].data[j]);
                }
            }
        }
    }
}

void load_gconv2(
    v_dt* boW_gconv2_map_self, 
    v_dt* boW_gconv2_neighbor_map
){
    float fdata;
    std::ifstream fin_weight("../data/weights/gconv2.lin_r.weight.bin", std::ios::binary);
    
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM; i++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int kk = 0; kk < BLOCK_NUM; kk++){  
                for(int j = 0; j < VDATA_SIZE; j++)
                {
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_gconv2_map_self[(kk*VDATA_SIZE + j) * BLOCK_NUM + i].data[ii] = fdata;
                    printf("boW_gconv2_map_self %f\n", fdata);
                }
            }
        }
    }


    std::ifstream fin_weight_neighbor("../data/weights/gconv2.lin_l.weight.bin", std::ios::binary);
    std::ifstream fin_bias_neighbor("../data/weights/gconv2.lin_l.bias.bin", std::ios::binary);

    printf("initailze the data in W\n");
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM; i++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int kk = 0; kk < BLOCK_NUM; kk++){  
                for(int j = 0; j < VDATA_SIZE; j++){
                    fin_weight_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_gconv2_neighbor_map[(kk*VDATA_SIZE + j) * BLOCK_NUM + i].data[ii] = fdata;
                }
            }
        }
    }

    for(int kk = 0; kk < BLOCK_NUM; kk++){  
        for(int j = 0; j < VDATA_SIZE; j++)
        {
            int i = BLOCK_NUM * VDATA_SIZE;
            fin_bias_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
            boW_gconv2_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
        
        }
    }

}

void load_gconv3(
    v_dt* boW_gconv3_map_self, 
    v_dt* boW_gconv3_neighbor_map
){
    float fdata;
    std::ifstream fin_weight("../data/weights/gconv3.lin_r.weight.bin", std::ios::binary);
    
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM; i++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int kk = 0; kk < BLOCK_NUM; kk++){  
                for(int j = 0; j < VDATA_SIZE; j++)
                {
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_gconv3_map_self[(kk*VDATA_SIZE + j) * BLOCK_NUM + i].data[ii] = fdata;
                }
            }
        }
    }

    std::ifstream fin_weight_neighbor("../data/weights/gconv3.lin_l.weight.bin", std::ios::binary);
    std::ifstream fin_bias_neighbor("../data/weights/gconv3.lin_l.bias.bin", std::ios::binary);

    printf("initailze the data in W\n");
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM; i++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int kk = 0; kk < BLOCK_NUM; kk++){  
                for(int j = 0; j < VDATA_SIZE; j++){
                    fin_weight_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_gconv3_neighbor_map[(kk*VDATA_SIZE + j) * BLOCK_NUM + i].data[ii] = fdata;
                }
            }
        }
    }

    for(int kk = 0; kk < BLOCK_NUM; kk++){  
        for(int j = 0; j < VDATA_SIZE; j++)
        {
            int i = BLOCK_NUM * VDATA_SIZE;
            fin_bias_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
            boW_gconv3_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
        
        }
    }
}

void load_gconv4(
    v_dt* boW_gconv4_map_self, 
    v_dt* boW_gconv4_neighbor_map
){
    float fdata;
    std::ifstream fin_weight("../data/weights/gconv4.lin_r.weight.bin", std::ios::binary);
    
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM; i++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int kk = 0; kk < BLOCK_NUM; kk++){  
                for(int j = 0; j < VDATA_SIZE; j++)
                {
                    fin_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_gconv4_map_self[(kk*VDATA_SIZE + j) * BLOCK_NUM + i].data[ii] = fdata;
                }
            }
        }
    }

    std::ifstream fin_weight_neighbor("../data/weights/gconv4.lin_l.weight.bin", std::ios::binary);
    std::ifstream fin_bias_neighbor("../data/weights/gconv4.lin_l.bias.bin", std::ios::binary);

    printf("initailze the data in W\n");
    // generate random numbers for boW_map
    for(int i = 0; i < BLOCK_NUM; i++){
        for(int ii = 0; ii < VDATA_SIZE; ii++){
            for(int kk = 0; kk < BLOCK_NUM; kk++){  
                for(int j = 0; j < VDATA_SIZE; j++){
                    fin_weight_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    boW_gconv4_neighbor_map[(kk*VDATA_SIZE + j) * BLOCK_NUM + i].data[ii] = fdata;
                }
            }
        }
    }

    for(int kk = 0; kk < BLOCK_NUM; kk++){  
        for(int j = 0; j < VDATA_SIZE; j++)
        {
            int i = BLOCK_NUM * VDATA_SIZE;
            fin_bias_neighbor.read(reinterpret_cast<char*>(&fdata), sizeof(float));
            boW_gconv4_neighbor_map[i*BLOCK_NUM + kk].data[j] = fdata;
        
        }
    }
    
}

void verify_gconv1(
    v_dt* boC_map_bank1,
    v_dt* boC_map_bank2,
    v_dt* boC_map_bank3
){  
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_gconv1.bin", std::ios::binary);
    const int BM = 1026;
    for(int i = 0; i < BM * VDATA_SIZE; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                if(i < 16384){
                    fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));

                    if(kk == 0 && fdata != boC_map_bank1[i].data[j]) 
                        printf("(%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_map_bank1[i].data[j]);
                    if(kk == 1 && fdata != boC_map_bank2[i].data[j]) 
                        printf("(%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_map_bank2[i].data[j]);
                    if(kk == 2 && fdata != boC_map_bank3[i].data[j]) 
                        printf("(%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_map_bank3[i].data[j]);
                }
            }
        }
    }
}




void verify_pooling1(
    v_float* boC_pooling1_bank1_map,
    v_float* boC_pooling1_bank2_map,
    v_float* boC_pooling1_bank3_map
){  
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_pooling1.bin", std::ios::binary);

    for(int i = 0; i < 64*64; i++){
        for(int kk = 0; kk < 3; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));

                if(kk == 0 && fdata != boC_pooling1_bank1_map[i].data[j]) 
                    printf("pooling 1: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling1_bank1_map[i].data[j]);
                if(kk == 1 && fdata != boC_pooling1_bank2_map[i].data[j]) 
                    printf("pooling 1: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling1_bank2_map[i].data[j]);
                if(kk == 2 && fdata != boC_pooling1_bank3_map[i].data[j]) 
                    printf("pooling 1: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling1_bank3_map[i].data[j]);
            
            }
        }
    }
}

void verify_pooling2(
    v_float* boC_pooling2_bank1_map, 
    v_float* boC_pooling2_bank2_map, 
    v_float* boC_pooling2_bank3_map
){
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_pooling2.bin", std::ios::binary);

    for(int i = 0; i < 32*32; i++){
        for(int kk = 0; kk < 3; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));

                if(kk == 0 && fdata != boC_pooling2_bank1_map[i].data[j]) 
                    printf("pooling 2: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling2_bank1_map[i].data[j]);
                if(kk == 1 && fdata != boC_pooling2_bank2_map[i].data[j]) 
                    printf("pooling 2: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling2_bank2_map[i].data[j]);
                if(kk == 2 && fdata != boC_pooling2_bank3_map[i].data[j]) 
                    printf("pooling 2: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling2_bank3_map[i].data[j]);
            
            }
        }
    }

}

void verify_pooling3(
    v_float* boC_pooling3_bank1_map, 
    v_float* boC_pooling3_bank2_map, 
    v_float* boC_pooling3_bank3_map
){
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_pooling3.bin", std::ios::binary);

    for(int i = 0; i < 16*16; i++){
        for(int kk = 0; kk < 3; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));

                if(kk == 0 && fdata != boC_pooling3_bank1_map[i].data[j]) 
                    printf("pooling 3: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling3_bank1_map[i].data[j]);
                if(kk == 1 && fdata != boC_pooling3_bank2_map[i].data[j]) 
                    printf("pooling 3: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling3_bank2_map[i].data[j]);
                if(kk == 2 && fdata != boC_pooling3_bank3_map[i].data[j]) 
                    printf("pooling 3: (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_pooling3_bank3_map[i].data[j]);
            
            }
        }
    }



}


void verify_gconv2(
    v_dt* boC_gconv2_map_bank1, 
    v_dt* boC_gconv2_map_bank2, 
    v_dt* boC_gconv2_map_bank3
){
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_gconv2.bin", std::ios::binary);
    for(int i = 0; i < 4096; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                if(kk == 0 && fdata != boC_gconv2_map_bank1[i].data[j]) 
                    printf("After_gconv2 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv2_map_bank1[i].data[j]);
                if(kk == 1 && fdata != boC_gconv2_map_bank2[i].data[j]) 
                    printf("After_gconv2 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv2_map_bank2[i].data[j]);
                if(kk == 2 && fdata != boC_gconv2_map_bank3[i].data[j]) 
                    printf("After_gconv2 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv2_map_bank3[i].data[j]);
                
            }
        }
    }
}


void verify_gconv3(
    v_dt* boC_gconv3_map_bank1, 
    v_dt* boC_gconv3_map_bank2, 
    v_dt* boC_gconv3_map_bank3
){
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_gconv3.bin", std::ios::binary);
    for(int i = 0; i < 1024; i++){
        for(int kk = 0; kk < BLOCK_NUM; kk++){
            for(int j = 0; j < VDATA_SIZE; j++){
                fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                if(kk == 0 && fdata != boC_gconv3_map_bank1[i].data[j]) 
                    printf("After_gconv3 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv3_map_bank1[i].data[j]);
                if(kk == 1 && fdata != boC_gconv3_map_bank2[i].data[j]) 
                    printf("After_gconv3 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv3_map_bank2[i].data[j]);
                if(kk == 2 && fdata != boC_gconv3_map_bank3[i].data[j]) 
                    printf("After_gconv3 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv3_map_bank3[i].data[j]);
                
            }
        }
    }
}

void verify_gconv4(
    v_dt* boC_gconv4_map_bank1, 
    v_dt* boC_gconv4_map_bank2, 
    v_dt* boC_gconv4_map_bank3
){
    float fdata;
    std::ifstream fin_tmpT("../data/intermediate/After_gconv4.bin", std::ios::binary);
    for(int i = 0; i < 16*16; i++){
        for(int kk = 0; kk < 3; kk++){
            for(int j = 0; j < 16; j++){
                fin_tmpT.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                if(kk == 0 && fdata != boC_gconv4_map_bank1[i].data[j]) 
                    printf("After_gconv4 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv4_map_bank1[i].data[j]);
                if(kk == 1 && fdata != boC_gconv4_map_bank2[i].data[j]) 
                    printf("After_gconv4 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv4_map_bank2[i].data[j]);
                if(kk == 2 && fdata != boC_gconv4_map_bank3[i].data[j]) 
                    printf("After_gconv4 (%d, %d) golden: %f, output: %f\n", i, kk * VDATA_SIZE + j, fdata, boC_gconv4_map_bank3[i].data[j]);
                
            }
        }
    }
}


void create_edges_layer1(
    v_edge * boA_p1_map,
    v_edge * boA_p2_map
){
    for(int i = 0; i < 16; i++){  // row index
        for(int j = 0; j < 128; j++){  // column index
            for(int bank = 0; bank < 4; bank++){
                int edge_count = 0;
                for(int k = 0; k < 9; k++){ 
                    bool condition1 = (i - 1 < 0) && (bank == 0);
                    bool condition2 = 0;
                    if(k == 0){ // define the in-going edge from the up
                        if(condition1){}
                        else edge_count++;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        if(condition2){}
                        else edge_count++;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        if(j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        if(j + 1 >= 128){}
                        else edge_count++;
                    }
                    else if (k == 4){ /// self edge
                        edge_count++;
                    }
                    else if (k == 5){  // upper left neighbor
                        if(condition1 || (j - 1 < 0) ){}
                        else edge_count++;
                    }
                    else if (k == 6){ // upper right neighbor
                        if(condition1 || j + 1 >= 128){}
                        else edge_count++;
                    }
                    else if (k == 7){ // bottom left neighbor
                        if(condition2 || j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 8){
                        if(condition2 || j + 1 >= 128){}
                        else edge_count++;
                    }
                }

                for(int k = 0; k < 9; k++){  
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = (i - 1 < 0) && (bank == 0);
                    bool condition2 = 0;
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 128 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *128 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 128 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 128 + (j + 1);
                        if(j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 128 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 128 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 128 + (j + 1);
                        if(condition1 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 128 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 128 + (j + 1);
                        if(condition2 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 128 + j;
                    iedge.value = 1.0/edge_count;
                    int index = k * 16 * 128 + i * 128 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_p1_map[index].data[bank] = iedge;
                }
            }
        }
    }

    // initialze the data in boA_p2_map
    // 4 means four directions of the edges, up, down, left, right
    for(int i = 0; i < 16; i++){  // row index
        for(int j = 0; j < 128; j++){  // column index
            for(int bank = 0; bank < 4; bank++){
                int edge_count = 0;
                for(int k = 0; k < 9; k++){ 
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 16) && (bank == 3);
                    if(k == 0){ // define the in-going edge from the up
                        if(condition1){}
                        else edge_count++;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        if(condition2){}
                        else edge_count++;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        if(j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        if(j + 1 >= 128){}
                        else edge_count++;
                    }
                    else if (k == 4){ /// self edge
                        edge_count++;
                    }
                    else if (k == 5){  // upper left neighbor
                        if(condition1 || (j - 1 < 0) ){}
                        else edge_count++;
                    }
                    else if (k == 6){ // upper right neighbor
                        if(condition1 || j + 1 >= 128){}
                        else edge_count++;
                    }
                    else if (k == 7){ // bottom left neighbor
                        if(condition2 || j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 8){
                        if(condition2 || j + 1 >= 128){}
                        else edge_count++;
                    }
                }

                for(int k = 0; k < 9; k++){ 
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 16) && (bank == 3);
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 128 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *128 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 128 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 128 + (j + 1);
                        if(j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 128 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 128 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 128 + (j + 1);
                        if(condition1 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 128 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 128 + (j + 1);
                        if(condition2 || j + 1 >= 128) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 128 + j;
                    iedge.value = 1.0/edge_count;
                    int index = k * 16 * 128 + i * 128 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_p2_map[index].data[bank] = iedge;
                }
            }
        }
    }

}


void create_edges_pooling1(
    v_edge * boA_pool1_map
){
    for(int i = 0; i < 8; i++){  // row index
        for(int j = 0; j < 64; j++){  // column index
            int index = i * 64 + j;
            int origini = i * 2;
            int originj = j * 2;
            int originscale = 64 * 2;

            edge iedge1;
            iedge1.src = origini * originscale + originj;
            iedge1.dst = index;
            iedge1.value = 1.0/4.0;
            iedge1.flag = 1;

            edge iedge2;
            iedge2.src = (origini + 1) * originscale + originj;
            iedge2.dst = index;
            iedge2.value = 1.0/4.0;
            iedge2.flag = 1;

            edge iedge3;
            iedge3.src = origini * originscale + originj + 1;
            iedge3.dst = index;
            iedge3.value = 1.0/4.0;
            iedge3.flag = 1;

            edge iedge4;
            iedge4.src = (origini + 1) * originscale + originj + 1;
            iedge4.dst = index;
            iedge4.value = 1.0/4.0;
            iedge4.flag = 1;

            boA_pool1_map[index].data[0] = iedge1;
            boA_pool1_map[index].data[1] = iedge1;
            boA_pool1_map[index].data[2] = iedge1;
            boA_pool1_map[index].data[3] = iedge1;

            boA_pool1_map[64*8 + index].data[0] = iedge2;
            boA_pool1_map[64*8 + index].data[1] = iedge2;
            boA_pool1_map[64*8 + index].data[2] = iedge2;
            boA_pool1_map[64*8 + index].data[3] = iedge2;

            boA_pool1_map[64*8*2 + index].data[0] = iedge3;
            boA_pool1_map[64*8*2 + index].data[1] = iedge3;
            boA_pool1_map[64*8*2 + index].data[2] = iedge3;
            boA_pool1_map[64*8*2 + index].data[3] = iedge3;

            boA_pool1_map[64*8*3 + index].data[0] = iedge4;
            boA_pool1_map[64*8*3 + index].data[1] = iedge4;
            boA_pool1_map[64*8*3 + index].data[2] = iedge4;
            boA_pool1_map[64*8*3 + index].data[3] = iedge4;
        }
    }

}

void create_edges_pooling2(
    v_edge * boA_pool2_map
){
    for(int i = 0; i < 8; i++){  // row index
        for(int j = 0; j < 32; j++){  // column index
            int index = i * 32 + j;
            int origini = i * 2;
            int originj = j * 2;
            int originscale = 32 * 2;

            edge iedge1;
            iedge1.src = origini * originscale + originj;
            iedge1.dst = index;
            iedge1.value = 1.0/4.0;
            iedge1.flag = 1;

            edge iedge2;
            iedge2.src = (origini + 1) * originscale + originj;
            iedge2.dst = index;
            iedge2.value = 1.0/4.0;
            iedge2.flag = 1;

            edge iedge3;
            iedge3.src = origini * originscale + originj + 1;
            iedge3.dst = index;
            iedge3.value = 1.0/4.0;
            iedge3.flag = 1;

            edge iedge4;
            iedge4.src = (origini + 1) * originscale + originj + 1;
            iedge4.dst = index;
            iedge4.value = 1.0/4.0;
            iedge4.flag = 1;

            boA_pool2_map[index].data[0] = iedge1;
            boA_pool2_map[index].data[1] = iedge1;
            boA_pool2_map[index].data[2] = iedge1;
            boA_pool2_map[index].data[3] = iedge1;
            boA_pool2_map[32*8 + index].data[0] = iedge2;
            boA_pool2_map[32*8 + index].data[1] = iedge2;
            boA_pool2_map[32*8 + index].data[2] = iedge2;
            boA_pool2_map[32*8 + index].data[3] = iedge2;
            boA_pool2_map[32*8*2 + index].data[0] = iedge3;
            boA_pool2_map[32*8*2 + index].data[1] = iedge3;
            boA_pool2_map[32*8*2 + index].data[2] = iedge3;
            boA_pool2_map[32*8*2 + index].data[3] = iedge3;
            boA_pool2_map[32*8*3 + index].data[0] = iedge4;
            boA_pool2_map[32*8*3 + index].data[1] = iedge4;
            boA_pool2_map[32*8*3 + index].data[2] = iedge4;
            boA_pool2_map[32*8*3 + index].data[3] = iedge4;
        }
    }
}




void create_edges_pooling3(
    v_edge * boA_pool3_map
){
    for(int i = 0; i < 4; i++){  // row index
        for(int j = 0; j < 16; j++){  // column index
            int index = i * 16 + j;
            int origini = i * 2;
            int originj = j * 2;
            int originscale = 16 * 2;

            edge iedge1;
            iedge1.src = origini * originscale + originj;
            iedge1.dst = index;
            iedge1.value = 1.0/4.0;
            iedge1.flag = 1;

            edge iedge2;
            iedge2.src = (origini + 1) * originscale + originj;
            iedge2.dst = index;
            iedge2.value = 1.0/4.0;
            iedge2.flag = 1;

            edge iedge3;
            iedge3.src = origini * originscale + originj + 1;
            iedge3.dst = index;
            iedge3.value = 1.0/4.0;
            iedge3.flag = 1;

            edge iedge4;
            iedge4.src = (origini + 1) * originscale + originj + 1;
            iedge4.dst = index;
            iedge4.value = 1.0/4.0;
            iedge4.flag = 1;

            boA_pool3_map[index].data[0] = iedge1;
            boA_pool3_map[index].data[1] = iedge1;
            boA_pool3_map[index].data[2] = iedge1;
            boA_pool3_map[index].data[3] = iedge1;
            boA_pool3_map[16*4 + index].data[0] = iedge2;
            boA_pool3_map[16*4 + index].data[1] = iedge2;
            boA_pool3_map[16*4 + index].data[2] = iedge2;
            boA_pool3_map[16*4 + index].data[3] = iedge2;
            boA_pool3_map[16*4*2 + index].data[0] = iedge3;
            boA_pool3_map[16*4*2 + index].data[1] = iedge3;
            boA_pool3_map[16*4*2 + index].data[2] = iedge3;
            boA_pool3_map[16*4*2 + index].data[3] = iedge3;
            boA_pool3_map[16*4*3 + index].data[0] = iedge4;
            boA_pool3_map[16*4*3 + index].data[1] = iedge4;
            boA_pool3_map[16*4*3 + index].data[2] = iedge4;
            boA_pool3_map[16*4*3 + index].data[3] = iedge4;
        }
    }

}


void create_edges_gconv2(
    v_edge * boA_gconv2_map
){
    for(int i = 0; i < 16; i++){  // row index
        for(int j = 0; j < 64; j++){  // column index
            for(int bank = 0; bank < 4; bank++){
                int edge_count = 0;
                for(int k = 0; k < 9; k++){ 
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 16);
                    if(k == 0){ // define the in-going edge from the up
                        if(condition1){}
                        else edge_count++;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        if(condition2){}
                        else edge_count++;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        if(j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        if(j + 1 >= 64){}
                        else edge_count++;
                    }
                    else if (k == 4){ /// self edge
                        edge_count++;
                    }
                    else if (k == 5){  // upper left neighbor
                        if(condition1 || (j - 1 < 0) ){}
                        else edge_count++;
                    }
                    else if (k == 6){ // upper right neighbor
                        if(condition1 || j + 1 >= 64){}
                        else edge_count++;
                    }
                    else if (k == 7){ // bottom left neighbor
                        if(condition2 || j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 8){
                        if(condition2 || j + 1 >= 64){}
                        else edge_count++;
                    }
                }

                for(int k = 0; k < 9; k++){ 
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 16);
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 64 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *64 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 64 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 64 + (j + 1);
                        if(j + 1 >= 64) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 64 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 64 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 64 + (j + 1);
                        if(condition1 || j + 1 >= 64) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 64 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 64 + (j + 1);
                        if(condition2 || j + 1 >= 64) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 64 + j;
                    iedge.value = 1.0/edge_count;
                    int index = k * 16 * 64 + i * 64 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_gconv2_map[index].data[bank] = iedge;
                }
            }
        }
    }

}


void create_edges_gconv3(
    v_edge * boA_gconv3_map
){
    for(int i = 0; i < 8; i++){  // row index
        for(int j = 0; j < 32; j++){  // column index
            for(int bank = 0; bank < 4; bank++){
                int edge_count = 0;
                for(int k = 0; k < 9; k++){ 
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 8);
                    if(k == 0){ // define the in-going edge from the up
                        if(condition1){}
                        else edge_count++;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        if(condition2){}
                        else edge_count++;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        if(j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        if(j + 1 >= 32){}
                        else edge_count++;
                    }
                    else if (k == 4){ /// self edge
                        edge_count++;
                    }
                    else if (k == 5){  // upper left neighbor
                        if(condition1 || (j - 1 < 0) ){}
                        else edge_count++;
                    }
                    else if (k == 6){ // upper right neighbor
                        if(condition1 || j + 1 >= 32){}
                        else edge_count++;
                    }
                    else if (k == 7){ // bottom left neighbor
                        if(condition2 || j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 8){
                        if(condition2 || j + 1 >= 32){}
                        else edge_count++;
                    }
                }

                for(int k = 0; k < 9; k++){ 
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 8);
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 32 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *32 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 32 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 32 + (j + 1);
                        if(j + 1 >= 32) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 32 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 32 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 32 + (j + 1);
                        if(condition1 || j + 1 >= 32) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 32 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 32 + (j + 1);
                        if(condition2 || j + 1 >= 32) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 32 + j;
                    iedge.value = 1.0/edge_count;
                    int index = k * 8 * 32 + i * 32 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_gconv3_map[index].data[bank] = iedge;
                }
            }
        }
    }
}


void create_edges_gconv4(
    v_edge * boA_gconv4_map
){
    for(int i = 0; i < 4; i++){  // row index
        for(int j = 0; j < 16; j++){  // column index
            for(int bank = 0; bank < 4; bank++){
                int edge_count = 0;
                for(int k = 0; k < 9; k++){ 
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 4);
                    if(k == 0){ // define the in-going edge from the up
                        if(condition1){}
                        else edge_count++;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        if(condition2){}
                        else edge_count++;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        if(j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        if(j + 1 >= 16){}
                        else edge_count++;
                    }
                    else if (k == 4){ /// self edge
                        edge_count++;
                    }
                    else if (k == 5){  // upper left neighbor
                        if(condition1 || (j - 1 < 0) ){}
                        else edge_count++;
                    }
                    else if (k == 6){ // upper right neighbor
                        if(condition1 || j + 1 >= 16){}
                        else edge_count++;
                    }
                    else if (k == 7){ // bottom left neighbor
                        if(condition2 || j - 1 < 0){}
                        else edge_count++;
                    }
                    else if (k == 8){
                        if(condition2 || j + 1 >= 16){}
                        else edge_count++;
                    }
                }

                for(int k = 0; k < 9; k++){ 
                    edge iedge;
                    iedge.src = 0;
                    bool condition1 = 0;
                    bool condition2 = (i + 1 >= 4);
                    if(k == 0){ // define the in-going edge from the up
                        iedge.src = (i - 1) * 16 + j;
                        if(condition1) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 1){ // define the in-going edge from the bottom
                        iedge.src = (i + 1)  *16 + j;
                        if(condition2) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 2){ // define the in-going edge from the left
                        iedge.src = i * 16 + (j - 1);
                        if(j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 3){ // define the in-going edge from the right
                        iedge.src = i * 16 + (j + 1);
                        if(j + 1 >= 16) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 4){ /// self edge
                        iedge.src = i * 16 + j;
                        iedge.flag = 1; // self edge is always valid
                    }
                    else if (k == 5){  // upper left neighbor
                        iedge.src = (i - 1) * 16 + (j - 1);
                        if(condition1 || (j - 1 < 0) ) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 6){ // upper right neighbor
                        iedge.src = (i - 1) * 16 + (j + 1);
                        if(condition1 || j + 1 >= 16) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 7){ // bottom left neighbor
                        iedge.src = (i + 1) * 16 + (j - 1);
                        if(condition2 || j - 1 < 0) iedge.flag = 0;
                        else iedge.flag = 1;
                    }
                    else if (k == 8){
                        iedge.src = (i + 1) * 16 + (j + 1);
                        if(condition2 || j + 1 >= 16) iedge.flag = 0;
                        else iedge.flag = 1;
                    }

                    iedge.dst = i * 16 + j;
                    iedge.value = 1.0/edge_count;
                    int index = k * 4 * 16 + i * 16 + j;
                    // if(std::isnan(iedge.value)) printf("There exist nan in boA\n");
                    boA_gconv4_map[index].data[bank] = iedge;
                }
            }
        }
    }
}


void allocateMLP(
    v_dt* weight_l1_map, 
    v_dt* weight_l2_map
){
    float fdata;
    std::ifstream fin_l1_weight("../../Layer/data/weights/fc1.weight.bin", std::ios::binary);
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 256; j++){
            for(int k = 0; k < 3; k++){
                int index = (i * 256 + j) * 3 + k;
                for(int kk = 0; kk < 16; kk++){
                    fin_l1_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                    weight_l1_map[index].data[kk] = fdata;
                }
            }
        }
    }

    std::ifstream fin_l1_bias("../../Layer/data/weights/fc1.bias.bin", std::ios::binary);
    for(int i = 0; i < 4; i++){
        int index = 64*256*3 + i;
        for(int j = 0; j < 16; j++){
            fin_l1_bias.read(reinterpret_cast<char*>(&fdata), sizeof(float));
            weight_l1_map[index].data[j] = fdata;
            // printf("fc1.bias: %.16f\n", fdata);
        }
    }

    std::ifstream fin_l2_weight("../../Layer/data/weights/fc2.weight.bin", std::ios::binary);
    for(int i = 0 ; i < 10; i++){
        for(int j = 0; j < 64; j++){
            fin_l2_weight.read(reinterpret_cast<char*>(&fdata), sizeof(float));
            weight_l2_map[j].data[i] = fdata;
        }
    }

    std::ifstream fin_l2_bias("../../Layer/data/weights/fc2.bias.bin", std::ios::binary);
    for(int i = 0; i < 10; i++){
        fin_l2_bias.read(reinterpret_cast<char*>(&fdata), sizeof(float));
        weight_l2_map[64].data[i] = fdata;
    }

}

