#include <iostream>
#include <cstring>
#include <hls_half.h>
#include "ap_fixed.h"

#ifndef DATADEFINE_H
#define DATADEFINE_H

#define VDATA_SIZE 16


typedef half datatype;



struct v_float {
    datatype data[VDATA_SIZE];
};

struct v_float_mlp {
    float data[VDATA_SIZE];
};

struct edge{
    int src;
    int dst;
    float value;
    int flag;
};

struct v_edge {
    edge data[4];
};

#define F_BLOCK_LEN 3 
#define NUM_FEA_BANK 4
#define NUM_VER_PER_BANK 128*32



struct v_dt {
    datatype data[VDATA_SIZE];
};

struct v_dt_mlp {
    float data[VDATA_SIZE];
};


#define BLOCK_NUM 3 // 48 * 48 = (6*8) * (6*8)
#define BUFFER_BLOCK 6

#endif