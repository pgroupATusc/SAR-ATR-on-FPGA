#include <iostream>
#include <cstring>
#include <hls_half.h>

#ifndef DATADEFINE_H
#define DATADEFINE_H

#define VDATA_SIZE 16

typedef half datatype;

struct v_dt {
    datatype data[VDATA_SIZE];
};


struct v_dt_host {
    float data[VDATA_SIZE];
};

#define BLOCK_NUM 3 // 48 * 48 = (6*8) * (6*8)
#define PRINT_FLAG 0
#define BUFFER_BLOCK 6

#endif
