#include <iostream>
#include <cstring>

#ifndef DATADEFINE_H
#define DATADEFINE_H

#define VDATA_SIZE 16

typedef float datatype;

struct v_dt {
    datatype data[VDATA_SIZE];
};

#define BLOCK_NUM 3 // 48 * 48 = (6*8) * (6*8)
#define PRINT_FLAG 0
#define BUFFER_BLOCK 6



#endif
