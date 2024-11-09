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


int main(int argc, char** argv) {
    std::cout << "argc = " << argc << std::endl;
	for(int i=0; i < argc; i++){
	    std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
	}

    std::string binaryFile = "./combine_top.xclbin";
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "device name:     " << device.get_info<xrt::info::device::name>() << "\n";
    std::cout << "device bdf:      " << device.get_info<xrt::info::device::bdf>() << "\n";

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin("./combine_top.xclbin");

    xrt::kernel krnl_dl1 = xrt::kernel(device, uuid, "dataloader:{dataloader_1}", xrt::kernel::cu_access_mode::exclusive);
    xrt::kernel krnl_dl2 = xrt::kernel(device, uuid, "dataloader:{dataloader_2}", xrt::kernel::cu_access_mode::exclusive);
    xrt::kernel krnl_dl3 = xrt::kernel(device, uuid, "dataloader:{dataloader_3}", xrt::kernel::cu_access_mode::exclusive);
    xrt::kernel krnl_mlp = xrt::kernel(device, uuid, "mlp:{mlp_1}");

    std::cout << "Allocate Buffer in Global Memory hello\n";

    const int num_vertex = 256;

    xrt::bo boX_bank1 = xrt::bo(device, size_t(sizeof(v_dt) * num_vertex), krnl_dl1.group_id(0));
    xrt::bo boX_bank2 = xrt::bo(device, size_t(sizeof(v_dt) * num_vertex), krnl_dl2.group_id(0));
    xrt::bo boX_bank3 = xrt::bo(device, size_t(sizeof(v_dt) * num_vertex), krnl_dl3.group_id(0));
    xrt::bo weight_l1 = xrt::bo(device, size_t(sizeof(v_dt) * (64 * 256 * 3 + 4)), krnl_mlp.group_id(0));
    xrt::bo weight_l2 = xrt::bo(device, size_t(sizeof(v_dt) * (64 + 1)), krnl_mlp.group_id(1));
    xrt::bo result_holder = xrt::bo(device, size_t(sizeof(v_dt)), krnl_mlp.group_id(3));

    v_dt * boX_map_bank1 = boX_bank1.map<v_dt *>();
    v_dt * boX_map_bank2 = boX_bank2.map<v_dt *>();
    v_dt * boX_map_bank3 = boX_bank3.map<v_dt *>();
    v_dt * weight_l1_map = weight_l1.map<v_dt *>();
    v_dt * weight_l2_map = weight_l2.map<v_dt *>();
    v_dt * result_holder_map = result_holder.map<v_dt *>();

    std::cout<< "Reading the input.bin\n";

    float fdata;
    std::ifstream fin("../../Layer/data/intermediate/After_gconv4.bin", std::ios::binary);
    for(int i = 0; i < num_vertex; i++){
        for(int f = 0; f < 3; f++){
            for(int j = 0; j < 16; j++){
                fin.read(reinterpret_cast<char*>(&fdata), sizeof(float));
                if(f == 0) boX_map_bank1[i].data[j] = fdata;
                if(f == 1) boX_map_bank2[i].data[j] = fdata;
                if(f == 2) boX_map_bank3[i].data[j] = fdata;
            }
        }
    }

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


    boX_bank1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boX_bank2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boX_bank3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    weight_l1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    weight_l2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set the input argument for loading weight
    auto load_weight_run = xrt::run(krnl_mlp); 
    load_weight_run.set_arg(0, weight_l1);
    load_weight_run.set_arg(1, weight_l2);
    load_weight_run.set_arg(2, 1);
    load_weight_run.set_arg(3, result_holder);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[EVENT] Start execution of loading the model weights\n";

    load_weight_run.start();
    load_weight_run.wait();

    std::cout << "[EVENT] Finish execution of loading the model weights\n";
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    // set the input argument for executing
    auto read1_run = xrt::run(krnl_dl1); 
    auto read2_run = xrt::run(krnl_dl2); 
    auto read3_run = xrt::run(krnl_dl3);
    auto mlp_run = xrt::run(krnl_mlp); 

    read1_run.set_arg(0, boX_bank1);
    read2_run.set_arg(0, boX_bank2);
    read3_run.set_arg(0, boX_bank3);
    mlp_run.set_arg(0, weight_l1);
    mlp_run.set_arg(1, weight_l2);
    mlp_run.set_arg(2, 0);
    mlp_run.set_arg(3, result_holder);

    start = std::chrono::high_resolution_clock::now();
    std::cout << "[EVENT] Start the execution of MLP inference\n";

    mlp_run.start();    
    read1_run.start(); 
    read2_run.start();
    read3_run.start();

    read1_run.wait(); 
    read2_run.wait();
    read3_run.wait();


    mlp_run.wait();


    std::cout << "[EVENT] Finish the execution of MLP inference\n";
    finish = std::chrono::high_resolution_clock::now();
    std::cout << "hardware multiplication took "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
              << " microseconds\n";

    result_holder.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    std::ifstream fin_After_fc2("../../Layer/data/intermediate/After_fc2.bin", std::ios::binary);
    for(int i = 0; i < 10; i++){
        fin_After_fc2.read(reinterpret_cast<char*>(&fdata), sizeof(float));
        printf("golden %f, myresult %f\n", fdata, result_holder_map[0].data[i]);
    }
}