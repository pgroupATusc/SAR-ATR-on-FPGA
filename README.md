# SAR-ATR-on-FPGA

# armdemo

Demo on FPGA:  python ./gui/demo-FPGA.py
Demo on CPU: python ./gui/demo-CPU.py



# synthesis the featue aggregation kernel

g++ -g -std=c++17 -Wall -O0 ../src/host.cpp -o ./app.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++
emconfigutil --platform xilinx_u280_gen3x16_xdma_1_202211_1


## software simulation
v++ -c -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k mmult -I../src ../src/mmultv2.cpp -o ./mmult.xo 
v++ -l -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./mmult.xo -o ./mmult.xclbin
export XCL_EMULATION_MODE=sw_emu

## hardware 
v++ -c -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k mmult -I../src ../src/mmultv2.cpp -o ./mmult.xo 
v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./mmult.xo -o ./mmult.xclbin  --kernel_frequency 200


## syntheze the feature transformation kernel

g++ -g -std=c++17 -Wall -O0 ../src/host.cpp -o ./app.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++
emconfigutil --platform xilinx_u280_gen3x16_xdma_1_202211_1

g++ -g -std=c++17 -Wall -O0 ../src/hostv2.cpp -o ./app2.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++


g++ -g -std=c++17 -Wall -O0 ../src/hostv3.cpp -o ./app3.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++
g++ -g -std=c++17 -Wall -O0 ../src/hostv4.cpp -o ./app4.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++


## software simulation
v++ -c -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k feagg_top -I../src ../src/feaggv4.cpp -o ./feagg_top.xo 
v++ -l -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./feagg_top.xo -o ./feagg_top.xclbin 
export XCL_EMULATION_MODE=sw_emu

## hardware 
v++ -c -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k feagg_top -I../src ../src/feaggv4.cpp -o ./feagg_top.xo --hls.clock 500000000:feagg_top
v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./feagg_top.xo -o ./feagg_top.xclbin  --kernel_frequency 240
v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./feagg_top.xo -o ./feagg_top.xclbin  --kernel_frequency 200 --clock.freqHz 200000000:feagg_top_1 --clock.defaultTolerance 0.30
# SAR-ATR-on-FPGA
