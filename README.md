# GNN Inference for SAR-ATR on FPGA

This is our demo work at FCCM 2023. The source code is based on our paper on Accelerating GNN-Based SAR Automatic Target Recognition on HBM-Enabled FPGA.

### Main Contributor
Bingyi Zhang

## Requirement 
1. Vitis Unified Software Platform (2023) - Target hardware Alveo U280
2. Python3

## Introduction
Synthetic Aperture Radar (SAR) automatic target recognition (ATR) is a key technique for remote-sensing image recognition. In real-world applications, massive SAR images are captured by airplanes or satellites, requiring high-throughput and low-latency processing. Recently, Graph Neural Networks (GNNs) have shown superior performance for SAR ATR in terms of accuracy and computational complexity. In this paper, we accelerate GNN-based SAR ATR on an FPGA. In the proposed design, we develop a customized data path and memory organization to execute various computation kernels of GNNs, including feature aggregation and feature transformation. We exploit the high bandwidth memory (HBM) of the FPGA to speed up data loading and store intermediate results. We employ the splitting kernel technique to improve the routability and frequency of the design on FPGA. We implement the proposed design using High-level Synthesis (HLS) on a state-of-the-art data-center FPGA board, the AMD Alveo U280. Compared with implementations on state-of-the-art CPUs (GPUs), our FPGA implementation achieves a 5.2x (l.57x) lower latency, a lOx (3.3x) higher throughput, and is 36.2x (7.35x) more energy efficient.

## Demo Execution in Target hardware

Demo on FPGA:  <code> python ./gui/demo-FPGA.py </code>

Demo on CPU: <code> python ./gui/demo-CPU.py </code>

(Only tested on Alveo U280 connected to a CPU using PCIe)

## Additional notes
Before execution, download large source files using links in each folder, 'link_to_download.txt'

## synthesis the featue aggregation kernel

<code> g++ -g -std=c++17 -Wall -O0 ../src/host.cpp -o ./app.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++ </code>

<code> emconfigutil --platform xilinx_u280_gen3x16_xdma_1_202211_1 </code>


## software simulation
<code> v++ -c -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k mmult -I../src ../src/mmultv2.cpp -o ./mmult.xo </code>

<code> v++ -l -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./mmult.xo -o ./mmult.xclbin </code>

<code> export XCL_EMULATION_MODE=sw_emu </code>

## hardware 
<code> v++ -c -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k mmult -I../src ../src/mmultv2.cpp -o ./mmult.xo </code>

<code> v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./mmult.xo -o ./mmult.xclbin  --kernel_frequency 200 </code>


## syntheze the feature transformation kernel

<code> g++ -g -std=c++17 -Wall -O0 ../src/host.cpp -o ./app.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++ </code>

<code> emconfigutil --platform xilinx_u280_gen3x16_xdma_1_202211_1 </code>

<code> g++ -g -std=c++17 -Wall -O0 ../src/hostv2.cpp -o ./app2.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++ </code>


<code> g++ -g -std=c++17 -Wall -O0 ../src/hostv3.cpp -o ./app3.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++ </code>

<code> g++ -g -std=c++17 -Wall -O0 ../src/hostv4.cpp -o ./app4.exe -I/opt/xilinx/xrt/include/ -I../src -L/opt/xilinx/xrt/lib -lxrt_coreutil  -lpthread -lrt -lstdc++ </code>


## software simulation
<code> v++ -c -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k feagg_top -I../src ../src/feaggv4.cpp -o ./feagg_top.xo </code>

<code> v++ -l -t sw_emu --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./feagg_top.xo -o ./feagg_top.xclbin </code>

<code> export XCL_EMULATION_MODE=sw_emu </code>

## hardware Simulation
<code> v++ -c -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg -k feagg_top -I../src ../src/feaggv4.cpp -o ./feagg_top.xo --hls.clock 500000000:feagg_top </code>

<code> v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./feagg_top.xo -o ./feagg_top.xclbin  --kernel_frequency 240 </code>

<code> v++ -l -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --config ../src/u280.cfg ./feagg_top.xo -o ./feagg_top.xclbin  --kernel_frequency 200 --clock.freqHz 200000000:feagg_top_1 --clock.defaultTolerance 0.30 </code>

## Citation
<code> @INPROCEEDINGS{10363615,
  author={Zhang, Bingyi and Kannan, Rajgopal and Prasanna, Viktor and Busart, Carl},
  booktitle={2023 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={Accelerating GNN-Based SAR Automatic Target Recognition on HBM-Enabled FPGA}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  keywords={Satellites;Target recognition;Bandwidth;Organizations;Throughput;Radar polarimetry;Kernel;Synthetic Aperture Radar;Automatic Target Recognition;Graph Neural Network;High Bandwidth Memory},
  doi={10.1109/HPEC58863.2023.10363615}}
</code>





