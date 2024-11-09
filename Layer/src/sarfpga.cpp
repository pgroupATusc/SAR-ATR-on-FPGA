#include <iostream>
#include <cstring>
#include <datatype.h>
#include <accelerator.h>
#include <accload.h>
#include <utils2.h>
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

#include <pybind11/pybind11.h>

namespace py = pybind11;



PYBIND11_MODULE(sarfpga, m){
    py::class_<acceleratorArray>(m, "acceleratorArray")
        .def(py::init<std::string>())
        .def("preparation", &acceleratorArray::preparation)
        .def("loadweight", &acceleratorArray::loadweight)
        .def("inference", &acceleratorArray::inference)
        .def("loadinput", &acceleratorArray::loadinput)
        .def("checkresult", &acceleratorArray::checkresult);
}