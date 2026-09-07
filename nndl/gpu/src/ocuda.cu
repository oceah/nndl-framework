#include "gpu/ocuda.h"

#include <string>
#include <stdexcept>
#include <iostream>

using std::cout;
using std::endl;

namespace ocean {

    static bool cuda_initialized = false;
    static cudaDeviceProp current_device_prop;

    int CUDA_Init(bool print_info) {
        if (cuda_initialized)
            return 0;
        // https://blog.csdn.net/xsc_c/article/details/24250345
        // 获取 CUDA 设备数
        int count;
        cudaGetDeviceCount(&count);// 返回具有计算能力的设备的数量
        if (count == 0) {
            if (print_info)
                fprintf(stderr, "There is no device.\n");
            return 1;
        }
        // 获取 CUDA 设备属性
        // function printDeviceProp
        auto printDeviceProp = [](const cudaDeviceProp &prop) {
            cout << "[" << prop.name << "]" << endl;
            cout << "totalGlobalMem = " << (double) prop.totalGlobalMem / 1073741824.0 << " GB" << endl;
            cout << "clockRate = " << (prop.clockRate / 1000000.0) << " GHz" << endl;
            cout << "maxThreadsPerBlock = " << prop.maxThreadsPerBlock << endl;
            cout << "maxThreadsDim = [" <<
                 prop.maxThreadsDim[0] << ' ' <<
                 prop.maxThreadsDim[1] << ' ' <<
                 prop.maxThreadsDim[2] << ']' << endl;
            cout << "maxBlocksPerMultiProcessor = " << prop.maxBlocksPerMultiProcessor << endl;
            cout << "maxThreadsPerMultiProcessor = " << prop.maxThreadsPerMultiProcessor << endl;
            cout << "maxGridSize = ["
                 << prop.maxGridSize[0] << ' '
                 << prop.maxGridSize[1] << ' '
                 << prop.maxGridSize[2] << ']' << endl;
        };
        int i;
        for (i = 0; i < count; ++i) {
            if (cudaGetDeviceProperties(&current_device_prop, i) == cudaSuccess)
                if (current_device_prop.major >= 1) {
                    if (print_info)
                        printDeviceProp(current_device_prop);
                    break;
                }
        }
        // if you can't find the device
        if (i == count) {
            if (print_info)
                fprintf(stderr, "There is no device supporting CUDA.\n");
            return 2;
        }
        // 设置 CUDA 设备
        cudaSetDevice(i);
        cuda_initialized = true;
        return 0;
    }

    bool CUDA_IsAvailable() {
        return cuda_initialized;
    }

    const cudaDeviceProp &CUDA_GetCurrentDeviceProp() {
        return current_device_prop;
    }

    void CUDNN_Check(cudnnStatus_t status) {
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error("CUDNN Error: " + std::to_string(status));
    }

} // namespace ocean
