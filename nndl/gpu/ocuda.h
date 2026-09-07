/**
******************************************************************************
* @file         ocuda.h
* @brief        CUDA header file.
* @author       透小犹
* @create       2024-10-08
* @lastupdate   2024-10-26
******************************************************************************
*/

#ifndef OCUDA_H
#define OCUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

namespace ocean {

/// @brief CUDA 初始化
/// @param print_info 是否打印设备信息
/// @retval 0 成功
/// @retval 1 未找到具有计算能力的设备
/// @retval 2 未找到支持 CUDA 的设备
    int CUDA_Init(bool print_info = false);

    bool CUDA_IsAvailable();

    const cudaDeviceProp &CUDA_GetCurrentDeviceProp();

    /// @brief 检查 CUDNN 错误
    /// @note 如果 status 不是 CUDNN_STATUS_SUCCESS => 抛出异常
    void CUDNN_Check(cudnnStatus_t status);

} // namespace ocean

#endif // OCUDA_H
