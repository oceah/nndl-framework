/**
******************************************************************************
* @file             utility.h
* @brief            neural network & deep learning utility
* @author           OceanH
* @create           2025-01-13
* @latestupdate     2025-02-19
******************************************************************************
*/

#ifndef NNDL_UTILITY_H
#define NNDL_UTILITY_H

/// @brief use cuda to accelerate
#define NNDL_USE_CUDA 1

#include <iostream>

#if NNDL_USE_CUDA
#define __ocean__host__ __host__
#define __ocean__device__ __device__
#define __ocean__host__device__ __host__ __device__

#include <cublas_v2.h>

#else
#define __ocean__host__
#define __ocean__device__
#define __ocean__host__device__
#endif

namespace ocean::nndl {
    /// @brief device type
    enum class EDevice : int8_t {
        ANY = 0, // ANY DEVICE
        CPU = 1, // CPU
#if NNDL_USE_CUDA
        CUDA = 2, // NVIDIA GPU
#endif
    };

    /// @brief definition of a specific device
    struct DeviceTypeDef {
        using size_type = size_t;

        EDevice type; // device type
        size_type id; // device id

        bool operator==(const DeviceTypeDef &d) const {
            return type == d.type && id == d.id;
        }

        friend std::ostream &operator<<(std::ostream &os, const DeviceTypeDef &d);
    };

    /// @design only host manage memory
    template<class T>
    struct tensor_kernel {
        using size_type = size_t;
        using shape_type = size_type *;

        size_type *pdim; // dimension
        size_type *pnumel; // number of elements
        size_type *pshape; // shape
        T *pdata; // data[row-major]

        /* +-+-+-+-+-+-+-+-+-+- constructors -+-+-+-+-+-+-+-+-+-+ */

        __ocean__host__device__ tensor_kernel();

        __ocean__device__ tensor_kernel(shape_type _pshape, T *_pdata);

        /* +-+-+-+-+-+-+-+-+-+- getters -+-+-+-+-+-+-+-+-+-+ */

        __ocean__host__ size_type dim(const DeviceTypeDef &device) const;

        __ocean__host__ size_type numel(const DeviceTypeDef &device) const;

        __ocean__host__ size_type size(const DeviceTypeDef &device, size_type i) const;

        /* +-+-+-+-+-+-+-+-+-+- interfaces -+-+-+-+-+-+-+-+-+-+ */

        __ocean__host__ void clear(const DeviceTypeDef &device);

        __ocean__host__ void move(const DeviceTypeDef &src, const DeviceTypeDef &dst);

        __ocean__host__ tensor_kernel copy(const DeviceTypeDef &src, const DeviceTypeDef &dst) const;

        /* +-+-+-+-+-+-+-+-+-+- copy ctrls -+-+-+-+-+-+-+-+-+-+ */

        /// @note use 'copy' to copy tensor_kernel instead of copy constructor
        __ocean__host__device__ tensor_kernel(const tensor_kernel &other) = delete;

        __ocean__host__device__ tensor_kernel &operator=(const tensor_kernel &other) = delete;

        __ocean__host__device__ tensor_kernel(tensor_kernel &&other) noexcept;

        __ocean__host__device__ tensor_kernel &operator=(tensor_kernel &&other) noexcept;
    };
} // namespace ocean::nndl

#include "inls/utility.inl"

#endif // NNDL_UTILITY_H
