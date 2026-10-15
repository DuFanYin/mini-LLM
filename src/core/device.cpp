#include "core/device.h"

#include "core/cuda_check.h"

#include <cuda_runtime_api.h>

namespace core {

DeviceInfo current_device() {
    DeviceInfo info;
    cuda_check(cudaGetDevice(&info.ordinal));
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, info.ordinal));
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    cuda_check(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
    return info;
}

} // namespace core
