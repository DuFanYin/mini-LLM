#include "core/stream.h"

#include "core/cuda_check.h"

namespace core {

Stream::Stream() {
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

Stream::~Stream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void Stream::synchronize() const {
    cuda_check(cudaStreamSynchronize(stream_));
}

} // namespace core
