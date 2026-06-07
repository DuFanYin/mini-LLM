# Phase 0a: Device Tensor Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce an owning device `Tensor`, a caching device allocator, and device-pointer kernel entry points — the foundation the device-resident inference path (Phase 0b) is built on.

**Architecture:** A new `core` static library (CUDA-only) provides `core::PoolAllocator` (a caching free-list over `cudaMalloc`) and `core::Tensor` (move-only fp32 device buffer with host↔device copy). The cuda kernel layer gains *device-pointer* entry points (`kernel::cuda::linear_device` / `gemm_nt_device`) that launch on already-resident data with **no copy and no sync** — the building block that removes the per-call transfer tax. Correctness is proven by gtest diffing device results against a host reference.

**Tech Stack:** C++23 (g++-15), CUDA 13 (nvcc, `CMAKE_CUDA_ARCHITECTURES=native` / sm_121), CMake, GoogleTest.

**Scope boundary:** This plan does NOT touch `executor_forward.cpp`, `KVCache`, or `MiniLlm`, and does NOT convert norm/rope to device. Those are Phase 0b. This plan only adds the foundation + proves the kernel seam. All new code is gated on `MINI_LLM_USE_CUDA`; the scalar build is unaffected.

---

## File Structure

Created:
- `src/core/cuda_check.h` — internal `CORE_CUDA_CHECK` macro (included only by `.cu`). Responsibility: uniform CUDA error checking.
- `src/core/allocator.h` / `src/core/allocator.cu` — `DeviceAllocator` interface + `PoolAllocator`. Responsibility: device memory allocation with reuse.
- `src/core/tensor.h` / `src/core/tensor.cu` — `core::Tensor`. Responsibility: owning fp32 device buffer + host↔device copy.
- `src/kernel/cuda/launch.h` — host-clean declarations of device-pointer kernel entry points. Responsibility: the "no-copy" kernel seam.
- `tests/core_test.cpp` — gtest for allocator, tensor, and the device kernel seam.

Modified:
- `src/kernel/cuda/gemm.cu` — add `linear_device` / `gemm_nt_device` (reusing the existing `nt_kernel`).
- `CMakeLists.txt` — add the `core` library (CUDA-only).
- `tests/CMakeLists.txt` — compile `core_test.cpp` into `tests_runner` and link `core`, only when `MINI_LLM_USE_CUDA`.

Header rule (important): public `core` headers (`tensor.h`, `allocator.h`) and `launch.h` MUST be host-clean — no `<cuda_runtime.h>`, no CUDA types — so plain `.cpp` (compiled by g++) can include them. CUDA types live only in `.cu` files via `cuda_check.h`.

---

## Task 1: `core` library skeleton + CUDA error macro

**Files:**
- Create: `src/core/cuda_check.h`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create the error-check header**

`src/core/cuda_check.h`:

```cpp
#pragma once

// Internal CUDA error checking for the core library. Include only from .cu
// translation units (it pulls in <cuda_runtime.h>); keep it out of public,
// host-includable headers.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace core {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        std::abort();
    }
}

} // namespace core

#define CORE_CUDA_CHECK(expr) ::core::cuda_check((expr), __FILE__, __LINE__)
```

- [ ] **Step 2: Add the `core` library to CMake**

In `CMakeLists.txt`, immediately after the `if(MINI_LLM_USE_CUDA) ... endif()` block that calls `enable_language(CUDA)` (around line 42), add:

```cmake
# Device foundation library (Phase 0). CUDA-only; the scalar build skips it.
if(MINI_LLM_USE_CUDA)
  add_library(core STATIC
    src/core/allocator.cu
    src/core/tensor.cu
  )
  target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(core PUBLIC CUDA::cudart mini_transformer_warnings)
endif()
```

(The two `.cu` files are created in Tasks 2 and 3. CMake configure will succeed only once they exist, so we do not configure until then — that is expected; this step just records the edit.)

- [ ] **Step 3: Commit**

```bash
git add src/core/cuda_check.h CMakeLists.txt
git commit -m "feat(core): add CUDA error-check header and core lib target"
```

---

## Task 2: `PoolAllocator`

**Files:**
- Create: `src/core/allocator.h`, `src/core/allocator.cu`
- Test: `tests/core_test.cpp` (created here)

- [ ] **Step 1: Write the allocator header**

`src/core/allocator.h`:

```cpp
#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace core {

// Abstract device allocator. allocate() returns device memory; deallocate()
// returns it for reuse. bytes passed to deallocate must match the allocate call.
class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr, std::size_t bytes) = 0;
};

// Caching free-list allocator over cudaMalloc. Freed blocks are kept in
// exact-size buckets and reused, so steady-state allocation makes no
// cudaMalloc/cudaFree calls. Single-threaded (one stream); not thread-safe.
class PoolAllocator final : public DeviceAllocator {
public:
    PoolAllocator() = default;
    ~PoolAllocator() override;
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr, std::size_t bytes) override;

    // Introspection for tests.
    std::size_t live_bytes() const { return live_bytes_; }         // currently handed out
    std::size_t reserved_bytes() const { return reserved_bytes_; } // total cudaMalloc'd

private:
    std::unordered_map<std::size_t, std::vector<void*>> free_;
    std::size_t live_bytes_ = 0;
    std::size_t reserved_bytes_ = 0;
};

} // namespace core
```

- [ ] **Step 2: Write the allocator implementation**

`src/core/allocator.cu`:

```cpp
#include "core/allocator.h"

#include "core/cuda_check.h"

#include <cuda_runtime.h>

namespace core {

void* PoolAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    auto it = free_.find(bytes);
    if (it != free_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        live_bytes_ += bytes;
        return ptr;
    }
    void* ptr = nullptr;
    CORE_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    reserved_bytes_ += bytes;
    live_bytes_ += bytes;
    return ptr;
}

void PoolAllocator::deallocate(void* ptr, std::size_t bytes) {
    if (ptr == nullptr) {
        return;
    }
    free_[bytes].push_back(ptr);
    live_bytes_ -= bytes;
}

PoolAllocator::~PoolAllocator() {
    // Free only the pooled (returned) blocks; best-effort, ignore teardown errors.
    for (auto& [size, ptrs] : free_) {
        for (void* ptr : ptrs) {
            cudaFree(ptr);
        }
    }
}

} // namespace core
```

- [ ] **Step 3: Write the failing test**

Create `tests/core_test.cpp`:

```cpp
#include "core/allocator.h"

#include <gtest/gtest.h>

namespace {

TEST(PoolAllocatorTest, ReusesFreedBlockWithoutNewReservation) {
    core::PoolAllocator alloc;

    void* a = alloc.allocate(256);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(alloc.reserved_bytes(), 256u);
    EXPECT_EQ(alloc.live_bytes(), 256u);

    alloc.deallocate(a, 256);
    EXPECT_EQ(alloc.live_bytes(), 0u);
    EXPECT_EQ(alloc.reserved_bytes(), 256u); // still reserved, now pooled

    void* b = alloc.allocate(256);            // should reuse the pooled block
    EXPECT_EQ(b, a);
    EXPECT_EQ(alloc.reserved_bytes(), 256u);  // no new cudaMalloc
    EXPECT_EQ(alloc.live_bytes(), 256u);
    alloc.deallocate(b, 256);
}

TEST(PoolAllocatorTest, ZeroBytesReturnsNull) {
    core::PoolAllocator alloc;
    EXPECT_EQ(alloc.allocate(0), nullptr);
    EXPECT_EQ(alloc.reserved_bytes(), 0u);
}

} // namespace
```

- [ ] **Step 4: Wire `core_test.cpp` into the build (CUDA-only)**

In `tests/CMakeLists.txt`, change the `add_executable(tests_runner ...)` list and the link libraries so the core test is included only under CUDA. Replace:

```cmake
add_executable(tests_runner
    ops_test.cpp
    io_test.cpp
    data_test.cpp
    decode_test.cpp
    loss_test.cpp
    model_test.cpp
)
target_link_libraries(tests_runner PRIVATE
    language_model
    GTest::gtest_main
    mini_transformer_warnings)
```

with:

```cmake
set(_tests_runner_srcs
    ops_test.cpp
    io_test.cpp
    data_test.cpp
    decode_test.cpp
    loss_test.cpp
    model_test.cpp
)
if(MINI_LLM_USE_CUDA)
    list(APPEND _tests_runner_srcs core_test.cpp)
endif()
add_executable(tests_runner ${_tests_runner_srcs})
target_link_libraries(tests_runner PRIVATE
    language_model
    GTest::gtest_main
    mini_transformer_warnings)
if(MINI_LLM_USE_CUDA)
    target_link_libraries(tests_runner PRIVATE core)
endif()
```

- [ ] **Step 5: Configure, build, run — verify the allocator tests pass**

```bash
rm -rf build && bash scripts/configure.sh cuda >/dev/null
cmake --build build --target tests_runner -j >/dev/null
./build/tests_runner --gtest_filter='PoolAllocatorTest.*'
```
Expected: `[  PASSED  ] 2 tests.`

- [ ] **Step 6: Commit**

```bash
git add src/core/allocator.h src/core/allocator.cu tests/core_test.cpp tests/CMakeLists.txt
git commit -m "feat(core): caching PoolAllocator over cudaMalloc with reuse"
```

---

## Task 3: `core::Tensor`

**Files:**
- Create: `src/core/tensor.h`, `src/core/tensor.cu`
- Test: `tests/core_test.cpp` (append)

- [ ] **Step 1: Write the tensor header**

`src/core/tensor.h`:

```cpp
#pragma once

#include <cstddef>
#include <vector>

namespace core {

class DeviceAllocator;

// Owning, move-only fp32 tensor in CUDA device memory. Allocates from a
// DeviceAllocator on construction and returns the block on destruction.
// numel() is the product of the shape dims (0 for an empty shape).
class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<std::size_t> shape, DeviceAllocator& alloc);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    float* data() { return data_; }
    const float* data() const { return data_; }
    std::size_t numel() const { return numel_; }
    const std::vector<std::size_t>& shape() const { return shape_; }

    void copy_from_host(const float* src); // H2D: copies numel() floats
    void copy_to_host(float* dst) const;   // D2H: copies numel() floats

private:
    void release() noexcept;

    float* data_ = nullptr;
    std::size_t numel_ = 0;
    std::vector<std::size_t> shape_;
    DeviceAllocator* alloc_ = nullptr;
};

} // namespace core
```

- [ ] **Step 2: Write the tensor implementation**

`src/core/tensor.cu`:

```cpp
#include "core/tensor.h"

#include "core/allocator.h"
#include "core/cuda_check.h"

#include <cuda_runtime.h>
#include <utility>

namespace core {

Tensor::Tensor(std::vector<std::size_t> shape, DeviceAllocator& alloc)
    : shape_(std::move(shape)), alloc_(&alloc) {
    numel_ = shape_.empty() ? 0 : 1;
    for (std::size_t d : shape_) {
        numel_ *= d;
    }
    data_ = (numel_ != 0) ? static_cast<float*>(alloc_->allocate(numel_ * sizeof(float))) : nullptr;
}

void Tensor::release() noexcept {
    if (data_ != nullptr && alloc_ != nullptr) {
        alloc_->deallocate(data_, numel_ * sizeof(float));
    }
    data_ = nullptr;
    numel_ = 0;
    shape_.clear();
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), numel_(other.numel_), shape_(std::move(other.shape_)), alloc_(other.alloc_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.alloc_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        numel_ = other.numel_;
        shape_ = std::move(other.shape_);
        alloc_ = other.alloc_;
        other.data_ = nullptr;
        other.numel_ = 0;
        other.alloc_ = nullptr;
    }
    return *this;
}

void Tensor::copy_from_host(const float* src) {
    if (numel_ != 0) {
        CORE_CUDA_CHECK(cudaMemcpy(data_, src, numel_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Tensor::copy_to_host(float* dst) const {
    if (numel_ != 0) {
        CORE_CUDA_CHECK(cudaMemcpy(dst, data_, numel_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

} // namespace core
```

- [ ] **Step 3: Write the failing test (append to `tests/core_test.cpp`)**

Add `#include "core/tensor.h"` near the top of `tests/core_test.cpp` (below the existing `#include "core/allocator.h"`), then append inside the anonymous namespace:

```cpp
TEST(TensorTest, HostRoundTripPreservesValues) {
    core::PoolAllocator alloc;
    core::Tensor t({2, 3}, alloc);
    EXPECT_EQ(t.numel(), 6u);
    ASSERT_NE(t.data(), nullptr);

    const std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t.copy_from_host(in.data());

    std::vector<float> out(6, 0.0f);
    t.copy_to_host(out.data());
    EXPECT_EQ(out, in);
}

TEST(TensorTest, MoveTransfersOwnership) {
    core::PoolAllocator alloc;
    core::Tensor a({4}, alloc);
    float* raw = a.data();

    core::Tensor b = std::move(a);
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.numel(), 0u);
}
```

Also add `#include <vector>` and `#include <utility>` at the top of `tests/core_test.cpp` if not already present.

- [ ] **Step 4: Build and run — verify tensor tests pass**

```bash
cmake --build build --target tests_runner -j >/dev/null
./build/tests_runner --gtest_filter='TensorTest.*'
```
Expected: `[  PASSED  ] 2 tests.`

- [ ] **Step 5: Commit**

```bash
git add src/core/tensor.h src/core/tensor.cu tests/core_test.cpp
git commit -m "feat(core): owning move-only device Tensor with host copy"
```

---

## Task 4: Device-pointer kernel seam (`linear_device` / `gemm_nt_device`)

**Files:**
- Create: `src/kernel/cuda/launch.h`
- Modify: `src/kernel/cuda/gemm.cu`
- Test: `tests/core_test.cpp` (append)

- [ ] **Step 1: Write the host-clean launch header**

`src/kernel/cuda/launch.h`:

```cpp
#pragma once

#include <cstddef>

// Device-pointer kernel entry points: all pointers are ALREADY in device memory.
// These launch on the default stream and do NOT copy host<->device and do NOT
// synchronize — the caller owns residency and ordering. This is the Phase 0
// building block for the device-resident inference path. Host-clean header (no
// CUDA types) so plain .cpp callers can include it.

namespace kernel::cuda {

// C[M,N] = A[M,K] * B[N,K]^T (row-major; B read transposed). Overwrites C.
void gemm_nt_device(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K);

// y[M,N] = x[M,K] * w[N,K]^T + (bias ? bias[N] : 0). Overwrites y. bias may be null.
void linear_device(const float* x, const float* w, const float* bias, float* y, std::size_t M, std::size_t N,
                   std::size_t K);

} // namespace kernel::cuda
```

- [ ] **Step 2: Implement the device entry points in `gemm.cu`**

In `src/kernel/cuda/gemm.cu`, add the include near the top (after `#include "kernel/cuda/device.cuh"`):

```cpp
#include "kernel/cuda/launch.h"
```

Then, immediately BEFORE the closing `} // namespace kernel` at the end of the file, add:

```cpp
namespace cuda {

void gemm_nt_device(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    const dim3 block(kTile, kTile);
    nt_kernel<<<grid_for(static_cast<int>(M), static_cast<int>(N)), block>>>(
        A, B, nullptr, C, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

void linear_device(const float* x, const float* w, const float* bias, float* y, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    const dim3 block(kTile, kTile);
    nt_kernel<<<grid_for(static_cast<int>(M), static_cast<int>(N)), block>>>(
        x, w, bias, y, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

} // namespace cuda
```

(`nt_kernel`, `grid_for`, and `kTile` live in `gemm.cu`'s anonymous namespace at `kernel` scope; the nested `kernel::cuda` block resolves them via enclosing-namespace lookup.)

- [ ] **Step 3: Write the failing test (append to `tests/core_test.cpp`)**

Add `#include "kernel/cuda/launch.h"` to the top of `tests/core_test.cpp`, then append inside the anonymous namespace:

```cpp
// Host reference: y[m,n] = bias[n] + sum_k x[m,k] * w[n,k].
TEST(KernelSeamTest, LinearDeviceMatchesHostReference) {
    const std::size_t M = 2, K = 2, N = 3;
    const std::vector<float> x = {1.0f, 2.0f,
                                  3.0f, 4.0f};            // [M,K]
    const std::vector<float> w = {1.0f, 0.0f,
                                  0.0f, 1.0f,
                                  1.0f, 1.0f};            // [N,K]
    const std::vector<float> bias = {0.5f, -0.5f, 1.0f};  // [N]

    std::vector<float> expected(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = bias[n];
            for (std::size_t k = 0; k < K; ++k) {
                acc += x[m * K + k] * w[n * K + k];
            }
            expected[m * N + n] = acc;
        }
    }

    core::PoolAllocator alloc;
    core::Tensor dx({M, K}, alloc);
    core::Tensor dw({N, K}, alloc);
    core::Tensor dbias({N}, alloc);
    core::Tensor dy({M, N}, alloc);
    dx.copy_from_host(x.data());
    dw.copy_from_host(w.data());
    dbias.copy_from_host(bias.data());

    kernel::cuda::linear_device(dx.data(), dw.data(), dbias.data(), dy.data(), M, N, K);

    std::vector<float> got(M * N, 0.0f);
    dy.copy_to_host(got.data()); // synchronous D2H orders after the kernel on the default stream
    for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "at index " << i;
    }
}

TEST(KernelSeamTest, GemmNtDeviceMatchesHostReference) {
    const std::size_t M = 3, K = 4, N = 2;
    std::vector<float> A(M * K), B(N * K);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    for (std::size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<float>((i % 3) + 1);
    }
    std::vector<float> expected(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[n * K + k];
            }
            expected[m * N + n] = acc;
        }
    }

    core::PoolAllocator alloc;
    core::Tensor dA({M, K}, alloc), dB({N, K}, alloc), dC({M, N}, alloc);
    dA.copy_from_host(A.data());
    dB.copy_from_host(B.data());

    kernel::cuda::gemm_nt_device(dA.data(), dB.data(), dC.data(), M, N, K);

    std::vector<float> got(M * N, 0.0f);
    dC.copy_to_host(got.data());
    for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "at index " << i;
    }
}
```

- [ ] **Step 4: Build and run — verify the seam tests pass**

```bash
cmake --build build --target tests_runner -j >/dev/null
./build/tests_runner --gtest_filter='KernelSeamTest.*'
```
Expected: `[  PASSED  ] 2 tests.`

- [ ] **Step 5: Commit**

```bash
git add src/kernel/cuda/launch.h src/kernel/cuda/gemm.cu tests/core_test.cpp
git commit -m "feat(kernel): device-pointer linear/gemm_nt entry points (no-copy seam)"
```

---

## Task 5: Full regression — both backends

**Files:** none (verification only)

- [ ] **Step 1: Full cuda build + entire suite**

```bash
rm -rf build && bash scripts/configure.sh cuda >/dev/null
cmake --build build --target tests_runner -j >/dev/null
./build/tests_runner 2>&1 | tail -3
```
Expected: all tests pass (24 prior + 6 new core/seam tests = 30), `[  PASSED  ] 30 tests.`

- [ ] **Step 2: Scalar build still green (core test compiled out)**

```bash
rm -rf build && bash scripts/configure.sh scalar >/dev/null
cmake --build build --target tests_runner -j >/dev/null
./build/tests_runner 2>&1 | tail -3
```
Expected: `[  PASSED  ] 24 tests.` (core_test.cpp excluded by the `MINI_LLM_USE_CUDA` guard).

- [ ] **Step 3: Restore cuda default**

```bash
rm -rf build && bash scripts/configure.sh cuda >/dev/null && cmake --build build --target tests_runner -j >/dev/null && echo OK
```

- [ ] **Step 4: Commit (if any build files changed) — otherwise skip**

No code changes in this task; nothing to commit unless a prior step left the tree dirty.

---

## Self-Review

**1. Spec coverage** (against roadmap §4 Phase 0 steps):
- 0.1 `core::Tensor` → Task 3. ✓
- 0.2 `DeviceAllocator` + pool → Task 2. ✓
- 0.5 device-pointer kernel entry points → Task 4 (`linear_device`/`gemm_nt_device` as the proven pattern). ✓
- Verification via cross-reference diff → Tasks 4–5. ✓
- 0.3 `CudaContext`/stream, 0.4 weights-on-device, 0.6 device KVCache, 0.7 device norm/rope, 0.8 executor rewire → **deliberately deferred to Phase 0b** (stated in Scope boundary). Not gaps; out of this plan's scope.

**2. Placeholder scan:** No TODO/TBD; every code step shows complete code; every command shows expected output. ✓

**3. Type consistency:** `core::PoolAllocator`, `core::DeviceAllocator` (`allocate`/`deallocate`), `core::Tensor` (`data`/`numel`/`shape`/`copy_from_host`/`copy_to_host`), `kernel::cuda::linear_device`/`gemm_nt_device` are used identically across header, impl, and tests. `nt_kernel`/`grid_for`/`kTile` are existing symbols in `gemm.cu`. ✓

---

## Notes for Phase 0b (next plan, not this one)
`CudaContext` (stream + allocator + cuBLAS handle), uploading `ModelWeights` to device tensors once, device `rms_norm`/`apply_rope`, device-resident `KVCache`, and rewiring `executor_forward` prefill/decode to thread `core::Tensor` through the stack with a single H2D (tokens) / D2H (logits) per step — proven by an end-to-end decode-latency drop vs `perf.md`.
