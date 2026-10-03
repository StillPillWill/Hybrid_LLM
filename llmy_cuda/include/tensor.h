#ifndef LLMY_TENSOR_H
#define LLMY_TENSOR_H

#include <cstdint>
#include <vector>
#include <string>

namespace llmy {

enum class DType : uint32_t {
    F32  = 0,
    F16  = 1,
    BF16 = 2,
};

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
    }
    return 0;
}

// A non-owning view into the mmap'd checkpoint. The data pointer is
// guaranteed aligned to at least 256 bytes (converter pads).
struct TensorRef {
    std::string name;
    DType dtype = DType::F32;
    int ndim = 0;
    std::vector<int64_t> shape;
    const void* data = nullptr;
    uint64_t nbytes = 0;

    // Convenience: treat as FP32 regardless of storage dtype. The current
    // converter always emits FP32, so no conversion is needed at load time.
    const float* as_f32() const {
        return reinterpret_cast<const float*>(data);
    }

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

}  // namespace llmy
#endif
