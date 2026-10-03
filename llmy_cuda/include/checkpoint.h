#ifndef LLMY_CHECKPOINT_H
#define LLMY_CHECKPOINT_H

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace llmy {

// Read-only mmap of a converted .llmy checkpoint file.
//
// File format (little-endian throughout):
//
//   offset  field
//   ------  -------------------------------------------------------------
//   0       magic       "LLMY" (4 bytes)
//   4       version     u32, currently 1
//   8       cfg_len     u32
//   12      config      cfg_len bytes, UTF-8 text (see Config::parse)
//   ...     n_tensors   u32
//   ...     index       n_tensors * 96-byte entries
//   4096    data        padded payload, aligned to 4096
//
// Index entry layout (96 bytes):
//   u32  name_len          (<= 63)
//   char name[64]          (NUL-padded)
//   u32  dtype             (0=f32, 1=f16, 2=bf16)
//   u32  ndim              (<= 4)
//   u32  shape[4]          (unused trailing dims are 0)
//   u64  offset            (from data_start)
//   u64  nbytes
class Checkpoint {
public:
    Checkpoint() = default;
    ~Checkpoint();

    Checkpoint(Checkpoint&& o) noexcept;
    Checkpoint& operator=(Checkpoint&& o) noexcept;
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    static Checkpoint open(const std::string& path);

    const std::string& config_text() const { return config_text_; }
    const std::unordered_map<std::string, TensorRef>& tensors() const { return tensors_; }

    bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }
    const TensorRef& get(const std::string& name) const;

    // Copy every tensor payload to device memory and repoint TensorRef::data
    // at the device allocation. The caller owns the allocations vector.
    void upload_to_device(std::vector<void*>& allocations);

private:
    void parse();

    int fd_ = -1;
    void* map_ = nullptr;
    size_t size_ = 0;
    std::string config_text_;
    std::unordered_map<std::string, TensorRef> tensors_;
};

}  // namespace llmy
#endif
