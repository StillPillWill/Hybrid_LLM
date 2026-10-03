#include "checkpoint.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace llmy {

#ifdef _WIN32
// Windows mmap emulation via CreateFileMapping. Simplest correct version.
Checkpoint::~Checkpoint() {
    if (map_) UnmapViewOfFile(map_);
    if (fd_ >= 0) CloseHandle((HANDLE)(intptr_t)fd_);
}
#else
Checkpoint::~Checkpoint() {
    if (map_) munmap(map_, size_);
    if (fd_ >= 0) close(fd_);
}
#endif

Checkpoint::Checkpoint(Checkpoint&& o) noexcept
    : fd_(o.fd_), map_(o.map_), size_(o.size_),
      config_text_(std::move(o.config_text_)),
      tensors_(std::move(o.tensors_)) {
    o.fd_ = -1;
    o.map_ = nullptr;
    o.size_ = 0;
}

Checkpoint& Checkpoint::operator=(Checkpoint&& o) noexcept {
    if (this != &o) {
        if (map_) {
#ifdef _WIN32
            UnmapViewOfFile(map_);
#else
            munmap(map_, size_);
#endif
        }
        if (fd_ >= 0) {
#ifdef _WIN32
            CloseHandle((HANDLE)(intptr_t)fd_);
#else
            close(fd_);
#endif
        }
        fd_ = o.fd_;
        map_ = o.map_;
        size_ = o.size_;
        config_text_ = std::move(o.config_text_);
        tensors_ = std::move(o.tensors_);
        o.fd_ = -1;
        o.map_ = nullptr;
        o.size_ = 0;
    }
    return *this;
}

Checkpoint Checkpoint::open(const std::string& path) {
    Checkpoint ck;

#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(h, &sz);
    HANDLE maph = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!maph) { CloseHandle(h); throw std::runtime_error("CreateFileMapping failed"); }
    void* view = MapViewOfFile(maph, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(maph);
    if (!view) { CloseHandle(h); throw std::runtime_error("MapViewOfFile failed"); }
    ck.fd_ = (int)(intptr_t)h;
    ck.map_ = view;
    ck.size_ = (size_t)sz.QuadPart;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("fstat failed"); }
    void* view = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
    ck.fd_ = fd;
    ck.map_ = view;
    ck.size_ = (size_t)st.st_size;
#endif

    ck.parse();
    return ck;
}

void Checkpoint::parse() {
    const uint8_t* base = (const uint8_t*)map_;
    const uint8_t* p = base;
    const uint8_t* end = base + size_;

    auto need = [&](size_t n) {
        if (p + n > end) throw std::runtime_error("checkpoint truncated");
    };

    need(12);
    if (std::memcmp(p, "LLMY", 4) != 0) throw std::runtime_error("bad magic");
    p += 4;
    uint32_t version;
    std::memcpy(&version, p, 4); p += 4;
    if (version != 1) throw std::runtime_error("unsupported checkpoint version");

    uint32_t cfg_len;
    std::memcpy(&cfg_len, p, 4); p += 4;
    need(cfg_len);
    config_text_.assign((const char*)p, cfg_len);
    p += cfg_len;

    need(4);
    uint32_t n;
    std::memcpy(&n, p, 4); p += 4;

    // The index can exceed one 4 KiB page for large checkpoints. The
    // converter aligns the payload after the complete variable-size index.
    // The converter stores the complete fixed-size tensor index immediately
    // after the count, then aligns the payload.  Advance past all index
    // entries before calculating the aligned data start.
    const size_t index_end = (size_t)(p - base) + (size_t)n * 108u;
    const size_t data_offset =
        (index_end + 4095u) & ~((size_t)4095u);
    if (data_offset > size_) throw std::runtime_error("invalid data offset");
    const uint8_t* data_start = base + data_offset;

    for (uint32_t i = 0; i < n; ++i) {
        need(108);
        uint32_t name_len;
        std::memcpy(&name_len, p, 4); p += 4;
        if (name_len > 63) throw std::runtime_error("tensor name too long");
        char name[64];
        std::memcpy(name, p, 64); p += 64;
        uint32_t dtype, ndim;
        std::memcpy(&dtype, p, 4); p += 4;
        std::memcpy(&ndim, p, 4); p += 4;
        if (ndim > 4) throw std::runtime_error("tensor has too many dims");
        uint32_t shape[4];
        std::memcpy(shape, p, 16); p += 16;
        uint64_t off, nbytes;
        std::memcpy(&off, p, 8); p += 8;
        std::memcpy(&nbytes, p, 8); p += 8;

        TensorRef t;
        t.name.assign(name, name_len);
        t.dtype = (DType)dtype;
        t.ndim = (int)ndim;
        for (int k = 0; k < (int)ndim; ++k) t.shape.push_back((int64_t)shape[k]);
        t.data = data_start + off;
        t.nbytes = nbytes;

        if (data_start + off + nbytes > end)
            throw std::runtime_error("tensor payload out of bounds: " + t.name);

        tensors_.emplace(t.name, std::move(t));
    }
}

const TensorRef& Checkpoint::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("missing tensor: " + name);
    return it->second;
}

void Checkpoint::upload_to_device(std::vector<void*>& allocations) {
    try {
        for (auto& item : tensors_) {
            TensorRef& tensor = item.second;
            void* device = nullptr;
            cudaError_t err = cudaMalloc(&device, (size_t)tensor.nbytes);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed for tensor: " + tensor.name);
            }
            err = cudaMemcpy(device, tensor.data, (size_t)tensor.nbytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(device);
                throw std::runtime_error("cudaMemcpy failed for tensor: " + tensor.name);
            }
            allocations.push_back(device);
            tensor.data = device;
        }
    } catch (...) {
        for (void* p : allocations) cudaFree(p);
        allocations.clear();
        throw;
    }
}

}  // namespace llmy
