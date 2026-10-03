cmake_minimum_required(VERSION 3.18)
project(llmy_cuda LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# GPU architecture. Default to Pascal (sm_61). Override on the command line:
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
if(MSVC)
    # CUDA 12.8 does not yet whitelist the VS 18 Insiders toolset.
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /EHsc")
    # Keep IEEE math for parity with the PyTorch reference. The recurrent
    # state and mHC/router reductions are sensitive enough that fast-math can
    # change the greedy token even when the first-token logits look close.
    set(CMAKE_CUDA_FLAGS_RELEASE "--allow-unsupported-compiler -O3 -lineinfo -Xptxas -O3 -Xcompiler /EHsc")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fno-math-errno")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo -Xptxas -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Shared library (Python loads this via ctypes)
# ---------------------------------------------------------------------------
add_library(llmy SHARED
    src/config.cpp
    src/checkpoint.cpp
    src/model.cpp
    src/llmy.cpp
    src/kernels/rmsnorm.cu
    src/kernels/silu.cu
    src/kernels/mhc.cu
    src/kernels/gdn.cu
    src/kernels/attn.cu
    src/kernels/csa.cu
    src/kernels/hca.cu
    src/kernels/router.cu
    src/kernels/moe.cu
    src/kernels/elementwise.cu
    src/kernels/lmhead.cu
    src/kernels/sample.cu
)
set_source_files_properties(src/model.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(llmy PUBLIC include)
target_link_libraries(llmy PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(llmy PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ---------------------------------------------------------------------------
# Standalone CLI (no Python needed for generation; still needs tokenizer via
# a small hard-coded vocabulary or an external JSON tokenizer file)
# ---------------------------------------------------------------------------
add_executable(llmy_cli
    src/main.cpp
)
target_include_directories(llmy_cli PRIVATE include)
target_link_libraries(llmy_cli PRIVATE llmy CUDA::cudart CUDA::cublas)
set_target_properties(llmy_cli PROPERTIES
    OUTPUT_NAME llmy
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
