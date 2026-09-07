add_library(nndl INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(nndl INTERFACE CUDA::cublas)

target_compile_options(nndl INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)