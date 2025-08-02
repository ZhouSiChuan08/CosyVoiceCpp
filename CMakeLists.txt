cmake_minimum_required(VERSION 3.12.0)
project(CosyVoiceCpp VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /W0")   # cl编译器设置utf-8编码
  add_compile_options(/Zc:__cplusplus)         # 强制 MSVC 正确报告 __cplusplus
  add_compile_definitions(_WIN32_WINNT=0x0A00) # 制定最低兼容的Windows版本 Windows 10

endif()
# 全局设置C++17标准（适用于所有目标）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 强制要求编译器支持该标准
message("Default C++ Standard: ${CMAKE_CXX_STANDARD}")
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}") # 查看当前C++编译器设置
message("CMake version: ${CMAKE_VERSION}") # 查看当前Cmake版本

# 设置可执行文件的输出目录为项目根目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# 设置库路径
set(VCPKG_X64-windows "F:/Zlab/C++/vcpkg/installed/x64-windows")
set(ThirdParty "ThirdParty")
message(STATUS "***VCPKG_X64-windows***: ${VCPKG_X64-windows}")
message(STATUS "          ***ThirdParty***  : ${ThirdParty}")

# 设置CMake查找前缀路径
set(CMAKE_PREFIX_PATH "${VCPKG_X64-windows};${ThirdParty}")
# 设置CMake查找模块路径
set(CMAKE_MODULE_PATH "${VCPKG_X64-windows}/share;${ThirdParty}/share")
# 设置CMake查找库路径
set(CMAKE_LIBRARY_PATH "${VCPKG_X64-windows}/lib;${ThirdParty}/lib")


# 准备静态库
add_library(debugTools ./src/debugTools.cpp)
add_library(tokenizer  ./src/tokenizer.cpp)
add_library(cosyVoice  ./src/cosyVoice.cpp)
add_library(toolFun    ./src/toolFun.cpp)
add_library(onnx       ./src/onnx.cpp)
add_library(tensorRT   ./src/tensorRT.cpp)
# 定义目标名称变量
set(TARGET_NAME main)
# 生成可执行文件
add_executable(
  ${TARGET_NAME} src/main.cpp
)
# 添加其他静态库目录
target_link_directories(${TARGET_NAME} PRIVATE ${VCPKG_X64-windows}/lib ${ThirdParty}/lib)  #单个执行文件链接库目录

# 设置CUDA相关路径
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe")


# 找到并链接 指定组件
find_package(Torch REQUIRED)
find_package(SndFile CONFIG REQUIRED)
message(STATUS "LibTorch status:")
message(STATUS "LibTorch    libraries: ${TORCH_LIBRARIES}")
message(STATUS " LibTorch   include  : ${TORCH_INCLUDE_DIRS}")
# 设置include目录
include_directories(
  include
  ${VCPKG_X64-windows}/include
  ${ThirdParty}/include
  ${TORCH_INCLUDE_DIRS}
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
)
get_property(include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Current include directories: ${include_dirs}")

# 链接到库
target_link_libraries(${TARGET_NAME} PRIVATE
  cosyVoice
  onnx
  tensorRT
  tokenizer
  toolFun
  debugTools
  SndFile::sndfile
  ${TORCH_LIBRARIES}
  nvinfer_10
  onnxruntime
  tokenizers_cpp
  tokenizers_c
  ws2_32
  Userenv
  NtDll
  Bcrypt
)
target_link_options(${TARGET_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
