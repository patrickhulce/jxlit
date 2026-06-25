cmake_minimum_required(VERSION 3.16)
project(bench_libjxl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Primary path: system libjxl via pkg-config (e.g. `brew install jpeg-xl` or
# `apt install libjxl-dev`). Provides the C API plus the resizable parallel
# runner used to control the thread count.
find_package(PkgConfig REQUIRED)
pkg_check_modules(JXL libjxl)
pkg_check_modules(JXL_THREADS libjxl_threads)

if(NOT JXL_FOUND OR NOT JXL_THREADS_FOUND)
  message(FATAL_ERROR
    "libjxl / libjxl_threads not found via pkg-config.\n"
    "Install the JPEG XL reference library, e.g.:\n"
    "  macOS:  brew install jpeg-xl\n"
    "  Debian: apt install libjxl-dev")
endif()

add_executable(bench-libjxl bench.cpp)
target_include_directories(bench-libjxl PRIVATE
  ${JXL_INCLUDE_DIRS} ${JXL_THREADS_INCLUDE_DIRS})
target_link_directories(bench-libjxl PRIVATE
  ${JXL_LIBRARY_DIRS} ${JXL_THREADS_LIBRARY_DIRS})
target_link_libraries(bench-libjxl PRIVATE
  ${JXL_LIBRARIES} ${JXL_THREADS_LIBRARIES})
