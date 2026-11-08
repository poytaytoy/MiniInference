cmake_minimum_required(VERSION 3.24)
project(InferenceEngine LANGUAGES C CXX CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build shared dependency libraries" OFF)

set(MODEL_PATH "${PROJECT_SOURCE_DIR}/assets/models/Qwen3-4B-Q4_K_M.gguf"
    CACHE FILEPATH "Default GGUF model for the application, benchmark and tests")
get_filename_component(MODEL_PATH "${MODEL_PATH}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")

option(BUILD_TESTING "builds the test cases")
option(BUILD_BENCHMARK "builds the benchmark")
option(BUILD_PYTHON "Build the Python extension" ON)
option(INFERENCE_SIMULATED "Use simulated token predictions until paged attention is implemented" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

include(CTest)
include(FetchContent)
# Use the pinned local JSON submodule; legacy minja reuses this registration.
set(JSON_BuildTests OFF CACHE BOOL "Build JSON tests" FORCE)
set(JSON_Install OFF CACHE BOOL "Install JSON headers" FORCE)
FetchContent_Declare(json SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json")
FetchContent_MakeAvailable(json)

# Only the GGUF reader and core tensor storage are needed.
# FORCE clears backend settings left over from earlier configurations.
set(GGML_CUDA OFF CACHE BOOL "Enable ggml CUDA backend" FORCE)
set(GGML_CPU ${BUILD_TESTING} CACHE BOOL "Enable ggml CPU backend for test references" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Enable ggml OpenMP support" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Build ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Build ggml examples" FORCE)

add_subdirectory(third_party/ggml)

file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/matmul/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp"
)

list(REMOVE_ITEM SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ChatHandler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Tokenizer.cpp"
)
add_library(inference_core STATIC ${SOURCES})
target_compile_definitions(inference_core PRIVATE INFERENCE_MODEL_PATH="${MODEL_PATH}")
target_compile_definitions(inference_core PUBLIC INFERENCE_SIMULATED=$<BOOL:${INFERENCE_SIMULATED}>)

target_include_directories(inference_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(inference_core PUBLIC cxx_std_20)
set_target_properties(inference_core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(inference_core PUBLIC ggml-base CUDA::cudart nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

if(BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_inference_engine bindings/python.cpp)
    target_link_libraries(_inference_engine PRIVATE inference_core)
    install(TARGETS _inference_engine LIBRARY DESTINATION inference_engine COMPONENT python)
endif()

# Keep the legacy C++ tokenizer/formatter only for existing tests and benchmarks.
if(BUILD_TESTING OR BUILD_BENCHMARK)
    set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
    set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "" FORCE)
    set(MINJA_FUZZTEST_ENABLED OFF CACHE BOOL "" FORCE)
    set(MINJA_USE_VENV OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(minja
        GIT_REPOSITORY https://github.com/google/minja.git
        GIT_TAG 021c2293c187789ef13d56c6cfd89c9b134fd80f
    )
    FetchContent_MakeAvailable(minja)

    add_subdirectory(third_party/tokenizers-cpp)
    target_sources(inference_core PRIVATE src/ChatHandler.cpp src/Tokenizer.cpp)
    target_link_libraries(inference_core PUBLIC minja tokenizers_cpp)
endif()

if(BUILD_TESTING)
    add_subdirectory(src/tests)
endif()

if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
