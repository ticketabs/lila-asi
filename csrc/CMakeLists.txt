cmake_minimum_required(VERSION 3.18)
project(lila_asi VERSION 2.6.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LILA_ASI_SOURCES
    asi_ed25519.c
    asi_v2.c
    asi_arch.c
    asi_dense.c
    lila_lm_head_candidate.c
    lila_asi.c
    asi_grammar_sparse.c
    e05_bridge.c
    asi_state_centric.c
    asi_kernels.c
    asi_graph.c
    asi_pairing.c
    asi_vision.c
    lila_asi_mobile_jni.c
)

add_library(lila_asi SHARED ${LILA_ASI_SOURCES})
add_library(lila_asi_static STATIC ${LILA_ASI_SOURCES})
set_target_properties(lila_asi_static PROPERTIES OUTPUT_NAME lila_asi)

target_include_directories(lila_asi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lila_asi_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(lila_asi PRIVATE _GNU_SOURCE)
target_compile_definitions(lila_asi_static PRIVATE _GNU_SOURCE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(NOT MSVC)
        target_compile_options(lila_asi PRIVATE -O3 -mavx2 -mfma -mf16c)
        target_compile_options(lila_asi_static PRIVATE -O3 -mavx2 -mfma -mf16c)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    if(NOT MSVC)
        target_compile_options(lila_asi PRIVATE -O3)
        target_compile_options(lila_asi_static PRIVATE -O3)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(lila_asi PRIVATE Threads::Threads m)
target_link_libraries(lila_asi_static PRIVATE Threads::Threads m)

find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    target_link_libraries(lila_asi PRIVATE OpenMP::OpenMP_C)
    target_link_libraries(lila_asi_static PRIVATE OpenMP::OpenMP_C)
endif()

if(ANDROID)
    find_library(LOG_LIB log)
    if(LOG_LIB)
        target_link_libraries(lila_asi PRIVATE ${LOG_LIB})
    endif()
endif()

if(NOT ANDROID AND NOT IOS)
    add_executable(lila-asi-cli asi_v2_cli.c)
    target_link_libraries(lila-asi-cli PRIVATE lila_asi_static Threads::Threads m)
    if(OpenMP_C_FOUND)
        target_link_libraries(lila-asi-cli PRIVATE OpenMP::OpenMP_C)
    endif()
endif()
