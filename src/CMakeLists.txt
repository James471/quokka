if(QUOKKA_PYTHON) 
  message(STATUS "Building Quokka with Python support (to disable, add '-DQUOKKA_PYTHON=OFF' to the CMake command-line options)")
  find_package(Python COMPONENTS Interpreter Development NumPy)
else()
  message(STATUS "Building Quokka *without* Python support")
endif()

if(Python_FOUND)
    add_compile_definitions(HAVE_PYTHON)
    include_directories(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
    link_libraries(${Python_LIBRARIES})
endif()

if(QUOKKA_OPENPMD)
  message(STATUS "Building Quokka with OpenPMD support")
  add_compile_definitions(QUOKKA_USE_OPENPMD)
  set(openPMD_USE_ADIOS2 ON CACHE BOOL "") # ADIOS2 is required
  set(openPMD_USE_HDF5 OFF CACHE BOOL "")
  set(openPMD_USE_PYTHON OFF CACHE BOOL "")
  set(openPMD_BUILD_TESTING OFF CACHE BOOL "")
  set(openPMD_BUILD_EXAMPLES OFF CACHE BOOL "")
  set(openPMD_BUILD_CLI_TOOLS OFF CACHE BOOL "")
  set(openPMD_INSTALL OFF CACHE BOOL "")
  add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/openPMD-api ${QuokkaCode_BINARY_DIR}/openPMD-api)
  include_directories(${OpenPMD_INCLUDE_DIRS_RET})
  link_libraries(openPMD::openPMD)
  set(openPMDSources "${CMAKE_CURRENT_SOURCE_DIR}/io/openPMD.cpp")
  message(STATUS "WARNING: OpenPMD plotfiles are ENABLED. Face-centered variables will only be available as cell-centered averages in plotfiles!")
else()
  set(openPMDSources "")
endif()

# HDF5
find_package(HDF5 REQUIRED)

# std::filesystem (required for GCC < 9)
if (NOT ((CMAKE_SYSTEM_NAME MATCHES "Darwin") AND ((CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))))
    link_libraries(stdc++fs)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    # only works with gcc!
    include(CodeCoverage)
    # `pip3 install fastcov --user`
    setup_target_for_coverage_fastcov(
        NAME coverage
        EXECUTABLE ${CMAKE_CTEST_COMMAND} -j2
        BASE_DIRECTORY "../"
        EXCLUDE "${PROJECT_SOURCE_DIR}/extern" "${PROJECT_SOURCE_DIR}/src/test_" "include/openmpi/ompi/mpi/cxx/" "/usr/"
        POST_CMD perl -i -pe s!${PROJECT_SOURCE_DIR}/!!g coverage.json && fastcov_to_sonarqube coverage.json
    )
    add_compile_options("-O1")
    append_coverage_compiler_flags()
endif()

if(ENABLE_ASAN)
  # enable AddressSanitizer for debugging
  message(STATUS "Compiling with AddressSanitizer and UndefinedBehaviorSanitizer *enabled*")
  add_compile_options(-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment)
  add_link_options(-fsanitize=address -fsanitize=undefined)
endif(ENABLE_ASAN)

if(ENABLE_HWASAN)
  # enable HWAddressSanitizer for debugging
  message(STATUS "Compiling with HWAddressSanitizer *enabled*")
  add_compile_options(-fsanitize=hwaddress)
  add_link_options(-fsanitize=hwaddress)
endif(ENABLE_HWASAN)

if(DISABLE_FMAD)
  message(STATUS "Fused multiply-add (FMAD) is disabled for device code.")
  set(AMReX_CUDA_FASTMATH OFF CACHE BOOL "" FORCE)

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-ffp-contract=off>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffp-contract=off>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVHPC")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Mnofma>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Mnofma>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gpu=nofma>)
  endif()
else()
  # fused MAD causes directional asymmetry -- enabled by default by nvcc and clang
  message(STATUS "Fused multiply-add (FMAD) is *enabled* for device code. Exact direction symmetry will NOT be preserved.")
endif(DISABLE_FMAD)

if(ENABLE_TESTS_FPE)
  set(QuokkaTestParams "amrex.fpe_trap_invalid=1 amrex.fpe_trap_overflow=1 amrex.fpe_trap_zero=1")
else()
  set(QuokkaTestParams "")
  message(STATUS "WARNING: Floating-point exceptions are *DISABLED* when running the test suite.")
endif(ENABLE_TESTS_FPE)

# emit error if warnings are produced
if(WARNINGS_AS_ERRORS)
  add_compile_options(-Werror -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas -Wno-strict-aliasing)
endif(WARNINGS_AS_ERRORS)

# emit register usage per thread from CUDA assembler
# if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
# endif()
# if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcuda-ptxas>)
#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-v>)
#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-command-line-argument>)
# endif()

include_directories(${amrex_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})
include_directories(${HDF5_INCLUDE_DIRS_RET})
include_directories(${Boost_INCLUDE_DIRS_RET})
include_directories(${CMAKE_SOURCE_DIR}/src)

# default network_name (for Microphysics EOS) for all tests:
set(microphysics_network_name "gamma_law")
# default output_dir (for Microphysics EOS) for all tests:
set(microphysics_output_dir "${CMAKE_CURRENT_BINARY_DIR}/")
setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")
include_directories(${gamma_law_dirs} ${CMAKE_CURRENT_BINARY_DIR} "includes/extern_parameters.H" "includes/network_properties.H")

link_libraries(AMReX::amrex)
link_libraries(fmt::fmt)
link_libraries(hdf5::hdf5)
link_libraries(yaml-cpp::yaml-cpp)

include(CTest)

set (QuokkaSourcesNoEOS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/DiagBase.cpp" 
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/DiagFilter.cpp" 
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/DiagFramePlane.cpp" 
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/DiagPDF.cpp" 
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/DiagPlotfile.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/DiagProjectionPlot.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/io/projection.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/experimental/DescriptorPrototype.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/cooling/ResampledCooling.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/util/fextract.cpp"
                        "${openPMDSources}")

set (QuokkaObjSources "${QuokkaSourcesNoEOS}" "${gamma_law_sources}")

add_subdirectory(problems)

add_executable(experimental_advection
  experimental/testAdvectionPrototype.cpp
  ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  setup_target_for_cuda_compilation(experimental_advection)
endif()

add_test(NAME experimental_advection COMMAND experimental_advection)

if(QUOKKA_BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Install pybind11 with `pip install pybind11` or provide pybind11_DIR.")
  endif()
  pybind11_add_module(pyquokka MODULE
    bindings/pyquokka.cpp
    experimental/DescriptorPrototype.cpp)
  set(_pyamrex_include_dir "${Quokka_pyAMReX_SOURCE_DIR}")
  if(NOT _pyamrex_include_dir)
    set(_pyamrex_include_dir "${CMAKE_SOURCE_DIR}/extern/pyamrex")
  endif()
  target_include_directories(pyquokka PRIVATE ${CMAKE_SOURCE_DIR}/src ${_pyamrex_include_dir}/src)
  target_link_libraries(pyquokka PRIVATE AMReX::amrex fmt::fmt hdf5::hdf5 yaml-cpp::yaml-cpp)
  set(pyAMReX_dim_target pyAMReX_${AMReX_SPACEDIM}d)
  if(TARGET ${pyAMReX_dim_target})
    add_dependencies(pyquokka ${pyAMReX_dim_target})
  endif()
  # scikit-build only copies targets that participate in the "install" step.
  install(TARGETS pyquokka DESTINATION .)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/python/ DESTINATION .)
  if(CMAKE_PYTHON_OUTPUT_DIRECTORY)
    install(DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/amrex DESTINATION .)
  endif()
endif()
