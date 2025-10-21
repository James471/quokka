if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_mhd_rotor test_mhd_rotor.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_mhd_rotor)
    endif()

    add_test(NAME MHDRotor COMMAND test_mhd_rotor ../inputs/mhd_rotor.in ${QuokkaTestParams} ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
