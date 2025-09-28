add_executable(test_hydro_shear_shocktube test_hydro_shear_shocktube.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_hydro_shear_shocktube)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME HydroShearShocktube COMMAND test_hydro_shear_shocktube ../inputs/shocktube_shear.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
