function(find_pyamrex)
    if(QUOKKA_pyamrex_src)
        message(STATUS "Compiling local pyAMReX ...")
        message(STATUS "pyAMReX source path: ${QUOKKA_pyamrex_src}")
        if(NOT IS_DIRECTORY ${QUOKKA_pyamrex_src})
            message(FATAL_ERROR "Specified directory QUOKKA_pyamrex_src='${QUOKKA_pyamrex_src}' does not exist!")
        endif()
    elseif(QUOKKA_pyamrex_internal)
        message(STATUS "Downloading pyAMReX ...")
        message(STATUS "pyAMReX repository: ${QUOKKA_pyamrex_repo} (${QUOKKA_pyamrex_branch})")
        include(FetchContent)
    endif()

    # pyAMReX must reuse the AMReX tree already configured for Quokka.
    set(pyAMReX_amrex_internal OFF CACHE BOOL "Download & build AMReX" FORCE)
    set(pyAMReX_amrex_src ${QuokkaCode_SOURCE_DIR}/extern/amrex CACHE PATH "Local path to AMReX source directory" FORCE)
    set(pyAMReX_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(pyAMReX_INSTALL OFF CACHE BOOL "" FORCE)

    if(QUOKKA_pyamrex_internal OR QUOKKA_pyamrex_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
        if(QUOKKA_pyamrex_src)
            add_subdirectory(${QUOKKA_pyamrex_src} ${QuokkaCode_BINARY_DIR}/pyamrex)
            set(Quokka_pyAMReX_SOURCE_DIR ${QUOKKA_pyamrex_src} PARENT_SCOPE)
        else()
            FetchContent_Declare(fetchedpyamrex
                GIT_REPOSITORY ${QUOKKA_pyamrex_repo}
                GIT_TAG        ${QUOKKA_pyamrex_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedpyamrex)
            set(Quokka_pyAMReX_SOURCE_DIR ${fetchedpyamrex_SOURCE_DIR} PARENT_SCOPE)
        endif()
    else()
        find_package(pyAMReX CONFIG REQUIRED)
        message(STATUS "pyAMReX: Found version '${pyAMReX_VERSION}'")
    endif()
endfunction()

set(_default_pyamrex_src "")
if(EXISTS "${QuokkaCode_SOURCE_DIR}/extern/pyamrex/CMakeLists.txt")
    set(_default_pyamrex_src "${QuokkaCode_SOURCE_DIR}/extern/pyamrex")
endif()

set(QUOKKA_pyamrex_src "${_default_pyamrex_src}"
    CACHE PATH
    "Local path to pyAMReX source directory (preferred if set)")
option(QUOKKA_pyamrex_internal "Download & build pyAMReX" ON)
set(QUOKKA_pyamrex_repo "https://github.com/AMReX-Codes/pyamrex.git"
    CACHE STRING
    "Repository URI to pull and build pyAMReX from if(QUOKKA_pyamrex_internal)")
set(QUOKKA_pyamrex_branch "development"
    CACHE STRING
    "Repository branch for QUOKKA_pyamrex_repo if(QUOKKA_pyamrex_internal)")

find_pyamrex()
