project (CMT)
#cmake_minimum_required (VERSION 2.6)
cmake_minimum_required(VERSION 2.8.12)
find_package(OpenCV REQUIRED)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package( Boost 1.5.9 REQUIRED COMPONENTS filesystem system program_options)
if(${Boost_FOUND})
	MESSAGE("Boost information:") 
	MESSAGE("  Boost_VERSION: ${Boost_VERSION}")
	MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
	MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
	MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 
else()
    MESSAGE(FATAL_ERROR "Boost not found in the system.")
endif()

add_definitions ("-Wall -std=c++11")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable (cmt common.cpp gui.cpp main.cpp
    CMT.cpp Consensus.cpp Fusion.cpp Matcher.cpp Tracker.cpp
    fastcluster/fastcluster.cpp)
add_executable (cmt_track common.cpp gui.cpp CppMT_tracker.cpp
    CMT.cpp Consensus.cpp Fusion.cpp Matcher.cpp Tracker.cpp
    fastcluster/fastcluster.cpp)
target_link_libraries(cmt ${OpenCV_LIBS})
target_link_libraries(cmt_track ${OpenCV_LIBS} ${Boost_LIBRARIES})


