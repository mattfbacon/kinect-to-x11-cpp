cmake_minimum_required(VERSION 3.14...3.24)

project(kinect-to-x11 VERSION 0.1 DESCRIPTION "Use a Kinect as a mouse on an X11 desktop" LANGUAGES CXX)

set(target_name kinect-to-x11)
set(source_dir src)

set(OpenGL_GL_PREFERENCE GLVND)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_package(OpenCV REQUIRED)

include(FetchContent)

FetchContent_Declare(
	freenect2
	GIT_REPOSITORY https://github.com/OpenKinect/libfreenect2.git
	GIT_TAG v0.2.1
)

FetchContent_MakeAvailable(freenect2)

add_executable(${target_name} ${source_dir}/main.cpp)
target_include_directories(${target_name} PUBLIC include)
target_compile_features(${target_name} PUBLIC cxx_std_20)
set_target_properties(${target_name} PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(${target_name} PRIVATE -Wall -Wextra -Werror -pedantic-errors -pedantic -Wconversion -Wsign-conversion)

option(BUILD_EXAMPLES OFF)
target_include_directories(freenect2 INTERFACE target/_deps/freenect2-src/include target/_deps/freenect2-build)
target_link_libraries(${target_name} PUBLIC freenect2)

target_include_directories(${target_name} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target_name} PUBLIC ${OpenCV_LIBS})
