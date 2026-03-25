cmake_minimum_required(VERSION 3.10)
project(RigidTemplateBA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies.
# Only link the OpenCV modules used by the current code to avoid pulling in
# unrelated runtime dependencies such as dnn, sfm, or openvino.
find_package(OpenCV REQUIRED COMPONENTS core calib3d)
find_package(Ceres REQUIRED)

# Include directories.
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# Executable targets.
add_executable(BAmotiontrack BAmotiontrack.cpp)
add_executable(generate_rigid_template generate_rigid_template.cpp)

foreach(target BAmotiontrack generate_rigid_template)
    # Expose the source-tree root so the binaries can resolve default data
    # files even when launched from an out-of-source build directory.
    target_compile_definitions(${target} PRIVATE
        PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(${target}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
    )

    target_compile_options(${target} PRIVATE -O3 -march=native)
endforeach()
