cmake_minimum_required(VERSION 3.10)
project(SilixelRefactor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# SDL3
set(SDL_SHARED ON CACHE BOOL "" FORCE)
set(SDL_STATIC OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/SDL)

# ImGui
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

add_executable(SilixelRefactor
    main.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(SilixelRefactor PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    extern/SDL/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(SilixelRefactor PRIVATE
    Vulkan::Vulkan
    SDL3::SDL3-shared
)

add_definitions(-DSRC_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\")