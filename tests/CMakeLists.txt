project(tests)

add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../include)

add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} gtest gtest_main)
