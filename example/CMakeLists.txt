set(TEST_NAME "hesaff_test")

set(SOURCE_FILES
    hesaff.cpp
)

add_executable(${TEST_NAME} ${SOURCE_FILES})

target_link_libraries(${TEST_NAME} hesaff)

install(
   TARGETS ${TEST_NAME}
  EXPORT Find${PROJECT_NAME}
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)


