project("hesaff" CXX C)
set(LIBRARY_NAME "hesaff")

set(SOURCE_FILES
    src/affine.cpp
    src/helpers.cpp
    src/hesaff.cpp
    src/pyramid.cpp
    src/siftdesc.cpp
)

add_library(${LIBRARY_NAME} ${SOURCE_FILES})

target_include_directories(${LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME} ${OpenCV_LIBRARIES})

install(
  TARGETS ${LIBRARY_NAME}
  EXPORT Find${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

