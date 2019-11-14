cmake_minimum_required(VERSION 2.6)

project(hesaff LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(GetGitRevisionDescription)
git_describe(VERSION --tags)# --dirty=d)

string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_SHA1 "${VERSION}")
set(VERSION_SHORT "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_SHA1}")

set(CMAKE_PROJECT_VERSION_MAJOR "${VERSION_MAJOR}")
set(CMAKE_PROJECT_VERSION_MINOR "${VERSION_MINOR}")
set(CMAKE_PROJECT_VERSION_PATCH "${VERSION_PATCH}${VERSION_SHA1}")
set(CPACK_GENERATOR "TGZ")
include(CPack)


find_package(OpenCV REQUIRED)

include(GNUInstallDirs)

add_subdirectory(hesaff)
add_subdirectory(example)

