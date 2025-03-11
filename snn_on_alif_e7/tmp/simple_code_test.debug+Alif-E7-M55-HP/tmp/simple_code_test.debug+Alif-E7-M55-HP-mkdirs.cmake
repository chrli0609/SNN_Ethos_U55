# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP")
  file(MAKE_DIRECTORY "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP")
endif()
file(MAKE_DIRECTORY
  "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/1"
  "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP"
  "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP/tmp"
  "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP/src/simple_code_test.debug+Alif-E7-M55-HP-stamp"
  "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP/src"
  "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP/src/simple_code_test.debug+Alif-E7-M55-HP-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP/src/simple_code_test.debug+Alif-E7-M55-HP-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/chris/skolsaker/ak5/mex/project/SNN_Ethos_U55/snn_on_alif_e7/tmp/simple_code_test.debug+Alif-E7-M55-HP/src/simple_code_test.debug+Alif-E7-M55-HP-stamp${cfgdir}") # cfgdir has leading slash
endif()
