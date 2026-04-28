build_lib_example(
  NAME pqtdag_pqtdag_latency_cdf
  SOURCE_FILES pqtdag_latency_cdf.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libmobility}
)

build_lib_example(
  NAME pqtdag_pqtdag_throughput
  SOURCE_FILES pqtdag_throughput.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
)

build_lib_example(
  NAME pqtdag_pqtdag_erasure
  SOURCE_FILES pqtdag_erasure.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
)
