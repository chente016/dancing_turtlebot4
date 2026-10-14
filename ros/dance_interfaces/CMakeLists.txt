cmake_minimum_required(VERSION 3.8)
project(dance_interfaces)

# Tools that turn the .srv file into Python / C++ code.
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)       # StartDance.srv uses builtin_interfaces/Time
find_package(rosidl_default_generators REQUIRED)

# The first argument must start with the package name. Add new .srv files here.
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StartDance.srv"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()