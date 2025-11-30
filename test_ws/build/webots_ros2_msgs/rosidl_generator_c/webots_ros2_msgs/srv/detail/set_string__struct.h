// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from webots_ros2_msgs:srv/SetString.idl
// generated code does not contain a copyright notice

#ifndef WEBOTS_ROS2_MSGS__SRV__DETAIL__SET_STRING__STRUCT_H_
#define WEBOTS_ROS2_MSGS__SRV__DETAIL__SET_STRING__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'value'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SetString in the package webots_ros2_msgs.
typedef struct webots_ros2_msgs__srv__SetString_Request
{
  rosidl_runtime_c__String value;
} webots_ros2_msgs__srv__SetString_Request;

// Struct for a sequence of webots_ros2_msgs__srv__SetString_Request.
typedef struct webots_ros2_msgs__srv__SetString_Request__Sequence
{
  webots_ros2_msgs__srv__SetString_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} webots_ros2_msgs__srv__SetString_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetString in the package webots_ros2_msgs.
typedef struct webots_ros2_msgs__srv__SetString_Response
{
  bool success;
} webots_ros2_msgs__srv__SetString_Response;

// Struct for a sequence of webots_ros2_msgs__srv__SetString_Response.
typedef struct webots_ros2_msgs__srv__SetString_Response__Sequence
{
  webots_ros2_msgs__srv__SetString_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} webots_ros2_msgs__srv__SetString_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // WEBOTS_ROS2_MSGS__SRV__DETAIL__SET_STRING__STRUCT_H_
