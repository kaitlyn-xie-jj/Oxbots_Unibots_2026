// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from webots_ros2_msgs:srv/EmitterSendString.idl
// generated code does not contain a copyright notice

#ifndef WEBOTS_ROS2_MSGS__SRV__DETAIL__EMITTER_SEND_STRING__STRUCT_H_
#define WEBOTS_ROS2_MSGS__SRV__DETAIL__EMITTER_SEND_STRING__STRUCT_H_

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

/// Struct defined in srv/EmitterSendString in the package webots_ros2_msgs.
typedef struct webots_ros2_msgs__srv__EmitterSendString_Request
{
  rosidl_runtime_c__String value;
} webots_ros2_msgs__srv__EmitterSendString_Request;

// Struct for a sequence of webots_ros2_msgs__srv__EmitterSendString_Request.
typedef struct webots_ros2_msgs__srv__EmitterSendString_Request__Sequence
{
  webots_ros2_msgs__srv__EmitterSendString_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} webots_ros2_msgs__srv__EmitterSendString_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/EmitterSendString in the package webots_ros2_msgs.
typedef struct webots_ros2_msgs__srv__EmitterSendString_Response
{
  int64_t result;
} webots_ros2_msgs__srv__EmitterSendString_Response;

// Struct for a sequence of webots_ros2_msgs__srv__EmitterSendString_Response.
typedef struct webots_ros2_msgs__srv__EmitterSendString_Response__Sequence
{
  webots_ros2_msgs__srv__EmitterSendString_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} webots_ros2_msgs__srv__EmitterSendString_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // WEBOTS_ROS2_MSGS__SRV__DETAIL__EMITTER_SEND_STRING__STRUCT_H_
