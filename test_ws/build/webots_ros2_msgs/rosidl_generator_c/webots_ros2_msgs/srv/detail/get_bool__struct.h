// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from webots_ros2_msgs:srv/GetBool.idl
// generated code does not contain a copyright notice

#ifndef WEBOTS_ROS2_MSGS__SRV__DETAIL__GET_BOOL__STRUCT_H_
#define WEBOTS_ROS2_MSGS__SRV__DETAIL__GET_BOOL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetBool in the package webots_ros2_msgs.
typedef struct webots_ros2_msgs__srv__GetBool_Request
{
  bool ask;
} webots_ros2_msgs__srv__GetBool_Request;

// Struct for a sequence of webots_ros2_msgs__srv__GetBool_Request.
typedef struct webots_ros2_msgs__srv__GetBool_Request__Sequence
{
  webots_ros2_msgs__srv__GetBool_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} webots_ros2_msgs__srv__GetBool_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/GetBool in the package webots_ros2_msgs.
typedef struct webots_ros2_msgs__srv__GetBool_Response
{
  bool value;
} webots_ros2_msgs__srv__GetBool_Response;

// Struct for a sequence of webots_ros2_msgs__srv__GetBool_Response.
typedef struct webots_ros2_msgs__srv__GetBool_Response__Sequence
{
  webots_ros2_msgs__srv__GetBool_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} webots_ros2_msgs__srv__GetBool_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // WEBOTS_ROS2_MSGS__SRV__DETAIL__GET_BOOL__STRUCT_H_
