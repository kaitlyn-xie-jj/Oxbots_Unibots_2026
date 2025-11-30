// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from webots_ros2_msgs:msg/UrdfRobot.idl
// generated code does not contain a copyright notice
#include "webots_ros2_msgs/msg/detail/urdf_robot__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "webots_ros2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "webots_ros2_msgs/msg/detail/urdf_robot__struct.h"
#include "webots_ros2_msgs/msg/detail/urdf_robot__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/string.h"  // init_pos, name, relative_path_prefix, robot_description, rotation, translation, urdf_path
#include "rosidl_runtime_c/string_functions.h"  // init_pos, name, relative_path_prefix, robot_description, rotation, translation, urdf_path

// forward declare type support functions


using _UrdfRobot__ros_msg_type = webots_ros2_msgs__msg__UrdfRobot;

static bool _UrdfRobot__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _UrdfRobot__ros_msg_type * ros_message = static_cast<const _UrdfRobot__ros_msg_type *>(untyped_ros_message);
  // Field name: name
  {
    const rosidl_runtime_c__String * str = &ros_message->name;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: urdf_path
  {
    const rosidl_runtime_c__String * str = &ros_message->urdf_path;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: robot_description
  {
    const rosidl_runtime_c__String * str = &ros_message->robot_description;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: relative_path_prefix
  {
    const rosidl_runtime_c__String * str = &ros_message->relative_path_prefix;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: translation
  {
    const rosidl_runtime_c__String * str = &ros_message->translation;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: rotation
  {
    const rosidl_runtime_c__String * str = &ros_message->rotation;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: normal
  {
    cdr << (ros_message->normal ? true : false);
  }

  // Field name: box_collision
  {
    cdr << (ros_message->box_collision ? true : false);
  }

  // Field name: init_pos
  {
    const rosidl_runtime_c__String * str = &ros_message->init_pos;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  return true;
}

static bool _UrdfRobot__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _UrdfRobot__ros_msg_type * ros_message = static_cast<_UrdfRobot__ros_msg_type *>(untyped_ros_message);
  // Field name: name
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->name.data) {
      rosidl_runtime_c__String__init(&ros_message->name);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->name,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'name'\n");
      return false;
    }
  }

  // Field name: urdf_path
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->urdf_path.data) {
      rosidl_runtime_c__String__init(&ros_message->urdf_path);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->urdf_path,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'urdf_path'\n");
      return false;
    }
  }

  // Field name: robot_description
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->robot_description.data) {
      rosidl_runtime_c__String__init(&ros_message->robot_description);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->robot_description,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'robot_description'\n");
      return false;
    }
  }

  // Field name: relative_path_prefix
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->relative_path_prefix.data) {
      rosidl_runtime_c__String__init(&ros_message->relative_path_prefix);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->relative_path_prefix,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'relative_path_prefix'\n");
      return false;
    }
  }

  // Field name: translation
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->translation.data) {
      rosidl_runtime_c__String__init(&ros_message->translation);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->translation,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'translation'\n");
      return false;
    }
  }

  // Field name: rotation
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->rotation.data) {
      rosidl_runtime_c__String__init(&ros_message->rotation);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->rotation,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'rotation'\n");
      return false;
    }
  }

  // Field name: normal
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->normal = tmp ? true : false;
  }

  // Field name: box_collision
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->box_collision = tmp ? true : false;
  }

  // Field name: init_pos
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->init_pos.data) {
      rosidl_runtime_c__String__init(&ros_message->init_pos);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->init_pos,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'init_pos'\n");
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_webots_ros2_msgs
size_t get_serialized_size_webots_ros2_msgs__msg__UrdfRobot(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _UrdfRobot__ros_msg_type * ros_message = static_cast<const _UrdfRobot__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name name
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->name.size + 1);
  // field.name urdf_path
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->urdf_path.size + 1);
  // field.name robot_description
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->robot_description.size + 1);
  // field.name relative_path_prefix
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->relative_path_prefix.size + 1);
  // field.name translation
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->translation.size + 1);
  // field.name rotation
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->rotation.size + 1);
  // field.name normal
  {
    size_t item_size = sizeof(ros_message->normal);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name box_collision
  {
    size_t item_size = sizeof(ros_message->box_collision);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name init_pos
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->init_pos.size + 1);

  return current_alignment - initial_alignment;
}

static uint32_t _UrdfRobot__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_webots_ros2_msgs__msg__UrdfRobot(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_webots_ros2_msgs
size_t max_serialized_size_webots_ros2_msgs__msg__UrdfRobot(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: name
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: urdf_path
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: robot_description
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: relative_path_prefix
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: translation
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: rotation
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: normal
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: box_collision
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: init_pos
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = webots_ros2_msgs__msg__UrdfRobot;
    is_plain =
      (
      offsetof(DataType, init_pos) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _UrdfRobot__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_webots_ros2_msgs__msg__UrdfRobot(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_UrdfRobot = {
  "webots_ros2_msgs::msg",
  "UrdfRobot",
  _UrdfRobot__cdr_serialize,
  _UrdfRobot__cdr_deserialize,
  _UrdfRobot__get_serialized_size,
  _UrdfRobot__max_serialized_size
};

static rosidl_message_type_support_t _UrdfRobot__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_UrdfRobot,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, webots_ros2_msgs, msg, UrdfRobot)() {
  return &_UrdfRobot__type_support;
}

#if defined(__cplusplus)
}
#endif
