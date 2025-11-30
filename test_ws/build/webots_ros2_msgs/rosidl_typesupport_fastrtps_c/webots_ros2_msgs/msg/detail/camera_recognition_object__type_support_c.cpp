// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from webots_ros2_msgs:msg/CameraRecognitionObject.idl
// generated code does not contain a copyright notice
#include "webots_ros2_msgs/msg/detail/camera_recognition_object__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "webots_ros2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "webots_ros2_msgs/msg/detail/camera_recognition_object__struct.h"
#include "webots_ros2_msgs/msg/detail/camera_recognition_object__functions.h"
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

#include "geometry_msgs/msg/detail/pose_stamped__functions.h"  // pose
#include "rosidl_runtime_c/string.h"  // model
#include "rosidl_runtime_c/string_functions.h"  // model
#include "std_msgs/msg/detail/color_rgba__functions.h"  // colors
#include "vision_msgs/msg/detail/bounding_box2_d__functions.h"  // bbox

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
size_t get_serialized_size_geometry_msgs__msg__PoseStamped(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
size_t max_serialized_size_geometry_msgs__msg__PoseStamped(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, geometry_msgs, msg, PoseStamped)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
size_t get_serialized_size_std_msgs__msg__ColorRGBA(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
size_t max_serialized_size_std_msgs__msg__ColorRGBA(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, ColorRGBA)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
size_t get_serialized_size_vision_msgs__msg__BoundingBox2D(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
size_t max_serialized_size_vision_msgs__msg__BoundingBox2D(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_webots_ros2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, vision_msgs, msg, BoundingBox2D)();


using _CameraRecognitionObject__ros_msg_type = webots_ros2_msgs__msg__CameraRecognitionObject;

static bool _CameraRecognitionObject__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _CameraRecognitionObject__ros_msg_type * ros_message = static_cast<const _CameraRecognitionObject__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    cdr << ros_message->id;
  }

  // Field name: pose
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, geometry_msgs, msg, PoseStamped
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->pose, cdr))
    {
      return false;
    }
  }

  // Field name: bbox
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, vision_msgs, msg, BoundingBox2D
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->bbox, cdr))
    {
      return false;
    }
  }

  // Field name: colors
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, ColorRGBA
      )()->data);
    size_t size = ros_message->colors.size;
    auto array_ptr = ros_message->colors.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: model
  {
    const rosidl_runtime_c__String * str = &ros_message->model;
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

static bool _CameraRecognitionObject__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _CameraRecognitionObject__ros_msg_type * ros_message = static_cast<_CameraRecognitionObject__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    cdr >> ros_message->id;
  }

  // Field name: pose
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, geometry_msgs, msg, PoseStamped
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->pose))
    {
      return false;
    }
  }

  // Field name: bbox
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, vision_msgs, msg, BoundingBox2D
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->bbox))
    {
      return false;
    }
  }

  // Field name: colors
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, ColorRGBA
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->colors.data) {
      std_msgs__msg__ColorRGBA__Sequence__fini(&ros_message->colors);
    }
    if (!std_msgs__msg__ColorRGBA__Sequence__init(&ros_message->colors, size)) {
      fprintf(stderr, "failed to create array for field 'colors'");
      return false;
    }
    auto array_ptr = ros_message->colors.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: model
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->model.data) {
      rosidl_runtime_c__String__init(&ros_message->model);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->model,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'model'\n");
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_webots_ros2_msgs
size_t get_serialized_size_webots_ros2_msgs__msg__CameraRecognitionObject(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _CameraRecognitionObject__ros_msg_type * ros_message = static_cast<const _CameraRecognitionObject__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name id
  {
    size_t item_size = sizeof(ros_message->id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name pose

  current_alignment += get_serialized_size_geometry_msgs__msg__PoseStamped(
    &(ros_message->pose), current_alignment);
  // field.name bbox

  current_alignment += get_serialized_size_vision_msgs__msg__BoundingBox2D(
    &(ros_message->bbox), current_alignment);
  // field.name colors
  {
    size_t array_size = ros_message->colors.size;
    auto array_ptr = ros_message->colors.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__ColorRGBA(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name model
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->model.size + 1);

  return current_alignment - initial_alignment;
}

static uint32_t _CameraRecognitionObject__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_webots_ros2_msgs__msg__CameraRecognitionObject(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_webots_ros2_msgs
size_t max_serialized_size_webots_ros2_msgs__msg__CameraRecognitionObject(
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

  // member: id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: pose
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_geometry_msgs__msg__PoseStamped(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: bbox
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_vision_msgs__msg__BoundingBox2D(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: colors
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__ColorRGBA(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: model
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
    using DataType = webots_ros2_msgs__msg__CameraRecognitionObject;
    is_plain =
      (
      offsetof(DataType, model) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _CameraRecognitionObject__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_webots_ros2_msgs__msg__CameraRecognitionObject(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_CameraRecognitionObject = {
  "webots_ros2_msgs::msg",
  "CameraRecognitionObject",
  _CameraRecognitionObject__cdr_serialize,
  _CameraRecognitionObject__cdr_deserialize,
  _CameraRecognitionObject__get_serialized_size,
  _CameraRecognitionObject__max_serialized_size
};

static rosidl_message_type_support_t _CameraRecognitionObject__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_CameraRecognitionObject,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, webots_ros2_msgs, msg, CameraRecognitionObject)() {
  return &_CameraRecognitionObject__type_support;
}

#if defined(__cplusplus)
}
#endif
