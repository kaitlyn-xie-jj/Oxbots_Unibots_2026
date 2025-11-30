// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from webots_ros2_msgs:srv/EmitterSendString.idl
// generated code does not contain a copyright notice

#ifndef WEBOTS_ROS2_MSGS__SRV__DETAIL__EMITTER_SEND_STRING__TRAITS_HPP_
#define WEBOTS_ROS2_MSGS__SRV__DETAIL__EMITTER_SEND_STRING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "webots_ros2_msgs/srv/detail/emitter_send_string__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace webots_ros2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const EmitterSendString_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: value
  {
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const EmitterSendString_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: value
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const EmitterSendString_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace webots_ros2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use webots_ros2_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const webots_ros2_msgs::srv::EmitterSendString_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  webots_ros2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use webots_ros2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const webots_ros2_msgs::srv::EmitterSendString_Request & msg)
{
  return webots_ros2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<webots_ros2_msgs::srv::EmitterSendString_Request>()
{
  return "webots_ros2_msgs::srv::EmitterSendString_Request";
}

template<>
inline const char * name<webots_ros2_msgs::srv::EmitterSendString_Request>()
{
  return "webots_ros2_msgs/srv/EmitterSendString_Request";
}

template<>
struct has_fixed_size<webots_ros2_msgs::srv::EmitterSendString_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<webots_ros2_msgs::srv::EmitterSendString_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<webots_ros2_msgs::srv::EmitterSendString_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace webots_ros2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const EmitterSendString_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: result
  {
    out << "result: ";
    rosidl_generator_traits::value_to_yaml(msg.result, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const EmitterSendString_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: result
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "result: ";
    rosidl_generator_traits::value_to_yaml(msg.result, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const EmitterSendString_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace webots_ros2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use webots_ros2_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const webots_ros2_msgs::srv::EmitterSendString_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  webots_ros2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use webots_ros2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const webots_ros2_msgs::srv::EmitterSendString_Response & msg)
{
  return webots_ros2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<webots_ros2_msgs::srv::EmitterSendString_Response>()
{
  return "webots_ros2_msgs::srv::EmitterSendString_Response";
}

template<>
inline const char * name<webots_ros2_msgs::srv::EmitterSendString_Response>()
{
  return "webots_ros2_msgs/srv/EmitterSendString_Response";
}

template<>
struct has_fixed_size<webots_ros2_msgs::srv::EmitterSendString_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<webots_ros2_msgs::srv::EmitterSendString_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<webots_ros2_msgs::srv::EmitterSendString_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<webots_ros2_msgs::srv::EmitterSendString>()
{
  return "webots_ros2_msgs::srv::EmitterSendString";
}

template<>
inline const char * name<webots_ros2_msgs::srv::EmitterSendString>()
{
  return "webots_ros2_msgs/srv/EmitterSendString";
}

template<>
struct has_fixed_size<webots_ros2_msgs::srv::EmitterSendString>
  : std::integral_constant<
    bool,
    has_fixed_size<webots_ros2_msgs::srv::EmitterSendString_Request>::value &&
    has_fixed_size<webots_ros2_msgs::srv::EmitterSendString_Response>::value
  >
{
};

template<>
struct has_bounded_size<webots_ros2_msgs::srv::EmitterSendString>
  : std::integral_constant<
    bool,
    has_bounded_size<webots_ros2_msgs::srv::EmitterSendString_Request>::value &&
    has_bounded_size<webots_ros2_msgs::srv::EmitterSendString_Response>::value
  >
{
};

template<>
struct is_service<webots_ros2_msgs::srv::EmitterSendString>
  : std::true_type
{
};

template<>
struct is_service_request<webots_ros2_msgs::srv::EmitterSendString_Request>
  : std::true_type
{
};

template<>
struct is_service_response<webots_ros2_msgs::srv::EmitterSendString_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // WEBOTS_ROS2_MSGS__SRV__DETAIL__EMITTER_SEND_STRING__TRAITS_HPP_
