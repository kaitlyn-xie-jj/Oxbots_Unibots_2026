// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from webots_ros2_msgs:srv/SpawnUrdfRobot.idl
// generated code does not contain a copyright notice

#ifndef WEBOTS_ROS2_MSGS__SRV__DETAIL__SPAWN_URDF_ROBOT__TRAITS_HPP_
#define WEBOTS_ROS2_MSGS__SRV__DETAIL__SPAWN_URDF_ROBOT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "webots_ros2_msgs/srv/detail/spawn_urdf_robot__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'robot'
#include "webots_ros2_msgs/msg/detail/urdf_robot__traits.hpp"

namespace webots_ros2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SpawnUrdfRobot_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot
  {
    out << "robot: ";
    to_flow_style_yaml(msg.robot, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SpawnUrdfRobot_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: robot
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot:\n";
    to_block_style_yaml(msg.robot, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SpawnUrdfRobot_Request & msg, bool use_flow_style = false)
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
  const webots_ros2_msgs::srv::SpawnUrdfRobot_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  webots_ros2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use webots_ros2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const webots_ros2_msgs::srv::SpawnUrdfRobot_Request & msg)
{
  return webots_ros2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>()
{
  return "webots_ros2_msgs::srv::SpawnUrdfRobot_Request";
}

template<>
inline const char * name<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>()
{
  return "webots_ros2_msgs/srv/SpawnUrdfRobot_Request";
}

template<>
struct has_fixed_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>
  : std::integral_constant<bool, has_fixed_size<webots_ros2_msgs::msg::UrdfRobot>::value> {};

template<>
struct has_bounded_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>
  : std::integral_constant<bool, has_bounded_size<webots_ros2_msgs::msg::UrdfRobot>::value> {};

template<>
struct is_message<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace webots_ros2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SpawnUrdfRobot_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SpawnUrdfRobot_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SpawnUrdfRobot_Response & msg, bool use_flow_style = false)
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
  const webots_ros2_msgs::srv::SpawnUrdfRobot_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  webots_ros2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use webots_ros2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const webots_ros2_msgs::srv::SpawnUrdfRobot_Response & msg)
{
  return webots_ros2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>()
{
  return "webots_ros2_msgs::srv::SpawnUrdfRobot_Response";
}

template<>
inline const char * name<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>()
{
  return "webots_ros2_msgs/srv/SpawnUrdfRobot_Response";
}

template<>
struct has_fixed_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<webots_ros2_msgs::srv::SpawnUrdfRobot>()
{
  return "webots_ros2_msgs::srv::SpawnUrdfRobot";
}

template<>
inline const char * name<webots_ros2_msgs::srv::SpawnUrdfRobot>()
{
  return "webots_ros2_msgs/srv/SpawnUrdfRobot";
}

template<>
struct has_fixed_size<webots_ros2_msgs::srv::SpawnUrdfRobot>
  : std::integral_constant<
    bool,
    has_fixed_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>::value &&
    has_fixed_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>::value
  >
{
};

template<>
struct has_bounded_size<webots_ros2_msgs::srv::SpawnUrdfRobot>
  : std::integral_constant<
    bool,
    has_bounded_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>::value &&
    has_bounded_size<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>::value
  >
{
};

template<>
struct is_service<webots_ros2_msgs::srv::SpawnUrdfRobot>
  : std::true_type
{
};

template<>
struct is_service_request<webots_ros2_msgs::srv::SpawnUrdfRobot_Request>
  : std::true_type
{
};

template<>
struct is_service_response<webots_ros2_msgs::srv::SpawnUrdfRobot_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // WEBOTS_ROS2_MSGS__SRV__DETAIL__SPAWN_URDF_ROBOT__TRAITS_HPP_
