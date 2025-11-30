// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from webots_ros2_msgs:srv/SpawnNodeFromString.idl
// generated code does not contain a copyright notice

#ifndef WEBOTS_ROS2_MSGS__SRV__DETAIL__SPAWN_NODE_FROM_STRING__BUILDER_HPP_
#define WEBOTS_ROS2_MSGS__SRV__DETAIL__SPAWN_NODE_FROM_STRING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "webots_ros2_msgs/srv/detail/spawn_node_from_string__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace webots_ros2_msgs
{

namespace srv
{

namespace builder
{

class Init_SpawnNodeFromString_Request_data
{
public:
  Init_SpawnNodeFromString_Request_data()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::webots_ros2_msgs::srv::SpawnNodeFromString_Request data(::webots_ros2_msgs::srv::SpawnNodeFromString_Request::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::webots_ros2_msgs::srv::SpawnNodeFromString_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::webots_ros2_msgs::srv::SpawnNodeFromString_Request>()
{
  return webots_ros2_msgs::srv::builder::Init_SpawnNodeFromString_Request_data();
}

}  // namespace webots_ros2_msgs


namespace webots_ros2_msgs
{

namespace srv
{

namespace builder
{

class Init_SpawnNodeFromString_Response_success
{
public:
  Init_SpawnNodeFromString_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::webots_ros2_msgs::srv::SpawnNodeFromString_Response success(::webots_ros2_msgs::srv::SpawnNodeFromString_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::webots_ros2_msgs::srv::SpawnNodeFromString_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::webots_ros2_msgs::srv::SpawnNodeFromString_Response>()
{
  return webots_ros2_msgs::srv::builder::Init_SpawnNodeFromString_Response_success();
}

}  // namespace webots_ros2_msgs

#endif  // WEBOTS_ROS2_MSGS__SRV__DETAIL__SPAWN_NODE_FROM_STRING__BUILDER_HPP_
