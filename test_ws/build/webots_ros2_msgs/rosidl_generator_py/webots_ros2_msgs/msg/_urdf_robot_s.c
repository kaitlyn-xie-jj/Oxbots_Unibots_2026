// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from webots_ros2_msgs:msg/UrdfRobot.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "webots_ros2_msgs/msg/detail/urdf_robot__struct.h"
#include "webots_ros2_msgs/msg/detail/urdf_robot__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool webots_ros2_msgs__msg__urdf_robot__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[43];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("webots_ros2_msgs.msg._urdf_robot.UrdfRobot", full_classname_dest, 42) == 0);
  }
  webots_ros2_msgs__msg__UrdfRobot * ros_message = _ros_message;
  {  // name
    PyObject * field = PyObject_GetAttrString(_pymsg, "name");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->name, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // urdf_path
    PyObject * field = PyObject_GetAttrString(_pymsg, "urdf_path");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->urdf_path, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // robot_description
    PyObject * field = PyObject_GetAttrString(_pymsg, "robot_description");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->robot_description, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // relative_path_prefix
    PyObject * field = PyObject_GetAttrString(_pymsg, "relative_path_prefix");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->relative_path_prefix, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // translation
    PyObject * field = PyObject_GetAttrString(_pymsg, "translation");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->translation, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // rotation
    PyObject * field = PyObject_GetAttrString(_pymsg, "rotation");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->rotation, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // normal
    PyObject * field = PyObject_GetAttrString(_pymsg, "normal");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->normal = (Py_True == field);
    Py_DECREF(field);
  }
  {  // box_collision
    PyObject * field = PyObject_GetAttrString(_pymsg, "box_collision");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->box_collision = (Py_True == field);
    Py_DECREF(field);
  }
  {  // init_pos
    PyObject * field = PyObject_GetAttrString(_pymsg, "init_pos");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->init_pos, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * webots_ros2_msgs__msg__urdf_robot__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of UrdfRobot */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("webots_ros2_msgs.msg._urdf_robot");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "UrdfRobot");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  webots_ros2_msgs__msg__UrdfRobot * ros_message = (webots_ros2_msgs__msg__UrdfRobot *)raw_ros_message;
  {  // name
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->name.data,
      strlen(ros_message->name.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "name", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // urdf_path
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->urdf_path.data,
      strlen(ros_message->urdf_path.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "urdf_path", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // robot_description
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->robot_description.data,
      strlen(ros_message->robot_description.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "robot_description", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // relative_path_prefix
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->relative_path_prefix.data,
      strlen(ros_message->relative_path_prefix.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "relative_path_prefix", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // translation
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->translation.data,
      strlen(ros_message->translation.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "translation", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rotation
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->rotation.data,
      strlen(ros_message->rotation.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "rotation", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // normal
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->normal ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "normal", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // box_collision
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->box_collision ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "box_collision", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // init_pos
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->init_pos.data,
      strlen(ros_message->init_pos.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "init_pos", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
