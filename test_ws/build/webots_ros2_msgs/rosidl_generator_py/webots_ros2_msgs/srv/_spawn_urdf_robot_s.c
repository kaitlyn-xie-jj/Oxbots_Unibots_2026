// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from webots_ros2_msgs:srv/SpawnUrdfRobot.idl
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
#include "webots_ros2_msgs/srv/detail/spawn_urdf_robot__struct.h"
#include "webots_ros2_msgs/srv/detail/spawn_urdf_robot__functions.h"

bool webots_ros2_msgs__msg__urdf_robot__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * webots_ros2_msgs__msg__urdf_robot__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool webots_ros2_msgs__srv__spawn_urdf_robot__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[62];
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
    assert(strncmp("webots_ros2_msgs.srv._spawn_urdf_robot.SpawnUrdfRobot_Request", full_classname_dest, 61) == 0);
  }
  webots_ros2_msgs__srv__SpawnUrdfRobot_Request * ros_message = _ros_message;
  {  // robot
    PyObject * field = PyObject_GetAttrString(_pymsg, "robot");
    if (!field) {
      return false;
    }
    if (!webots_ros2_msgs__msg__urdf_robot__convert_from_py(field, &ros_message->robot)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * webots_ros2_msgs__srv__spawn_urdf_robot__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SpawnUrdfRobot_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("webots_ros2_msgs.srv._spawn_urdf_robot");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SpawnUrdfRobot_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  webots_ros2_msgs__srv__SpawnUrdfRobot_Request * ros_message = (webots_ros2_msgs__srv__SpawnUrdfRobot_Request *)raw_ros_message;
  {  // robot
    PyObject * field = NULL;
    field = webots_ros2_msgs__msg__urdf_robot__convert_to_py(&ros_message->robot);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "robot", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "webots_ros2_msgs/srv/detail/spawn_urdf_robot__struct.h"
// already included above
// #include "webots_ros2_msgs/srv/detail/spawn_urdf_robot__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool webots_ros2_msgs__srv__spawn_urdf_robot__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[63];
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
    assert(strncmp("webots_ros2_msgs.srv._spawn_urdf_robot.SpawnUrdfRobot_Response", full_classname_dest, 62) == 0);
  }
  webots_ros2_msgs__srv__SpawnUrdfRobot_Response * ros_message = _ros_message;
  {  // success
    PyObject * field = PyObject_GetAttrString(_pymsg, "success");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->success = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * webots_ros2_msgs__srv__spawn_urdf_robot__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SpawnUrdfRobot_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("webots_ros2_msgs.srv._spawn_urdf_robot");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SpawnUrdfRobot_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  webots_ros2_msgs__srv__SpawnUrdfRobot_Response * ros_message = (webots_ros2_msgs__srv__SpawnUrdfRobot_Response *)raw_ros_message;
  {  // success
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->success ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "success", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
