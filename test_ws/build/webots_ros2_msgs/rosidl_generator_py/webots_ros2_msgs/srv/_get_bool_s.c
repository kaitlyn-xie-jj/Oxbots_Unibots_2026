// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from webots_ros2_msgs:srv/GetBool.idl
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
#include "webots_ros2_msgs/srv/detail/get_bool__struct.h"
#include "webots_ros2_msgs/srv/detail/get_bool__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool webots_ros2_msgs__srv__get_bool__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[47];
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
    assert(strncmp("webots_ros2_msgs.srv._get_bool.GetBool_Request", full_classname_dest, 46) == 0);
  }
  webots_ros2_msgs__srv__GetBool_Request * ros_message = _ros_message;
  {  // ask
    PyObject * field = PyObject_GetAttrString(_pymsg, "ask");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->ask = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * webots_ros2_msgs__srv__get_bool__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GetBool_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("webots_ros2_msgs.srv._get_bool");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GetBool_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  webots_ros2_msgs__srv__GetBool_Request * ros_message = (webots_ros2_msgs__srv__GetBool_Request *)raw_ros_message;
  {  // ask
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->ask ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ask", field);
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
// #include "webots_ros2_msgs/srv/detail/get_bool__struct.h"
// already included above
// #include "webots_ros2_msgs/srv/detail/get_bool__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool webots_ros2_msgs__srv__get_bool__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[48];
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
    assert(strncmp("webots_ros2_msgs.srv._get_bool.GetBool_Response", full_classname_dest, 47) == 0);
  }
  webots_ros2_msgs__srv__GetBool_Response * ros_message = _ros_message;
  {  // value
    PyObject * field = PyObject_GetAttrString(_pymsg, "value");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->value = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * webots_ros2_msgs__srv__get_bool__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GetBool_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("webots_ros2_msgs.srv._get_bool");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GetBool_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  webots_ros2_msgs__srv__GetBool_Response * ros_message = (webots_ros2_msgs__srv__GetBool_Response *)raw_ros_message;
  {  // value
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->value ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "value", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
