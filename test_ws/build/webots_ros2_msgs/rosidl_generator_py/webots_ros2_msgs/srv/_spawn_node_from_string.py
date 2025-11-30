# generated from rosidl_generator_py/resource/_idl.py.em
# with input from webots_ros2_msgs:srv/SpawnNodeFromString.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SpawnNodeFromString_Request(type):
    """Metaclass of message 'SpawnNodeFromString_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('webots_ros2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'webots_ros2_msgs.srv.SpawnNodeFromString_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__spawn_node_from_string__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__spawn_node_from_string__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__spawn_node_from_string__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__spawn_node_from_string__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__spawn_node_from_string__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SpawnNodeFromString_Request(metaclass=Metaclass_SpawnNodeFromString_Request):
    """Message class 'SpawnNodeFromString_Request'."""

    __slots__ = [
        '_data',
    ]

    _fields_and_field_types = {
        'data': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.data = kwargs.get('data', str())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.data != other.data:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def data(self):
        """Message field 'data'."""
        return self._data

    @data.setter
    def data(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'data' field must be of type 'str'"
        self._data = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_SpawnNodeFromString_Response(type):
    """Metaclass of message 'SpawnNodeFromString_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('webots_ros2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'webots_ros2_msgs.srv.SpawnNodeFromString_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__spawn_node_from_string__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__spawn_node_from_string__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__spawn_node_from_string__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__spawn_node_from_string__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__spawn_node_from_string__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SpawnNodeFromString_Response(metaclass=Metaclass_SpawnNodeFromString_Response):
    """Message class 'SpawnNodeFromString_Response'."""

    __slots__ = [
        '_success',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.success != other.success:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value


class Metaclass_SpawnNodeFromString(type):
    """Metaclass of service 'SpawnNodeFromString'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('webots_ros2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'webots_ros2_msgs.srv.SpawnNodeFromString')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__spawn_node_from_string

            from webots_ros2_msgs.srv import _spawn_node_from_string
            if _spawn_node_from_string.Metaclass_SpawnNodeFromString_Request._TYPE_SUPPORT is None:
                _spawn_node_from_string.Metaclass_SpawnNodeFromString_Request.__import_type_support__()
            if _spawn_node_from_string.Metaclass_SpawnNodeFromString_Response._TYPE_SUPPORT is None:
                _spawn_node_from_string.Metaclass_SpawnNodeFromString_Response.__import_type_support__()


class SpawnNodeFromString(metaclass=Metaclass_SpawnNodeFromString):
    from webots_ros2_msgs.srv._spawn_node_from_string import SpawnNodeFromString_Request as Request
    from webots_ros2_msgs.srv._spawn_node_from_string import SpawnNodeFromString_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
