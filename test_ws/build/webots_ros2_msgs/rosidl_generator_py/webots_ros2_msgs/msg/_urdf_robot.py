# generated from rosidl_generator_py/resource/_idl.py.em
# with input from webots_ros2_msgs:msg/UrdfRobot.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_UrdfRobot(type):
    """Metaclass of message 'UrdfRobot'."""

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
                'webots_ros2_msgs.msg.UrdfRobot')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__urdf_robot
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__urdf_robot
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__urdf_robot
            cls._TYPE_SUPPORT = module.type_support_msg__msg__urdf_robot
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__urdf_robot

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class UrdfRobot(metaclass=Metaclass_UrdfRobot):
    """Message class 'UrdfRobot'."""

    __slots__ = [
        '_name',
        '_urdf_path',
        '_robot_description',
        '_relative_path_prefix',
        '_translation',
        '_rotation',
        '_normal',
        '_box_collision',
        '_init_pos',
    ]

    _fields_and_field_types = {
        'name': 'string',
        'urdf_path': 'string',
        'robot_description': 'string',
        'relative_path_prefix': 'string',
        'translation': 'string',
        'rotation': 'string',
        'normal': 'boolean',
        'box_collision': 'boolean',
        'init_pos': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.name = kwargs.get('name', str())
        self.urdf_path = kwargs.get('urdf_path', str())
        self.robot_description = kwargs.get('robot_description', str())
        self.relative_path_prefix = kwargs.get('relative_path_prefix', str())
        self.translation = kwargs.get('translation', str())
        self.rotation = kwargs.get('rotation', str())
        self.normal = kwargs.get('normal', bool())
        self.box_collision = kwargs.get('box_collision', bool())
        self.init_pos = kwargs.get('init_pos', str())

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
        if self.name != other.name:
            return False
        if self.urdf_path != other.urdf_path:
            return False
        if self.robot_description != other.robot_description:
            return False
        if self.relative_path_prefix != other.relative_path_prefix:
            return False
        if self.translation != other.translation:
            return False
        if self.rotation != other.rotation:
            return False
        if self.normal != other.normal:
            return False
        if self.box_collision != other.box_collision:
            return False
        if self.init_pos != other.init_pos:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def name(self):
        """Message field 'name'."""
        return self._name

    @name.setter
    def name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'name' field must be of type 'str'"
        self._name = value

    @builtins.property
    def urdf_path(self):
        """Message field 'urdf_path'."""
        return self._urdf_path

    @urdf_path.setter
    def urdf_path(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'urdf_path' field must be of type 'str'"
        self._urdf_path = value

    @builtins.property
    def robot_description(self):
        """Message field 'robot_description'."""
        return self._robot_description

    @robot_description.setter
    def robot_description(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'robot_description' field must be of type 'str'"
        self._robot_description = value

    @builtins.property
    def relative_path_prefix(self):
        """Message field 'relative_path_prefix'."""
        return self._relative_path_prefix

    @relative_path_prefix.setter
    def relative_path_prefix(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'relative_path_prefix' field must be of type 'str'"
        self._relative_path_prefix = value

    @builtins.property
    def translation(self):
        """Message field 'translation'."""
        return self._translation

    @translation.setter
    def translation(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'translation' field must be of type 'str'"
        self._translation = value

    @builtins.property
    def rotation(self):
        """Message field 'rotation'."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'rotation' field must be of type 'str'"
        self._rotation = value

    @builtins.property
    def normal(self):
        """Message field 'normal'."""
        return self._normal

    @normal.setter
    def normal(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'normal' field must be of type 'bool'"
        self._normal = value

    @builtins.property
    def box_collision(self):
        """Message field 'box_collision'."""
        return self._box_collision

    @box_collision.setter
    def box_collision(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'box_collision' field must be of type 'bool'"
        self._box_collision = value

    @builtins.property
    def init_pos(self):
        """Message field 'init_pos'."""
        return self._init_pos

    @init_pos.setter
    def init_pos(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'init_pos' field must be of type 'str'"
        self._init_pos = value
