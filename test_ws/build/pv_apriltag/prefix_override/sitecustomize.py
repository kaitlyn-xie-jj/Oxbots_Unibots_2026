import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/verity-xie/Unibots 2026/test_ws/install/pv_apriltag'
