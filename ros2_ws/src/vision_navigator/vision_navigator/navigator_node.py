"""Navigator node — turns /detections into /cmd_nav decisions.

Stub: implementation lands in the next step.
"""

import rclpy
from rclpy.node import Node


class NavigatorNode(Node):
    def __init__(self) -> None:
        super().__init__('navigator_node')
        self.get_logger().info('navigator_node stub running (no decisions yet).')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
