"""Detector node — runs a quantized TFLite model on /camera/image.

Stub: implementation lands in the next step.
"""

import rclpy
from rclpy.node import Node


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('detector_node')
        self.get_logger().info('detector_node stub running (no inference yet).')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
