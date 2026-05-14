"""Camera node — streams test images to /camera/image at a fixed rate.

This stands in for a real camera driver. By publishing static frames at 10 Hz
the rest of the pipeline can be developed and benchmarked without a robot,
a webcam, or Gazebo. To switch to a live camera later, replace `_load_frames`
with an OpenCV `VideoCapture` and keep the rest of the node identical.
"""

from pathlib import Path

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__('camera_node')

        self.declare_parameter('image_dir', 'test_images')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('loop', True)

        image_dir_param = self.get_parameter('image_dir').get_parameter_value().string_value
        rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._loop = self.get_parameter('loop').get_parameter_value().bool_value

        image_dir = Path(image_dir_param)
        if not image_dir.is_absolute():
            image_dir = Path(get_package_share_directory('vision_navigator')) / image_dir

        self._frames = self._load_frames(image_dir)
        if not self._frames:
            self.get_logger().error(f"No images found in '{image_dir}'. Camera node will idle.")
        else:
            self.get_logger().info(f"Loaded {len(self._frames)} frame(s) from '{image_dir}'.")

        self._bridge = CvBridge()
        self._index = 0

        self._publisher = self.create_publisher(Image, '/camera/image', 10)
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_next_frame)

    def _load_frames(self, directory: Path) -> list:
        if not directory.is_dir():
            return []
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        frames = []
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn(f"Could not decode {path.name}, skipping.")
                continue
            frames.append((path.name, img))
        return frames

    def _publish_next_frame(self) -> None:
        if not self._frames:
            return

        if self._index >= len(self._frames):
            if not self._loop:
                self.get_logger().info("Reached end of frame list; loop=False, stopping timer.")
                self._timer.cancel()
                return
            self._index = 0

        name, frame = self._frames[self._index]
        msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = name
        self._publisher.publish(msg)
        self.get_logger().debug(f"Published frame {self._index}: {name}")
        self._index += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
