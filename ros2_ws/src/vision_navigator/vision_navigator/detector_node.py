"""Detector node — runs a quantized SSD MobileNet V1 (COCO) TFLite model on
incoming camera frames and publishes detections.

Subscribes:
    /camera/image    sensor_msgs/Image (bgr8)

Publishes:
    /detections      vision_msgs/Detection2DArray
                     - bbox in input-image pixel coords
                     - results[0].hypothesis.class_id : COCO label string
                     - results[0].hypothesis.score    : confidence [0..1]

Model artefacts default to the package's share directory; override with the
`model_path` and `labels_path` parameters if you swap models later.
"""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from ai_edge_litert.interpreter import Interpreter
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


def _resolve(default_path: Path, override: str) -> Path:
    return Path(override) if override else default_path


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('detector_node')

        share = Path(get_package_share_directory('vision_navigator'))
        self.declare_parameter('model_path', '')
        self.declare_parameter('labels_path', '')
        self.declare_parameter('score_threshold', 0.5)

        model_path = _resolve(
            share / 'models' / 'detect.tflite',
            self.get_parameter('model_path').get_parameter_value().string_value,
        )
        labels_path = _resolve(
            share / 'models' / 'labelmap.txt',
            self.get_parameter('labels_path').get_parameter_value().string_value,
        )
        self._threshold = self.get_parameter('score_threshold').get_parameter_value().double_value

        self._labels = labels_path.read_text().splitlines()
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._outputs = self._interp.get_output_details()
        _, self._in_h, self._in_w, _ = self._input['shape']

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self._sub = self.create_subscription(Image, '/camera/image', self._on_image, 10)
        self._logged_first = False

        self.get_logger().info(
            f"detector ready: model={model_path.name}, "
            f"input={self._in_w}x{self._in_h}, labels={len(self._labels)}, "
            f"threshold={self._threshold:.2f}"
        )

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self._interp.set_tensor(self._input['index'], input_data)
        self._interp.invoke()

        boxes = self._interp.get_tensor(self._outputs[0]['index'])[0]
        class_ids = self._interp.get_tensor(self._outputs[1]['index'])[0]
        scores = self._interp.get_tensor(self._outputs[2]['index'])[0]
        num = int(self._interp.get_tensor(self._outputs[3]['index'])[0])

        out = Detection2DArray()
        out.header = msg.header

        for i in range(num):
            score = float(scores[i])
            if score < self._threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            cx = float((xmin + xmax) / 2.0 * w)
            cy = float((ymin + ymax) / 2.0 * h)
            box_w = float((xmax - xmin) * w)
            box_h = float((ymax - ymin) * h)

            det = Detection2D()
            det.header = msg.header
            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = box_w
            det.bbox.size_y = box_h

            cls_idx = int(class_ids[i])
            label = self._labels[cls_idx] if 0 <= cls_idx < len(self._labels) else str(cls_idx)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score
            det.results.append(hyp)

            out.detections.append(det)

        self._pub.publish(out)

        if not self._logged_first:
            self.get_logger().info(
                f"first frame processed: '{msg.header.frame_id}' "
                f"-> {len(out.detections)} detection(s) above threshold"
            )
            self._logged_first = True
        else:
            self.get_logger().debug(
                f"frame '{msg.header.frame_id}' -> {len(out.detections)} detection(s)"
            )


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