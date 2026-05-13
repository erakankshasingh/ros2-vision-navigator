from setuptools import find_packages, setup

package_name = 'vision_navigator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Akanksha Singh',
    maintainer_email='erakankshasingh@gmail.com',
    description='On-device vision navigation pipeline using ROS2 and a quantized TFLite model.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = vision_navigator.camera_node:main',
            'detector_node = vision_navigator.detector_node:main',
            'navigator_node = vision_navigator.navigator_node:main',
        ],
    },
)
