import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'protoplasm_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # World files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # RViz config
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # Model files (recursive glob for subdirectories)
        *[(os.path.join('share', package_name, 'models', d),
           glob(os.path.join('models', d, '*')))
          for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))],
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='ankith',
    maintainer_email='ankith@todo.com',
    description='Protoplasm SAR Swarm — Gazebo Harmonic simulation with PX4 SITL',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_agent = protoplasm_gazebo.drone_agent:main',
            'swarm_manager = protoplasm_gazebo.swarm_manager:main',
            'px4_bridge = protoplasm_gazebo.px4_bridge:main',
            'gcs_node = protoplasm_gazebo.gcs_node:main',
            'dashboard = protoplasm_gazebo.dashboard:main',
        ],
    },
)
