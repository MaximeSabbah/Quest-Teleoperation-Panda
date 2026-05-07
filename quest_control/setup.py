from setuptools import find_packages, setup

package_name = 'quest_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/quest_control.launch.py']),
        ('share/' + package_name + '/config', ['config/quest_controllers.yaml']),
        ('share/' + package_name + '/config', ['config/fastdds.xml']),
        ('share/' + package_name + '/config', ['config/agimus_control_params.yaml']),
        ('share/' + package_name + '/config', ['config/ocp_definition_file.yaml']),
        ('share/' + package_name + '/config', ['config/ocp_definition_file_old.yaml']),
        ('share/' + package_name + '/urdf', ['urdf/cube.sdf']),
        ('share/' + package_name + '/urdf', ['urdf/obstacles.xacro']),
        ('share/' + package_name + '/urdf', ['urdf/environment.urdf.xacro']),
        ('share/' + package_name + '/urdf', ['urdf/support.urdf.xacro']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gepetto',
    maintainer_email='gepetto@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'quest_streamer = quest_control.quest_streamer:main',
            'quick_replay = quest_control.quick_replay:main',
        ],
    },
)
