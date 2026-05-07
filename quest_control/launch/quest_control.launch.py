from launch import LaunchContext, LaunchDescription
from launch.actions import (
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
    DeclareLaunchArgument,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, FindExecutable
from launch_ros.parameter_descriptions import ParameterValue

from agimus_demos_common.launch_utils import (
    generate_default_franka_args,
    generate_include_launch,
    get_use_sim_time,
)
from agimus_demos_common.static_transform_publisher_node import (
    static_transform_publisher_node,
)


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:
    #print('----------------IN LAUNCH SETUP-------------------------')

    rviz_config_path = PathJoinSubstitution(
        [
            FindPackageShare("quest_control"),
            "rviz",
            "config.rviz",
        ]
    )


    franka_robot_launch = generate_include_launch("franka_common_lfc.launch.py")#, extra_launch_arguments={"rviz_config_path": rviz_config_path})
    ocp_choice_arg = LaunchConfiguration("ocp")
    use_collision_detection = (
        context.perform_substitution(ocp_choice_arg).lower()
        == "custom_with_collision_avoidance"
    )

    agimus_controller_yaml = PathJoinSubstitution(
        [
            FindPackageShare("quest_control"),
            "config",
            "agimus_control_params.yaml",
        ]
    )

    if use_collision_detection:
        extra_params = {
            "ocp": {
                "definition_yaml_file": "package://quest_control/config/ocp_definition_file.yaml"
            }
        }
    else:
        extra_params = {}
    
    
    wait_for_non_zero_joints_node = Node(
        package="agimus_demos_common",
        executable="wait_for_non_zero_joints_node",
        parameters=[get_use_sim_time()],
        output="screen",
    )  

    #I dont actually get this node. What does an MPC input look like? I will try LFC directly and do IK myself 
    agimus_controller_node = Node(
        package="agimus_controller_ros",
        executable="agimus_controller_node",
        name="agimus_controller_node",
        parameters=[
            get_use_sim_time(),
            agimus_controller_yaml,
            extra_params,
            {"trajectory_buffer": "constant"},
        ],
        output="screen",
        remappings=[("robot_description", "robot_description_with_collision")],
    )


    simple_trajectory_publisher_node = Node(
        package="quest_control",
        executable="quest_streamer",
        parameters=[
            get_use_sim_time(),
            {"task_name": LaunchConfiguration("task_name")},
        ],
        arguments=[],
        output="screen",
    )

    environment_description = ParameterValue(
        Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("agimus_demo_03_mpc_dummy_traj"), #agimus_demo_03_mpc_dummy_traj
                        "urdf",
                        "obstacles.xacro", #envrionmenturdf.xacro
                    ]
                ),
                # Convert dict to list of parameters
            ]
        ),
        value_type=str,
    )
    environment_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="environment_publisher",
        output="screen",
        remappings=[("robot_description", "environment_description")],
        parameters=[{"robot_description": environment_description}],
    )

    tf_node = static_transform_publisher_node(
        frame_id="fer_link0",
        child_frame_id="obstacle1",
    )



    return [
        franka_robot_launch,
        wait_for_non_zero_joints_node,
        tf_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=wait_for_non_zero_joints_node,
                on_exit=[
                    agimus_controller_node,
                    environment_publisher_node,
                ],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=agimus_controller_node,
                on_start=TimerAction(
                    period=5.0,
                    actions=[simple_trajectory_publisher_node],
                ),
            )
        ),
    ]


    #This would work if Agimus would not want to have a weird collision thingy
    #return [
    #franka_robot_launch,
    #wait_for_non_zero_joints_node,
    #agimus_controller_node,
    #simple_trajectory_publisher_node,
    #]
#[
        #franka_robot_launch,
        #wait_for_non_zero_joints_node,
        #RegisterEventHandler(
        #    event_handler=OnProcessExit(
        #        target_action=wait_for_non_zero_joints_node,
        #        on_exit=[agimus_controller_node],
        #    )
        #),
        #RegisterEventHandler(
        #    event_handler=OnProcessStart(
        #        target_action=agimus_controller_node,
        #        on_start=TimerAction(
        #            period=5.0,
        #            actions=[simple_trajectory_publisher_node],
        #        ),
        #    )
        #),


    #]


def generate_launch_description():
    ocp_choice = DeclareLaunchArgument(
        "ocp",
        default_value="custom_with_collision_avoidance",
        description="Select the ocp to use. Either the default one or the one from this package that does collision avoidance.",
        choices=["default_ocp", "custom_with_collision_avoidance"],
    )
    task_name_arg = DeclareLaunchArgument(
        "task_name",
        default_value="red_cube_hybrid",
        description="Dataset folder name and task label. Use underscores: grab_the_red_cube",
    )
    fastdds_config = PathJoinSubstitution(
        [FindPackageShare("quest_control"), "config", "fastdds.xml"]
    )
    return LaunchDescription(
        [
            ocp_choice,
            task_name_arg,
            SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", fastdds_config),
        ]
        + generate_default_franka_args()
        + [OpaqueFunction(function=launch_setup)]
    )

    #return LaunchDescription(generate_default_franka_args() + [OpaqueFunction(function=launch_setup)])