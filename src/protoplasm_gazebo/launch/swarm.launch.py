import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    px4_dir = os.path.expanduser('~/swarm_drone_ws/PX4-Autopilot')
    build_dir = os.path.join(px4_dir, 'build/px4_sitl_default')
    
    # ── Configuration ──
    NUM_DRONES = 10

    # Generate spawn positions in a 2x5 grid near the GCS (-575, 0)
    # Row 0: y = -20, -10, 0, 10, 20   at x = -560
    # Row 1: y = -20, -10, 0, 10, 20   at x = -565
    poses = []
    for i in range(NUM_DRONES):
        col = i % 5
        row = i // 5
        x = -560.0 + (row * -5.0)
        y = -20.0 + (col * 10.0)
        poses.append((x, y))

    # 1. MicroXRCEAgent (Bridge between PX4 uORB and ROS 2 DDS)
    micro_xrce = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='log'
    )
    
    # 2. Main PX4 Instance (ID 0) + Gazebo Physics Engine
    # Running headlessly saves massive amounts of CPU/GPU since we use the Pygame dashboard
    px4_main = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500'],
        cwd=px4_dir,
        additional_env={
            'PX4_GZ_WORLD': 'disaster_zone',
            'PX4_GZ_MODEL_POSE': f'{poses[0][0]},{poses[0][1]},0.2,0,0,0',
            # Force Gazebo to use the NVIDIA RTX 4060 GPU
            '__NV_PRIME_RENDER_OFFLOAD': '1',
            '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
            'LIBGL_ALWAYS_SOFTWARE': '0'
        },
        output='log'
    )
    
    # 3. Standalone PX4 Instances (IDs 1 to NUM_DRONES-1)
    px4_instances = []
    for i in range(1, NUM_DRONES):
        cmd = [os.path.join(build_dir, 'bin/px4'), '-i', str(i)]
        env = {
            'PX4_GZ_STANDALONE': '1',
            'PX4_SIM_MODEL': 'gz_x500',
            'PX4_GZ_WORLD': 'disaster_zone',
            'PX4_GZ_MODEL_POSE': f"{poses[i][0]},{poses[i][1]},0.2,0,0,0"
        }
        process = ExecuteProcess(
            cmd=cmd, cwd=build_dir,
            additional_env=env, output='log'
        )
        # Stagger each PX4 spawn by 3 seconds
        px4_instances.append(TimerAction(period=5.0 + (i * 3.0), actions=[process]))

    # 4. Protoplasm AI Drone Agents
    agents = []
    for i in range(NUM_DRONES):
        namespace = "" if i == 0 else f"/px4_{i}"
        agent = Node(
            package='protoplasm_gazebo',
            executable='drone_agent',
            name=f'drone_agent_{i}',
            arguments=[str(i), namespace],
            parameters=[{
                'spawn_x': poses[i][0],
                'spawn_y': poses[i][1],
            }],
            output='screen'
        )
        # Last PX4 spawns at ~32s; start AI agents at 35s, staggered by 1s each
        agents.append(TimerAction(period=35.0 + (i * 1.0), actions=[agent]))
        
    # 5. Tactical Pygame Dashboard
    dashboard = TimerAction(
        period=36.0,
        actions=[Node(
            package='protoplasm_gazebo',
            executable='dashboard',
            name='dashboard',
            output='screen'
        )]
    )

    return LaunchDescription([
        micro_xrce,
        px4_main,
        *px4_instances,
        *agents,
        dashboard
    ])
