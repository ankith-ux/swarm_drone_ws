# Protoplasm Swarm — Installation & Run Instructions

These instructions are tailored for **Ubuntu 24.04** running **ROS 2 Jazzy**, which matches the development environment used for this PoC. They cover the setup of Gazebo Harmonic, PX4 Autopilot SITL, MicroXRCE-DDS, and the Protoplasm workspace.

---

## 1. Prerequisites & System Setup

### Install ROS 2 Jazzy & Gazebo Harmonic
Ensure you have **ROS 2 Jazzy** installed on Ubuntu 24.04. Then, install **Gazebo Harmonic** and the necessary ROS-Gazebo bridges:
```bash
sudo apt update
# Install the ROS 2 Jazzy desktop and the Gazebo Harmonic bridge
sudo apt install -y ros-jazzy-desktop ros-jazzy-ros-gz python3-pip

# Install the pygame dependency used for the tactical dashboard
pip3 install --user pygame numpy
```
*(Note: Gazebo Harmonic is the recommended simulator for PX4 v1.15+ and works natively with ROS 2 Jazzy).*

### Install Micro XRCE-DDS Agent
The Micro XRCE-DDS Agent is required to bridge PX4's uORB messages to ROS 2 DDS topics.
```bash
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```

---

## 2. Build PX4 Autopilot (SITL)

The swarm uses PX4 Autopilot running in Software-In-The-Loop (SITL) mode.
```bash
cd ~/swarm_drone_ws
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot

# Install PX4 dependencies (this script provided by PX4 sets up the Ubuntu toolchain)
bash ./Tools/setup/ubuntu.sh

# Build the PX4 SITL target for Gazebo
# Note: Building this the first time may take several minutes.
make px4_sitl gz_x500
```

---

## 3. Build the Protoplasm Workspace

Clone or extract the `protoplasm_gazebo` package into your ROS 2 workspace's `src` directory, then build it using `colcon`.

```bash
cd ~/swarm_drone_ws
# Ensure the protoplasm_gazebo package is inside the src/ directory here

# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Build the workspace
colcon build --packages-select protoplasm_gazebo
```

---

## 4. Running the Simulation

The primary launch file for the 10-drone swarm with the Pygame tactical dashboard is `swarm.launch.py`. This launch file automatically handles starting the MicroXRCE Agent, the PX4 instances, Gazebo Harmonic, the ROS 2 drone agents, and the dashboard.

Open a new terminal and run:

```bash
cd ~/swarm_drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Launch the full 10-drone swarm scenario
ros2 launch protoplasm_gazebo swarm.launch.py
```

### What to Expect:
1. **Gazebo** will launch headlessly in the background (configured for GPU offloading for performance).
2. **10 PX4 SITL instances** will boot up, staggered by a few seconds.
3. The **Tactical Dashboard** (Pygame window) will open.
4. After about 35 seconds, the drone agents will arm, take off, and begin executing the autonomous SPREAD regime to explore the 1000x1000m arena.
5. As drones discover Points of Interest (POIs), you will see their regimes shift to CONVERGE and SOLIDIFY, and detections will be relayed back to the GCS via the multi-hop daisy-chain mesh.

### 5. Running the Ground Control Station & Swarm Manager

Because the primary launch file focuses on the physical drone simulation, you need to launch the **GCS Node** and **Swarm Manager** separately to receive the relayed reports and handle the global pheromone decay.

Open a **second terminal** and run:
```bash
cd ~/swarm_drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Run the Swarm Manager (handles pheromone decay and mission scoring)
ros2 run protoplasm_gazebo swarm_manager &

# Run the GCS Node (listens for relayed SURVIVOR_FOUND reports)
ros2 run protoplasm_gazebo gcs_node
```
