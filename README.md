## 🟦 Background

This project introduces a host-side ROS2 architecture designed for the **Unibots 2026 competition**.  
The goal is to create a **robust, modular, and scalable** system that supports vision processing, global localization, perception, planning, and strategy execution in both simulation and real-robot environments.

To achieve this, we adopt a **ROS2 modular design**, in which each subsystem is implemented as an independent node with a clear responsibility. All components communicate through ROS topics and the TF transform system, enabling loose coupling, clean data flow, and easy extensibility.

Under this architecture:

- **In simulation (Webots)**  
  We use `apriltag_ros` to detect AprilTags and a custom `pv_sim_bridge` node to compute the robot’s global pose using tag observations combined with the arena’s known field layout.

- **On the real robot**  
  We use **PhotonVision**, a proven, high-performance vision system that performs AprilTag detection and pose estimation directly on the robot, providing reliable localization output.

Despite the differences between simulation and real hardware, both environments unify their localization results under the same set of ROS topics (`/pv/*`).  
This ensures that higher-level modules—navigation, planning, and strategy—work identically across both environments without any code changes.

This architecture reduces development complexity, accelerates testing, and provides a stable foundation for future expansions such as multi-sensor fusion, game-element perception, and autonomous decision-making.



## 🧩 System Architecture Overview 

This project uses a clean and modular ROS2-based architecture to power our Unibots robot.  
The system is designed so that each part of the robot (camera, localization, ball detection, planning, control, etc.) works like a small member in a team—with each module doing one clear job and passing information to the next.

The overall goal is to make the robot understand:
1) what it sees,  
2) where it is,  
3) where the game objects are, and  
4) how it should move to achieve the task.

---

## 📷 1. Camera — The Robot’s “Eyes”

**What it does:**  
The camera captures images from the field.

**Who uses its output:**  
- The Localization node (to find robot position using AprilTags)  
- The Ball Detection node (to find game objects)

**Simulation:**  
Webots provides the camera topics:
- `/epuck/camera/image_color`
- `/epuck/camera/camera_info`

**Real robot:**  
A USB or onboard camera provides equivalent topics.

---

## 📡 2. Localization — “Where am I?”

Localization converts camera observations into a **global robot position** on the field.

### In Simulation  
We use two ROS nodes:
- `apriltag_ros` to detect AprilTags in the camera image  
- `pv_sim_bridge` (sim_bridge) to convert tag detections into `map → base_link` robot pose using:
  - known arena tag positions  
  - camera calibration  
  - robot-to-camera extrinsics  

### On the Real Robot  
We use **PhotonVision (PV)**:
- PV detects AprilTags  
- PV computes robot pose directly using its built-in field layout and calibration  
- The pose is formatted into the same `/pv/*` topics as in simulation

### Unified output (same in sim & real):
- `/pv/estimated_robot_pose`
- `/pv/robot_yaw_deg`
- `/pv/has_target`

This makes the higher-level code identical in both environments.

---

## 🔍 3. Ball Detection — “Where are the balls?” ⭕️ Todo

This node analyzes camera images to find balls or other game pieces.

**Input:** Camera images  
**Output:** Detected object positions (relative to the robot or field); Trying to use Yolo
**Used by:** Path planning & strategy modules

It acts like the robot’s **scout**, reporting target positions.

---

## 🧭 4. Path Planning & Control — “How do I get there?” ⭕️ Todo

This module receives:
- The robot’s current pose (from Localization)  
- The ball positions (from Ball Detection)  
- High-level goals (from Strategy)

It computes:
- How the robot should move  
- What path to take  
- How to control the wheels smoothly  

**Output:** Motion commands (e.g., `/cmd_vel`)

It acts as the robot’s **driver**.

---

## 🧠 5. Strategy — “What should I do next?” ⭕️ Todo

The Strategy layer makes high-level decisions such as:
- Which ball to collect  
- How to score  
- When to reposition  
- What to do if the robot loses vision  

It sends goals to the Path Planner but does not directly control movement.
It sends command to those function part like intake to control them.

---


## Why We Use PhotonVision on the Real Robot  
(Official Website: **https://photonvision.org**)

For the real robot, we use **PhotonVision (PV)** as our primary AprilTag detection and pose estimation system. In real competition environments like Unibots, PV offers several key advantages that make it both practical and reliable:

- **High reliability and real-time performance**  
  PhotonVision is optimized for embedded hardware and delivers fast, stable, low-latency AprilTag detection without requiring us to write our own vision algorithms.

- **Built-in pose estimation**  
  PV automatically combines camera calibration, robot-to-camera extrinsics, and field tag layout to compute the robot’s global pose. This eliminates the need for custom localization code on the robot.

- **Excellent Web UI for tuning and debugging**  
  The PhotonVision UI lets us view the camera feed, adjust exposure, test tag detection, and verify calibration in real time—without recompiling or redeploying any robot code.

- **Easy on-field adjustments**  
  Lighting, reflections, and camera alignment can change during competition. PV allows us to tune parameters or recalibrate instantly on the field, ensuring consistent detection performance throughout the event.

- **Battle-tested community reliability**  
  PV is widely used in FRC and other robotics competitions, with strong community support, documentation, and proven stability under real-world conditions.

Thanks to these advantages, PhotonVision provides a robust, efficient, and highly tunable vision system for our real robot, significantly reducing development complexity while improving on-site adaptability.

