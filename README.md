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
