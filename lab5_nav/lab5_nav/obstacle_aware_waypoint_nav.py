#!/usr/bin/env python3
import ast
import math
from typing import List, Optional, Sequence, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node


def _parse_waypoints(raw: Sequence) -> List[Tuple[float, float, Optional[float]]]:
    """Return waypoints as (x, y, yaw?) tuples; yaw may be None."""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Failed to parse string waypoints: {exc}") from exc

    if isinstance(raw, (list, tuple)):
        if not raw:
            return []
        if isinstance(raw[0], (list, tuple)):
            seqs = raw
        else:
            if len(raw) % 2 != 0:
                raise ValueError("Flat waypoint list must contain an even number of entries")
            seqs = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    else:
        raise ValueError(f"Unsupported waypoints parameter type: {type(raw)}")

    parsed: List[Tuple[float, float, Optional[float]]] = []
    for idx, entry in enumerate(seqs):
        if not isinstance(entry, (list, tuple)):
            raise ValueError(f"Waypoint #{idx} must be a sequence, got {type(entry)}")
        if len(entry) < 2:
            raise ValueError(f"Waypoint #{idx} must contain at least x and y")
        if len(entry) >= 3:
            parsed.append((float(entry[0]), float(entry[1]), float(entry[2])))
        else:
            parsed.append((float(entry[0]), float(entry[1]), None))
    return parsed


def _parse_initial_pose(raw) -> Optional[Tuple[float, float, float]]:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Failed to parse initial_pose string: {exc}") from exc

    if raw in (None, [], ()):
        return None

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        yaw = float(raw[2]) if len(raw) >= 3 else 0.0
        return float(raw[0]), float(raw[1]), yaw

    raise ValueError("initial_pose must be [], [x, y] or [x, y, yaw]")


def _yaw_to_quaternion(z_yaw: float) -> Tuple[float, float, float, float]:
    half = z_yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _duration_from_seconds(seconds: float) -> DurationMsg:
    msg = DurationMsg()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - msg.sec) * 1e9)
    return msg


class ObstacleAwareWaypointNav(Node):
    """Send a set of map-frame waypoints to Nav2 once the stack is ready."""

    def __init__(self) -> None:
        super().__init__("obstacle_aware_waypoint_nav")

        self.declare_parameter("use_sim_time", False)
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("waypoints", [])
        self.declare_parameter("initial_pose", [])
        self.declare_parameter("time_allowance", 600.0)
        self.declare_parameter("behavior_tree", "")
        self.declare_parameter("goal_speed", 0.0)

        self._goal_frame = str(self.get_parameter("goal_frame").value)
        self._time_allowance = float(self.get_parameter("time_allowance").value)
        self._behavior_tree = str(self.get_parameter("behavior_tree").value).strip()
        self._goal_speed = float(self.get_parameter("goal_speed").value)

        try:
            raw_wps = self.get_parameter("waypoints").value
            parsed_wps = _parse_waypoints(raw_wps)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            parsed_wps = []

        if not parsed_wps:
            self.get_logger().warn("No waypoints provided; navigator will idle.")

        self._targets = self._prepare_targets(parsed_wps)

        # Optional initial pose to seed AMCL so Nav2 can localize immediately.
        self._initial_pose_msg: Optional[PoseWithCovarianceStamped] = None
        try:
            pose_tuple = _parse_initial_pose(self.get_parameter("initial_pose").value)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            pose_tuple = None

        if pose_tuple is not None:
            self._initial_pose_msg = self._build_initial_pose_msg(pose_tuple)
            self._initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
            self._initial_pose_timer = self.create_timer(1.0, self._publish_initial_pose_once)
            self._initial_pose_repeats = 0
        else:
            self._initial_pose_pub = None
            self._initial_pose_timer = None
            self._initial_pose_repeats = 0

        self._nav_through_client = ActionClient(self, NavigateThroughPoses, "navigate_through_poses")
        self._nav_to_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self._goal_handle = None
        self._result_future = None
        self._goal_sent = False
        self._last_feedback_time = self.get_clock().now()
        self._last_wait_log = self.get_clock().now() - Duration(seconds=10.0)

        self._starter_timer = self.create_timer(0.5, self._try_send_goal)

        if self._targets:
            self.get_logger().info(f"Prepared {len(self._targets)} waypoint(s) in frame '{self._goal_frame}'.")

    def _prepare_targets(self, waypoints: List[Tuple[float, float, Optional[float]]]) -> List[Tuple[float, float, float]]:
        targets: List[Tuple[float, float, float]] = []
        fallback_yaw = 0.0
        for idx, entry in enumerate(waypoints):
            x, y, yaw = entry
            if yaw is None and idx + 1 < len(waypoints):
                nxt = waypoints[idx + 1]
                yaw = math.atan2(nxt[1] - y, nxt[0] - x)
            if yaw is None:
                yaw = fallback_yaw
            fallback_yaw = yaw
            targets.append((x, y, yaw))
        return targets

    def _build_initial_pose_msg(self, pose_tuple: Tuple[float, float, float]) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self._goal_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        x, y, yaw = pose_tuple
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        _, _, qz, qw = _yaw_to_quaternion(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = math.radians(10.0) ** 2
        msg.pose.covariance = cov
        return msg

    def _publish_initial_pose_once(self) -> None:
        if not self._initial_pose_pub or not self._initial_pose_msg:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self._initial_pose_msg.header.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = self._initial_pose_msg.pose
        self._initial_pose_pub.publish(msg)
        self._initial_pose_repeats += 1
        if self._initial_pose_repeats >= 5 and self._initial_pose_timer:
            self._initial_pose_timer.cancel()

    def _build_pose_list(self) -> List[PoseStamped]:
        poses: List[PoseStamped] = []
        now = self.get_clock().now().to_msg()
        for x, y, yaw in self._targets:
            pose = PoseStamped()
            pose.header.frame_id = self._goal_frame
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
            _, _, qz, qw = _yaw_to_quaternion(yaw)
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            poses.append(pose)
        return poses

    def _try_send_goal(self) -> None:
        if self._goal_sent or not self._targets:
            return

        count = len(self._targets)

        if count > 1:
            if not self._nav_through_client.wait_for_server(timeout_sec=0.0):
                self._log_waiting_message("navigate_through_poses")
                return
            goal_msg = NavigateThroughPoses.Goal()
            goal_msg.poses = self._build_pose_list()
            goal_msg.time_allowance = _duration_from_seconds(self._time_allowance)
            if self._behavior_tree:
                goal_msg.behavior_tree = self._behavior_tree
            if self._goal_speed > 0.0:
                goal_msg.speed = float(self._goal_speed)
            self.get_logger().info(f"Sending {count} waypoints to Nav2 (NavigateThroughPoses).")
            future = self._nav_through_client.send_goal_async(goal_msg, feedback_callback=self._feedback_cb)
        else:
            if not self._nav_to_client.wait_for_server(timeout_sec=0.0):
                self._log_waiting_message("navigate_to_pose")
                return
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = self._build_pose_list()[0]
            goal_msg.time_allowance = _duration_from_seconds(self._time_allowance)
            if self._behavior_tree:
                goal_msg.behavior_tree = self._behavior_tree
            if self._goal_speed > 0.0:
                goal_msg.speed = float(self._goal_speed)
            self.get_logger().info("Sending single waypoint to Nav2 (NavigateToPose).")
            future = self._nav_to_client.send_goal_async(goal_msg, feedback_callback=self._feedback_cb)

        self._goal_sent = True
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected the navigation request.")
            self._goal_handle = None
            return
        self.get_logger().info("Nav2 accepted the navigation goal.")
        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback) -> None:
        now = self.get_clock().now()
        if now - self._last_feedback_time < Duration(seconds=2.0):
            return
        self._last_feedback_time = now
        try:
            dist = feedback.feedback.distance_remaining
            eta = feedback.feedback.estimated_time_remaining
            eta_sec = eta.sec + eta.nanosec / 1e9
            self.get_logger().info(f"Distance remaining: {dist:.2f} m, ETA: {eta_sec:.1f} s")
        except AttributeError:
            self.get_logger().info("Navigation feedback received.")

    def _result_cb(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to get Nav2 result: {exc}")
            return

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation task completed successfully.")
        else:
            self.get_logger().warn(f"Navigation finished with status {status}.")

    def _log_waiting_message(self, action_name: str) -> None:
        now = self.get_clock().now()
        if now - self._last_wait_log >= Duration(seconds=5.0):
            self.get_logger().info(f"Waiting for '{action_name}' action server...")
            self._last_wait_log = now

    def destroy_node(self):
        if self._starter_timer is not None:
            self._starter_timer.cancel()
        if self._initial_pose_timer is not None:
            self._initial_pose_timer.cancel()
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ObstacleAwareWaypointNav()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
