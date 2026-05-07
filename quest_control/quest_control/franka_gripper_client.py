from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import GripperCommand
from franka_msgs.action import Grasp


class FrankaGripperClient(object):
    def __init__(self, node: Node, arm_id="fer"):
        self._node = node
        self._client = ActionClient(
            self._node, GripperCommand, "/fer_gripper/gripper_action"
        )
        self._action_client = ActionClient(self._node, Grasp, "/fer_gripper/grasp")
        self._busy = False  # True while a command is executing

    def send_goal(self, position: float, max_effort: float) -> bool:
        """Sends a goal to the GripperCommand action server. Returns False if busy."""
        if self._busy:
            return False
        self._busy = True
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort
        future = self._client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)
        return True

    def grasp(self, width: float = 0.0, speed: float = 0.04, force: int = 10.0) -> bool:
        """Sends a grasp goal. Returns False if busy."""
        if self._busy:
            return False
        self._busy = True
        goal_msg = Grasp.Goal()
        goal_msg.width = width
        goal_msg.speed = speed
        goal_msg.force = force
        goal_msg.epsilon.inner = 0.08
        goal_msg.epsilon.outer = 0.08

        self._node.get_logger().info("Sending goal to close the gripper...")
        future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.fake_feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        """Handles the response when the goal is accepted/rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().info("Goal rejected.")
            self._busy = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handles the final result of the action."""
        self._busy = False

    def feedback_callback(self, feedback_msg):
        """Handles feedback from the action server."""
        self._node.get_logger().info(
            f"Feedback: Position = {feedback_msg.feedback.position}"
        )

    def fake_feedback_callback(self, feedback_msg):
        """Handles feedback from the action server."""
        self._node.get_logger().info("Feedback: Position = ")