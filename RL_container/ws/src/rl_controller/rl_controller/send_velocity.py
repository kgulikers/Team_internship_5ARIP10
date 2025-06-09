import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
from origin_msgs.msg import ControlMode
from origin_msgs.srv import SetControlMode, ReturnControlMode

import torch
import torch.nn.modules.container
import torch.nn as nn
import numpy as np

NR_RETRIES = 3
NUM_LIDAR = 180
MAX_LINEAR_SPEED = 3.0
MAX_ANGULAR_SPEED = 5.0

class ClientService():
    def __init__(self, node, srv_type, srv_name):
        self.node = node
        self.cli = self.node.create_client(srv_type=srv_type, srv_name=srv_name)

    def make_request(self, request_msg):
        counter = 0
        while not self.cli.wait_for_service(timeout_sec=0.5) and counter < NR_RETRIES:
            self.node.get_logger().info('service not available, waiting again...')
            counter += 1
        if counter == NR_RETRIES:
            self.node.get_logger().info('service not available, abort...')
            return None
        else:
            future = self.cli.call_async(request_msg)
            rclpy.spin_until_future_complete(self.node, future)
            self.node.get_logger().info('service call done')
            return future.result() if hasattr(future, 'result') else None

class MLP26_64_64_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)

class SendVelocity(Node):
    def __init__(self):
        super().__init__('send_velocity_node')

        self.velocity_publisher = self.create_publisher(Twist, '/robot/cmd_vel_user', 10)
        self.lidar_subscription = self.create_subscription(PointCloud2, '/robot/lidar/points', self.lidar_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/robot/odom', self.odom_callback, 10)

        self.request_control = ClientService(self, SetControlMode, '/robot/cmd_vel_controller/set_control_mode')
        self.release_control = ClientService(self, ReturnControlMode, '/robot/cmd_vel_controller/reset_control_mode')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
        self.model = MLP26_64_64_2().to(self.device)

        sd_path = '/home/user/ws/src/rl_controller/rl_controller/model_180_state_dict.pt'
        loaded_dict = torch.load(sd_path, map_location=self.device)
        fixed_dict = {f"net.{k}": v for k, v in loaded_dict.items()}
        self.model.load_state_dict(fixed_dict)
        self.model.eval()

        self.lin_vel = np.array([0.0, 0.0, 0.0])
        self.ang_vel = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0])
        self.yaw = 0.0
        self.last_action = [0.0, 0.0]
        self.lidar_pooled = [100.0] * 16

        self.obs = torch.tensor([[0.0] * 26], dtype=torch.float32).to(self.device)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def ang_z(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.yaw = self.ang_z(msg.pose.pose.orientation)
        self.lin_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.ang_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    def pool_region(self, dists, start_deg, end_deg, num_bins):
        N = len(dists)
        start_idx = int((start_deg + 180) / 360 * N)
        end_idx = int((end_deg + 180) / 360 * N)
        region = dists[start_idx:end_idx]
        step = max(1, len(region) // num_bins)
        pooled = [min(region[i * step:(i + 1) * step]) for i in range(num_bins)]
        return pooled

    def lidar_callback(self, msg):
        raw_ranges = np.full(NUM_LIDAR, 100.0)
        angle_bins = np.linspace(-np.pi/2, np.pi/2, NUM_LIDAR)

        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if abs(z) > 0.1:
                continue
            r = np.hypot(x, y)
            a = np.arctan2(y, x)
            if a < -2.0 * np.pi / 3 or a > 2.0 * np.pi / 3:
                continue
            bin_idx = np.searchsorted(angle_bins, a)
            if 0 <= bin_idx < NUM_LIDAR:
                raw_ranges[bin_idx] = min(raw_ranges[bin_idx], r)


        front = self.pool_region(raw_ranges, -60, 60, 10)
        self.get_logger().info(f"Front LiDAR pooled bins: {front}")
        left = self.pool_region(raw_ranges, 60, 120, 3)
        right = self.pool_region(raw_ranges, -120, -60, 3)

        self.lidar_pooled = front + left + right

    def timer_callback(self):
        goal = np.array([5.0, -5.0])

        delta = goal - self.position
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        x_rel = cos_yaw * delta[0] + sin_yaw * delta[1]
        y_rel = -sin_yaw * delta[0] + cos_yaw * delta[1]
        to_goal = [x_rel, y_rel]

        obs_vector = list(self.lin_vel) + list(self.ang_vel) + self.lidar_pooled + to_goal + self.last_action

        if np.any(np.isnan(obs_vector)) or np.any(np.isinf(obs_vector)):
            self.get_logger().warn(f'Invalid obs_vector! nan/inf found: {obs_vector}')
        else:
            self.get_logger().info(f'obs_vector is fine: {obs_vector}')

        self.obs = torch.tensor([obs_vector], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action = self.model(self.obs)

        linear_vel = float(action[0][0].cpu())
        angular_vel = float(action[0][1].cpu())

        self.get_logger().info(f"Raw model output → linear: {linear_vel:.4f}, angular: {angular_vel:.4f}")

        linear_vel = np.clip(linear_vel, -1.0, 1.0)
        angular_vel = np.clip(angular_vel, -1.0, 1.0)

        self.get_logger().info(f"Clipped output → linear: {linear_vel:.4f}, angular: {angular_vel:.4f}")

        self.last_action = [linear_vel, angular_vel]

        twist = Twist()
        twist.linear.x = MAX_LINEAR_SPEED * linear_vel
        twist.angular.z = MAX_ANGULAR_SPEED * angular_vel

        self.get_logger().info(f"Sending Twist → linear.x: {twist.linear.x:.4f}, angular.z: {twist.angular.z:.4f}")
        self.velocity_publisher.publish(twist)

        self.get_logger().info(f'x_rel: {x_rel:.2f}, y_rel: {y_rel:.2f}')
        self.get_logger().info(f'Publishing velocity: {linear_vel:.2f} m/s forward, {angular_vel * 180 / np.pi:.2f} deg/s turn')


    def destroy_node(self):
        stop_msg = Twist()
        self.velocity_publisher.publish(stop_msg)

        req = ReturnControlMode.Request()
        req.mode_from.mode = ControlMode.USER
        self.release_control.make_request(req)
        self.get_logger().info('Released control of the robot')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SendVelocity()

    # Request control 
    req1 = SetControlMode.Request()
    req1.mode.mode = ControlMode.USER
    node.request_control.make_request(req1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, stopping robot...')
        stop_msg = Twist()
        node.velocity_publisher.publish(stop_msg)

        # Release control
        req2 = ReturnControlMode.Request()
        req2.mode_from.mode = ControlMode.USER
        node.release_control.make_request(req2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()