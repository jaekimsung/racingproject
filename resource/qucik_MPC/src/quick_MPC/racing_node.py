import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import os

# 모듈 임포트 (파일 이름에 맞게 수정하세요)
from .path_manager import PathManager
from .steering_mpc import SteeringMPC
from .speed_pid import SpeedPID

class RacingNode(Node):
    def __init__(self):
        super().__init__('racing_node')
        
        # 1. 파라미터 설정 (Launch 파일에서 넘어오는 값들)
        self.declare_parameters("", [
            ("path_csv", "src/racingproject/data/waypoints.csv"),
            ("mpc_Np", 10),
            ("wheelbase", 1.023),
            ("speed_kp", 0.5), ("speed_ki", 0.1), ("speed_kd", 0.05)
        ])
        
        path_csv = self.get_parameter("path_csv").value
        wheelbase = self.get_parameter("wheelbase").value
        
        # 2. 모듈 초기화
        self.get_logger().info("Initializing Modules...")
        
        # (A) Path Manager: 최적 경로 로드
        self.path_manager = PathManager(path_csv)
        
        # (B) MPC: 조향 제어
        self.mpc = SteeringMPC(
            Np=self.get_parameter("mpc_Np").value,
            dt=0.05,
            wheelbase=wheelbase
        )
        
        # (C) PID: 속도 제어
        self.pid = SpeedPID(
            kp=self.get_parameter("speed_kp").value,
            ki=self.get_parameter("speed_ki").value,
            kd=self.get_parameter("speed_kd").value
        )

        # 3. ROS2 통신 설정
        self.sub_state = self.create_subscription(
            Float32MultiArray, 
            '/mobile_system_control/ego_vehicle', 
            self.state_callback, 
            10
        )
        
        self.pub_control = self.create_publisher(
            Vector3Stamped, 
            '/mobile_system_control/control_msg', 
            10
        )
        
        # 제어 주기 0.05s (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.current_state = None # [x, y, theta, v, delta]

    def state_callback(self, msg):
        """센서 데이터 수신"""
        if len(msg.data) >= 5:
            self.current_state = msg.data # x, y, theta, v, steer

    def control_loop(self):
        """메인 제어 루프"""
        if self.current_state is None:
            return

        # 1. 상태 파싱
        x, y, theta, v, current_steer = self.current_state
        
        # 2. 경로 매칭 (가장 가까운 점 찾기)
        idx = self.path_manager.find_closest_index(x, y)
        
        # 3. Local Path 가져오기 (MPC용)
        # mpc_Np만큼의 미래 경로 데이터를 가져옴
        path_xy, path_vel, path_kappa, path_heading = \
            self.path_manager.get_local_segment(idx, self.mpc.Np)
            
        # 4. 에러 계산 (Global -> Vehicle Frame)
        # 가장 가까운 경로점(target) 기준 오차 계산
        target_x, target_y = path_xy[0]
        target_head = path_heading[0]
        
        dx = target_x - x
        dy = target_y - y
        
        # 횡방향 오차 (Lateral Error): 차량 기준 y축 거리
        e_y = -math.sin(theta) * dx + math.cos(theta) * dy
        
        # 헤딩 오차 (Heading Error): 각도 차이 정규화
        e_psi = target_head - theta
        e_psi = (e_psi + math.pi) % (2 * math.pi) - math.pi # Wrap Angle
        
        # 5. 목표 속도 설정 (CSV에서 가져온 최적 속도)
        # 현재 위치의 목표 속도 사용
        target_v = path_vel[0]
        
        # 6. 제어 입력 계산
        # (A) MPC Steering
        mpc_state = [e_y, e_psi, current_steer]
        steer_cmd_rad = self.mpc.solve(mpc_state, path_kappa, target_v)
        
        # (B) PID Throttle/Brake
        throttle, brake = self.pid.step(target_v, v, 0.05)
        
        # 7. 명령 발행 (Publish)
        self.publish_command(throttle, brake, steer_cmd_rad)

    def publish_command(self, throttle, brake, steer_rad):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "Team_Name" # [중요] 팀 이름 변경 필요
        
        # 조향각 정규화 (-20도~20도 -> -1.0~1.0)
        max_steer_rad = 0.349
        steer_norm = np.clip(steer_rad / max_steer_rad, -1.0, 1.0)
        
        msg.vector.x = float(throttle)
        msg.vector.y = float(steer_norm)
        msg.vector.z = float(brake)
        
        self.pub_control.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RacingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()