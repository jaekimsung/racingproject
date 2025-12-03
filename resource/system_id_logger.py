import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3Stamped
import time
import os

class SystemIDLogger(Node):
    def __init__(self):
        super().__init__('system_id_logger')
        
        # 데이터 구독
        self.sub_state = self.create_subscription(
            Float32MultiArray, 
            '/mobile_system_control/ego_vehicle', 
            self.state_callback, 
            10
        )
        
        self.sub_control = self.create_subscription(
            Vector3Stamped,
            '/mobile_system_control/control_msg',
            self.control_callback,
            10
        )

        # 상태 변수 초기화
        self.prev_v = 0.0
        self.prev_time = time.time()
        self.current_accel = 0.0
        
        # 기록용 변수
        self.max_accel_record = 0.0
        self.max_decel_record = 0.0 # 감속도는 음수 가속도
        
        # 현재 입력 상태
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0

        # 화면 갱신 타이머 (0.1초마다)
        self.timer = self.create_timer(0.1, self.print_status)

    def state_callback(self, msg):
        """속도를 받아 가속도 계산"""
        if len(msg.data) < 4: return
        
        current_v = msg.data[3] # velocity
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt > 0.001: # 0으로 나누기 방지
            # 가속도 계산 (a = dv / dt)
            dv = current_v - self.prev_v
            self.current_accel = dv / dt
            
            # 최대 기록 갱신 (노이즈 방지를 위해 속도가 1m/s 이상일 때만)
            if current_v > 1.0:
                if self.current_accel > self.max_accel_record:
                    self.max_accel_record = self.current_accel
                
                # 감속도는 음수 값이므로 최소값을 찾음
                if self.current_accel < self.max_decel_record:
                    self.max_decel_record = self.current_accel

            self.prev_v = current_v
            self.prev_time = current_time

    def control_callback(self, msg):
        """현재 내가 입력한 제어값 확인"""
        self.throttle = msg.vector.x
        self.steer = msg.vector.y
        self.brake = msg.vector.z

    def print_status(self):
        """터미널에 대시보드 출력"""
        # 화면 지우기 (Windows: cls, Linux/Mac: clear)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*50)
        print(f"   🚗 Real-time System Identification 🚗")
        print("="*50)
        print(f" [Input]  Throttle: {self.throttle:.2f} | Brake: {self.brake:.2f}")
        print(f" [State]  Speed:    {self.prev_v * 3.6:.1f} km/h  ({self.prev_v:.2f} m/s)")
        print(f" [Now]    Accel:    {self.current_accel:.3f} m/s^2")
        print("-" * 50)
        print(f" 🏆 MAX ACCEL (가속력): {self.max_accel_record:.3f} m/s^2")
        print(f" 🛑 MAX DECEL (제동력): {abs(self.max_decel_record):.3f} m/s^2")
        print("="*50)
        print(" * Tip: 풀악셀/풀브레이크 후 위 값을 기록하세요.")

def main(args=None):
    rclpy.init(args=args)
    node = SystemIDLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()