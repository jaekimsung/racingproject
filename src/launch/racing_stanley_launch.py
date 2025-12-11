from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_csv = Path(__file__).resolve().parent.parent / "racingproject" / "data" / "final_margin1m.csv"

    path_csv = LaunchConfiguration("path_csv")

    declare_path_csv = DeclareLaunchArgument(
        "path_csv",
        default_value=str(default_csv),
        description="Path to waypoint CSV file used for the racing line.",
    )

    node_params = {
        "path_csv": path_csv,  # 기준 경로 CSV 파일 경로
        "lookahead_distance": 7.0,
        "speed_kp": 0.8,  # 속도 PID Kp
        "speed_ki": 0.0,  # 속도 PID Ki
        "speed_kd": 0.01,  # 속도 PID Kd
        "v_high": 17.0,  # 직선 구간 목표 속도 [m/s] 최대속도 56km/h = 15.5m/s
        "v_low": 13.0,  # 코너 구간 목표 속도 [m/s]
        "kappa_th": 0.08,  # 코너 판단용 곡률 임계값, 이 곡률 넘어가면 감속

        # 이 구간 안에 차가 있으면 무조건 v_low
        "slow_x_min": -146.0,
        "slow_x_max": -96.0,
        "slow_y_min": 7.0,
        "slow_y_max": 56.0,
        
        "control_dt": 0.05,  # 제어 주기/샘플 타임 [s]
        "max_steer_deg": 20.0,  # 최대 조향각 [deg]
        "max_steer_rate_deg": 60.0,  # 최대 조향각 속도 [deg/s]
        "wheelbase": 1.023,  # 휠베이스 길이 [m]
        
        "stanley_k": 1.65,  # Stanley 횡방향 오차 게인
        "stanley_softening": 0.1,  # 저속 안정화를 위한 소프트닝 항 [m/s]
        "stanley_heading_gain": 1.0,  # 헤딩 오차 가중치
    }

    racing_node = Node(
        package="racingproject",
        executable="racing_node_stanley",
        name="racing_node_stanley",
        output="screen",
        parameters=[node_params],
    )

    return LaunchDescription([declare_path_csv, racing_node])
