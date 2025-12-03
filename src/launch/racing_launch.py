from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_csv = Path(__file__).resolve().parent.parent / "racingproject" / "data" / "optimal.csv"

    path_csv = LaunchConfiguration("path_csv")

    declare_path_csv = DeclareLaunchArgument(
        "path_csv",
        default_value=str(default_csv),
        description="Path to waypoint CSV file used for the racing line.",
    )

    node_params = {
        "path_csv": path_csv,  # 기준 경로 CSV 파일 경로
        "lookahead_points": 10,  # 제어 시 앞쪽으로 최소한 볼 포인트 개수 (포인트 간 간격 0.5m)
        "braking_distance": 5.0,  # 감속 판단용 앞보기 거리 [m]
        
        "speed_kp": 0.5,  # 속도 PID Kp
        "speed_ki": 0.1,  # 속도 PID Ki
        "speed_kd": 0.01,  # 속도 PID Kd
        
        "v_high": 6.0,  # 직선 구간 목표 속도 [m/s] 최대속도 56km/h = 15.5m/s
        "v_low": 4.0,  # 코너 구간 목표 속도 [m/s]
        "kappa_th": 0.08,  # 코너 판단용 곡률 임계값, 이 곡률 넘어가면 감속 (높이면 감속 구간 줄어듦)

        # 이 구간 안에 차가 있으면 무조건 v_low
        "slow_x_min": 75.0,
        "slow_x_max": 90.0,
        "slow_y_min": 12.0,
        "slow_y_max": 45.0,
        
        "mpc_Np": 30,  # MPC 예측 지평선 길이
        "mpc_Nc": 5,  # MPC 제어 지평선 길이
        "control_dt": 0.05,  # 제어 주기/샘플 타임 [s]
        
        "max_steer_deg": 20.0,  # 최대 조향각 [deg]
        "wheelbase": 1.023,  # 휠베이스 길이 [m]
    }

    racing_node = Node(
        package="racingproject",
        executable="racing_node",
        name="racing_node",
        output="screen",
        parameters=[node_params],
    )

    return LaunchDescription([declare_path_csv, racing_node])
