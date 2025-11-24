from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("racingproject"))
    default_csv = pkg_share / "data" / "waypoints.csv"

    path_csv = LaunchConfiguration("path_csv")

    declare_path_csv = DeclareLaunchArgument(
        "path_csv",
        default_value=str(default_csv),
        description="Path to waypoint CSV file used for the racing line.",
    )

    node_params = {
        "path_csv": path_csv,  # 기준 경로 CSV 파일 경로
        "sample_ds": 0.5,  # 경로 리샘플 간격 [m]
        "max_offset": 3.0,  # 곡률 기반 레이싱 라인 최대 횡방향 오프셋 [m]
        "offset_gain": 1.0,  # 곡률→오프셋 스케일
        "offset_power": 1.0,  # 곡률 정규화 후 거듭제곱 (비선형성)
        "curvature_smooth_window": 11,  # 곡률 이동 평균 창 크기
        "lookahead_points": 30,  # 제어 시 앞쪽으로 볼 포인트 개수
        
        "speed_kp": 0.5,  # 속도 PID Kp
        "speed_ki": 0.1,  # 속도 PID Ki
        "speed_kd": 0.01,  # 속도 PID Kd
        
        "v_high": 15.0,  # 직선 구간 목표 속도 [m/s] 최대속도 56km/h = 15.5m/s
        "v_low": 8.0,  # 코너 구간 목표 속도 [m/s]
        
        "kappa_th": 0.05,  # 코너 판단용 곡률 임계값
        "mpc_Np": 10,  # MPC 예측 지평선 길이
        "mpc_Nc": 5,  # MPC 제어 지평선 길이
        "control_dt": 0.05,  # 제어 주기/샘플 타임 [s]
        
        "max_steer_deg": 20.0,  # 최대 조향각 [deg]
        "max_steer_rate_deg": 45.0,  # 최대 조향각 속도 [deg/s]
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
