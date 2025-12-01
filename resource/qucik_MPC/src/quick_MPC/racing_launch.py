from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("racingproject"))
    
    # 기본 경로 파일 지정 (QP로 만든 최적 경로가 이 이름으로 저장되어 있어야 함)
    default_csv = pkg_share / "data" / "waypoints.csv"

    path_csv = LaunchConfiguration("path_csv")

    declare_path_csv = DeclareLaunchArgument(
        "path_csv",
        default_value=str(default_csv),
        description="Path to waypoint CSV file used for the racing line.",
    )

    node_params = {
        "path_csv": path_csv,  # 기준 경로 CSV 파일 경로
        "sample_ds": 0.5,      # 경로 리샘플 간격 [m] (오프라인 최적화 간격과 일치 권장)
        
        # [핵심 수정 1] 오프라인 최적화 경로를 100% 신뢰하므로 자체 변형 끄기
        "max_offset": 0.0,     # 기존 3.0 -> 0.0 (오프셋 끄기)
        "offset_gain": 0.0,    # 기존 1.0 -> 0.0 (오프셋 끄기)
        
        "offset_power": 1.0,   # (오프셋이 꺼져서 영향 없음)
        "curvature_smooth_window": 11,
        "lookahead_points": 30,  # 제어 시 앞쪽으로 볼 포인트 개수
        
        # 속도 PID 게인 (필요시 튜닝)
        "speed_kp": 0.5,
        "speed_ki": 0.1,
        "speed_kd": 0.01,
        
        # [핵심 수정 2] 차량 한계 속도 반영 (56km/h = 약 15.5m/s)
        # 만약 racing_node.py가 CSV의 velocity를 쓰도록 수정되었다면 이 값은 무시되지만,
        # 안전을 위해 맞춰주는 것이 좋습니다.
        "v_high": 15.5,        # 직선 구간 목표 속도 (기존 15.0 -> 15.5)
        "v_low": 8.0,          # 코너 구간 최소 보장 속도
        
        "kappa_th": 0.05,      # 코너 판단용 곡률 임계값
        
        # MPC 파라미터
        "mpc_Np": 10,          # 예측 지평선 (부드러운 주행을 원하면 15~20으로 늘려보세요)
        "mpc_Nc": 5,           # 제어 지평선
        "control_dt": 0.05,    # 제어 주기 [s] (20Hz)
        
        # 차량 제원 (공지사항 기준)
        "max_steer_deg": 20.0,       # 최대 조향각 [deg] (+/- 20도)
        "max_steer_rate_deg": 45.0,  # 최대 조향각 속도 [deg/s]
        "wheelbase": 1.023,          # 휠베이스 길이 [m]
    }

    racing_node = Node(
        package="racingproject",
        executable="racing_node",
        name="racing_node",
        output="screen",
        parameters=[node_params],
    )

    return LaunchDescription([declare_path_csv, racing_node])