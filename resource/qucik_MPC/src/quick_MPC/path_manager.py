"""Path management: Loads optimized trajectory CSV and provides lookahead segments."""

import numpy as np
import pandas as pd
import os

class PathManager:
    """
    최적화된 경로(CSV)를 로드하고, 현재 차량 위치에 맞는 경로 세그먼트를 제공합니다.
    """

    def __init__(self, csv_path: str):
        self.load_path(csv_path)

    def load_path(self, csv_path: str):
        """CSV 파일을 로드하여 메모리에 저장합니다."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Path file not found: {csv_path}")
        
        # Pandas로 빠르게 로드 (x, y, velocity, heading, curvature)
        df = pd.read_csv(csv_path)
        
        # Numpy 배열로 변환 (연산 속도 최적화)
        self.path_xy = df[['x', 'y']].values
        self.velocity = df['velocity'].values
        self.curvature = df['curvature'].values
        self.heading = df['heading'].values
        
        self.num_points = len(self.path_xy)
        print(f"[PathManager] Loaded {self.num_points} waypoints from {csv_path}")

    def find_closest_index(self, x: float, y: float) -> int:
        """
        현재 차량 위치(x, y)에서 가장 가까운 경로점의 인덱스를 찾습니다.
        (전체 탐색 대신 벡터 연산 사용)
        """
        # 유클리드 거리 제곱 계산 (sqrt 생략하여 연산 속도 향상)
        deltas = self.path_xy - np.array([x, y])
        dist_sq = np.einsum('ij,ij->i', deltas, deltas)
        return int(np.argmin(dist_sq))

    def get_local_segment(self, idx: int, horizon: int):
        """
        현재 인덱스부터 미래 N스텝까지의 경로 정보를 가져옵니다.
        트랙이 순환(Loop)한다고 가정하고 인덱스를 처리합니다.
        """
        indices = np.arange(idx, idx + horizon) % self.num_points
        
        return (
            self.path_xy[indices],      # [N, 2] 좌표
            self.velocity[indices],     # [N] 목표 속도
            self.curvature[indices],    # [N] 목표 곡률
            self.heading[indices]       # [N] 목표 헤딩
        )