"""
High-Performance Linear MPC for Path Tracking.
Uses CVXPY with OSQP solver for real-time performance.
"""

import numpy as np
import cvxpy as cp

class SteeringMPC:
    def __init__(
        self,
        Np: int = 10,           # 예측 구간 (Prediction Horizon)
        dt: float = 0.05,       # 제어 주기 (0.05s = 20Hz)
        wheelbase: float = 1.023, 
        max_steer: float = 0.349, # +/- 20도
        max_steer_rate: float = 0.785 # +/- 45도/초
    ):
        self.Np = Np
        self.dt = dt
        self.L = wheelbase
        self.max_steer = max_steer
        self.max_steer_rate = max_steer_rate

        # 가중치 행렬 (Tuning needed)
        # Q: 상태 오차 페널티 [횡방향오차, 헤딩오차, 조향각]
        self.Q = np.diag([20.0, 5.0, 0.1]) 
        # R: 입력 변화량 페널티 [조향속도]
        self.R = np.diag([1.0])

        # Warm Start를 위한 이전 솔루션 저장소
        self.prev_u = np.zeros((1, self.Np))

    def solve(self, state, ref_curvature, target_vel):
        """
        MPC 최적화 문제를 풉니다.
        
        Args:
            state: [e_y (횡방향오차), e_psi (헤딩오차), delta (현재조향각)]
            ref_curvature: 미래 경로의 곡률 배열 [Np]
            target_vel: 현재 구간 목표 속도 (Scalar)
        
        Returns:
            opt_steer: 다음 스텝 조향각 (rad)
        """
        # 1. 속도가 너무 느리면 0.1로 고정 (나눗셈 방지)
        v = max(target_vel, 0.1)

        # 2. 선형화된 모델 행렬 구성 (Linearized Kinematic Bicycle Model)
        # State: x = [e_y, e_psi, delta]
        # Input: u = [delta_rate] (조향각 변화율)
        
        # x_{k+1} = A * x_k + B * u_k + D_k (외란/피드포워드)
        A = np.array([
            [1.0, v * self.dt, 0.0],
            [0.0, 1.0, v * self.dt / self.L],
            [0.0, 0.0, 1.0]
        ])
        
        B = np.array([
            [0.0],
            [0.0],
            [self.dt]
        ])

        # 3. CVXPY 변수 선언
        x = cp.Variable((3, self.Np + 1)) # 예측 상태
        u = cp.Variable((1, self.Np))     # 제어 입력 (조향 속도)

        cost = 0
        constraints = []
        
        # 초기 상태 제약
        constraints.append(x[:, 0] == state)

        # 4. 예측 구간 루프 (Horizon Loop)
        for k in range(self.Np):
            # (A) 비용 함수 추가
            # 목표는 오차 0, 조향각은 0이 아니라 '곡률에 맞는 각도'가 되는 것이 이상적임
            # 하지만 간소화를 위해 오차 0을 목표로 하고, 곡률은 모델의 외란(Disturbance)으로 처리
            cost += cp.quad_form(x[:, k], self.Q) + cp.quad_form(u[:, k], self.R)

            # (B) 동역학 제약 조건 (피드포워드 포함)
            # e_psi_next = e_psi + (v/L * delta) * dt - (v * kappa) * dt
            # 여기서 -v*kappa*dt 항이 도로 곡률에 의한 헤딩 변화량(Disturbance)
            kappa_k = ref_curvature[k] if k < len(ref_curvature) else 0.0
            disturbance = np.array([0.0, -v * kappa_k * self.dt, 0.0])

            constraints.append(x[:, k+1] == A @ x[:, k] + B @ u[:, k] + disturbance)

            # (C) 물리적 제약 조건
            constraints.append(cp.abs(u[:, k]) <= self.max_steer_rate) # 조향 속도 제한
            constraints.append(cp.abs(x[2, k+1]) <= self.max_steer)    # 조향각 제한

        # 5. 문제 풀기 (OSQP Solver)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # warm_start=True: 이전 계산 결과를 초기값으로 사용하여 속도 비약적 향상
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"[MPC Error] {e}")
            return float(state[2]) # 실패 시 현재 조향각 유지

        # 6. 결과 반환
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # 최적의 조향 속도(u)를 현재 조향각에 더해서 반환
            steer_rate = u.value[0, 0]
            next_steer = state[2] + steer_rate * self.dt
            
            # 다음 웜스타트를 위해 해 저장
            self.prev_u = u.value
            return float(next_steer)
        else:
            print(f"[MPC Fail] Status: {prob.status}")
            return float(state[2])