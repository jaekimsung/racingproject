import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg # Removed as requested
from scipy.interpolate import CubicSpline

# ==========================================
# 1. 설정 파라미터 (Tuning Parameters)
# ==========================================
TRACK_WIDTH = 8.0           # 도로 전폭 [m]
VEHICLE_WIDTH = 1.2         # 차량 전폭 [m] (약 1.2m 가정)
SAFETY_MARGIN = 0.8         # 도로 끝에서 띄울 여유 거리 [m]
# 실제 사용 가능한 횡방향 오프셋 한계 (+/-)
MAX_OFFSET = (TRACK_WIDTH / 2.0) - (VEHICLE_WIDTH / 2.0) - SAFETY_MARGIN

# 속도 프로파일 생성용 물리 파라미터
MU = 1.2                    # 타이어 마찰 계수 (튜닝 필요, 1.0 ~ 1.5)
G = 9.81                    # 중력 가속도
MAX_ACCEL = 4.0             # 최대 가속도 [m/s^2] (후륜 구동 고려)
MAX_DECEL = 8.0             # 최대 감속도 [m/s^2] (브레이크 성능)
MAX_VELOCITY = 15.5         # 차량의 절대 최대 속도 [m/s] (최대 56 km/h)

def load_and_resample(waypoints, step_size=0.5):
    """CSV를 읽고 등간격(step_size)으로 촘촘하게 재샘플링합니다."""
    try:
        data = pd.read_csv(waypoints)
        points = data[['x', 'y']].values
    except:
        # 헤더가 없는 경우 처리
        data = pd.read_csv(waypoints, header=None)
        points = data.iloc[:, :2].values

    # 중복 점 제거
    diff = np.diff(points, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    valid_idx = np.insert(dist > 0.01, 0, True)
    points = points[valid_idx]

    # 스플라인 보간으로 등간격 생성
    # 루프(Loop) 구조인 경우 마지막 점과 첫 점을 이어줌
    is_loop = np.linalg.norm(points[0] - points[-1]) < 1.0
    if is_loop:
        points = np.vstack([points, points[0]])

    # 누적 거리 계산
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    
    # 재샘플링
    total_len = cum_dist[-1]
    s_new = np.arange(0, total_len, step_size)
    
    cs_x = CubicSpline(cum_dist, points[:, 0], bc_type='periodic' if is_loop else 'not-a-knot')
    cs_y = CubicSpline(cum_dist, points[:, 1], bc_type='periodic' if is_loop else 'not-a-knot')
    
    return np.column_stack([cs_x(s_new), cs_y(s_new)]), is_loop

def calc_normal_vectors(path):
    """각 점에서의 법선 벡터(Normal Vector) 계산"""
    diffs = np.diff(path, axis=0)
    # 마지막 점의 기울기는 이전 점과 동일하게 처리
    diffs = np.vstack([diffs, diffs[-1]])
    
    normals = []
    for d in diffs:
        # 90도 회전: (dx, dy) -> (-dy, dx)
        n = np.array([-d[1], d[0]])
        norm = np.linalg.norm(n)
        if norm > 0:
            normals.append(n / norm)
        else:
            normals.append(np.array([0, 1]))
    return np.array(normals)

def optimize_trajectory(ref_path, is_loop):
    """QP를 이용한 최소 곡률 경로 생성"""
    N = len(ref_path)
    normals = calc_normal_vectors(ref_path)
    
    # 결정 변수: 횡방향 이동량 n
    n = cp.Variable(N)
    
    # 비용 함수: 경로의 2차 미분(곡률) 최소화
    cost = 0
    for i in range(1, N-1):
        p_prev = ref_path[i-1] + n[i-1] * normals[i-1]
        p_curr = ref_path[i]   + n[i]   * normals[i]
        p_next = ref_path[i+1] + n[i+1] * normals[i+1]
        
        # 가속도(곡률) 벡터의 크기 최소화
        cost += cp.sum_squares(p_prev - 2*p_curr + p_next)
        
    # 제약 조건
    constraints = [
        n <= MAX_OFFSET,
        n >= -MAX_OFFSET
    ]
    
    # 루프 트랙인 경우 시작점과 끝점을 일치시킴
    if is_loop:
        constraints += [n[0] == n[-1]]
    else:
        # 루프가 아니면 시작/끝은 중심선 유지
        constraints += [n[0] == 0, n[-1] == 0]
        
    # 최적화 수행
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print("Optimizing Racing Line... (This may take a moment)")
    prob.solve(solver=cp.OSQP, verbose=True)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Optimization Failed!")
        return None
        
    # 최적 경로 좌표 계산
    opt_n = n.value
    opt_path = np.zeros_like(ref_path)
    for i in range(N):
        opt_path[i] = ref_path[i] + opt_n[i] * normals[i]
        
    return opt_path

def generate_velocity_profile(path):
    """3-Pass 알고리즘으로 물리적 한계 속도 계산"""
    # 1. 곡률 계산
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5 + 1e-6
    radius = 1.0 / curvature
    
    # 2. 횡방향 한계 속도 (Friction Limit)
    v_max_lat = np.sqrt(MU * G * radius)
    v_max_lat = np.minimum(v_max_lat, MAX_VELOCITY)
    
    # 3. 전후방 패스로 가감속 한계 적용
    N = len(path)
    v_final = np.zeros(N)
    dist = np.linalg.norm(np.diff(path, axis=0), axis=1)
    dist = np.append(dist, dist[-1]) # 마지막 거리 보정
    
    # Backward Pass (감속 제한)
    v_final[-1] = v_max_lat[-1]
    for i in range(N-2, -1, -1):
        v_allowable = np.sqrt(v_final[i+1]**2 + 2 * MAX_DECEL * dist[i])
        v_final[i] = np.min([v_max_lat[i], v_allowable])
        
    # Forward Pass (가속 제한)
    v_final[0] = np.min([v_final[0], v_max_lat[0]]) # 시작 속도
    for i in range(1, N):
        v_allowable = np.sqrt(v_final[i-1]**2 + 2 * MAX_ACCEL * dist[i-1])
        v_final[i] = np.min([v_final[i], v_allowable])
        
    return v_final, curvature

def save_csv(path, velocity, curvature, filename="optimal_trajectory.csv"):
    # Heading 계산
    headings = np.zeros(len(path))
    diffs = np.diff(path, axis=0)
    headings[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings[-1] = headings[-2] # 마지막 점 보정
    
    df = pd.DataFrame({
        'x': path[:, 0],
        'y': path[:, 1],
        'velocity': velocity,
        'heading': headings,
        'curvature': curvature
    })
    df.to_csv(filename, index=False)
    print(f"File saved to {filename}")

def visualize_road(ref_path, opt_path, velocity):
    """
    도로를 검정색 배경으로 그리고, 중앙선과 최적 경로를 시각화합니다.
    """
    # 도로 경계 계산을 위한 법선 벡터
    normals = calc_normal_vectors(ref_path)
    
    # 도로 폭의 절반
    half_width = TRACK_WIDTH / 2.0
    
    # 좌/우 경계선 계산
    left_bound = ref_path + normals * half_width
    right_bound = ref_path - normals * half_width
    
    # 도로 영역 폴리곤 생성 (좌측 경계 -> 우측 경계 역순)
    road_poly = np.vstack([left_bound, right_bound[::-1]])
    
    plt.figure(figsize=(12, 10))
    
    # 1. 도로 배경 그리기 (검정색)
    plt.fill(road_poly[:, 0], road_poly[:, 1], color='black', label='Road Surface')
    
    # 2. 중앙선 그리기 (흰색 점선)
    plt.plot(ref_path[:, 0], ref_path[:, 1], 'w--', label='Center Line', linewidth=1.5, alpha=0.8)
    
    # 3. 최적 경로 그리기
    sc = plt.scatter(opt_path[:, 0], opt_path[:, 1], c=velocity, 
                     cmap='plasma', s=5, label='Optimal Path', zorder=5)
    
    plt.colorbar(sc, label='Target Velocity [m/s]')
    plt.legend()
    plt.title("Optimal Racing Line on Generated Road")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.savefig("optimal_path.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    input_csv = "waypoints.csv" # 입력 파일명
    
    print("1. Loading Map...")
    # 경로를 로드하고 0.5m 간격으로 촘촘하게 만듭니다.
    ref_path, is_loop = load_and_resample(input_csv, step_size=0.5)
    
    print("2. Optimizing Geometry (Min Curvature)...")
    opt_path = optimize_trajectory(ref_path, is_loop)
    
    if opt_path is not None:
        print("3. Generating Velocity Profile...")
        velocity, curvature = generate_velocity_profile(opt_path)
        
        print("4. Saving Result...")
        save_csv(opt_path, velocity, curvature)
    
        print("5. Visualizing result...")
        visualize_road(ref_path, opt_path, velocity)