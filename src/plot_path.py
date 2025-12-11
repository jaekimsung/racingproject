import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def visualize_path(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: 파일 '{csv_path}'를 찾을 수 없습니다.")
        return

    print(f"Loading path from {csv_path}...")
    
    # ROS 노드와 동일한 방식으로 CSV 로드 (헤더 유무 자동 처리)
    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except ValueError:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.shape[1] < 2:
        print("Error: CSV 파일은 최소 2개(x, y)의 열을 가져야 합니다.")
        return

    # X, Y 좌표 추출
    x = data[:, 0]
    y = data[:, 1]

    # 그래프 설정
    plt.figure(figsize=(12, 10))
    
    # 전체 경로 그리기
    plt.plot(x, y, '-b', linewidth=1, label='Path')
    
    # 시작점(초록색)과 끝점(빨간색) 표시
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start (Idx 0)')
    plt.plot(x[-1], y[-1], 'rx', markersize=10, label=f'End (Idx {len(x)-1})')

    # 인덱스 마커 표시 (너무 빽빽하지 않게 전체의 50개 정도만 표시)
    total_points = len(x)
    step = max(1, total_points // 50)  
    
    for i in range(0, total_points, step):
        plt.plot(x[i], y[i], '.k', markersize=3)
        # 텍스트로 인덱스 번호와 좌표 표시
        label = f"{i}\n({x[i]:.1f}, {y[i]:.1f})"
        plt.annotate(label, (x[i], y[i]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8, alpha=0.7)

    # 그래프 꾸미기
    plt.title(f"Path Visualization: {os.path.basename(csv_path)}")
    plt.xlabel("Global X [m]")
    plt.ylabel("Global Y [m]")
    plt.axis('equal')  # 비율을 실제와 같게 설정 (찌그러짐 방지)
    plt.grid(True)
    plt.legend()
    
    # 마우스 커서를 올리면 좌표가 나오는 인터랙티브 모드
    plt.show()

if __name__ == "__main__":
    # 사용법: python plot_path.py <파일경로>
    # 파일 경로를 직접 아래 변수에 넣어서 실행해도 됩니다.
    
    # 예시: 터미널 인자가 있으면 그걸 쓰고, 없으면 기본값 사용
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        visualize_path(file_path)
    else:
        # 여기에 테스트할 csv 파일 경로를 입력하세요
        default_path = "/home/jaekimsung/mobilesystemcontrol/racingproject/src/racingproject/data/final_margin1m.csv" 
        print(f"인자가 전달되지 않아 기본 경로를 시도합니다: {default_path}")
        print("사용법: python3 plot_path.py <csv_file_path>")
        visualize_path(default_path)