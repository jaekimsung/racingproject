from setuptools import find_packages, setup


package_name = "racingproject"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/data",
            [
                "src/racingproject/data/waypoints.csv",
                "src/racingproject/data/optimal.csv",
            ],
        ),
        ("share/" + package_name + "/launch", ["src/launch/racing_launch.py", "src/launch/racing_stanley_launch.py"]),
        ("lib/" + package_name, ["scripts/racing_node", "scripts/racing_node_stanley"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "cvxpy", "rclpy", "ament_index_python"],
    zip_safe=True,
    maintainer="Chadol Team",
    maintainer_email="user@example.com",
    description="Racing controller for SKKU Mobile System Control.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "racing_node = racingproject.racing_node:main",
            "racing_node_stanley = racingproject_stanley.racing_node:main",
        ],
    },
)
