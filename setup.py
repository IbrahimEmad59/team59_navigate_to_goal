from setuptools import find_packages, setup

package_name = 'team59_navigate_to_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ibrahim',
    maintainer_email='ibrahim.alshayeb59@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "get_object_range=team59_navigate_to_goal.get_object_range:main",
            "go_to_goal=team59_navigate_to_goal.go_to_goal:main",
            "waypoints_loader=team59_navigate_to_goal.waypoints_loader:main",
            "chase_object_with_waypoints=team59_navigate_to_goal.chase_object_with_waypoints:main", 
            "obstacle_avoidance=team59_navigate_to_goal.obstacle_avoidance:main", 
        ],
    },
)
