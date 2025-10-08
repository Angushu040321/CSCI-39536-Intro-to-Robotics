from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lab3_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', 
glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mpandretti',
    maintainer_email='michael.andretti31@myhunter.cuny.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'odom_logger = lab3_pkg.odom_logger:main',
        'plot_odom = lab3_pkg.plot_odom:main',
        'square = lab3_pkg.square:main',
        'pd_controller = lab3_pkg.pd_controller:main',
        'wall_following = lab3_pkg.wall_following:main',
        'compare_logger = lab3_pkg.compare_logger:main',
        'compare_plot = lab3_pkg.compare_plot:main',
        ],
    },
)
