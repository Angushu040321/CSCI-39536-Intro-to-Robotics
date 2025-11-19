import os
from glob import glob
from setuptools import setup

# 1. CRITICAL: Ensure this line is present and correct
package_name = 'lab5_nav' 

setup(
    # 2. CRITICAL: Ensure this line correctly uses the variable
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files in the 'launch' subdirectory
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][y]'))),
        # Include all config files in the 'config' subdirectory
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        # Include the maps directory if needed
        (os.path.join('share', package_name, 'maps'), glob(os.path.join('maps', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='angushu',
    maintainer_email='your_email@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)