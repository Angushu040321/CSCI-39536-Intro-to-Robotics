from setuptools import find_packages, setup

package_name = 'final_lab'

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
    maintainer='Final Lab Group',
    maintainer_email='group@todo.todo',
    description='TurtleBot3 color following package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_follower = final_lab.color_follower:main',
        ],
    },
)