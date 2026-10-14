from glob import glob

from setuptools import setup

package_name = 'dance_director'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO maintainer name',
    maintainer_email='you@example.com',
    description='Director node: publishes the choreography to every robot.',
    license='Proprietary',
    entry_points={
        # `ros2 run dance_director director` starts the node.
        'console_scripts': [
            'director = dance_director.director_node:main',
            'start_dance = dance_director.start_dance:main',
        ],
    },
)