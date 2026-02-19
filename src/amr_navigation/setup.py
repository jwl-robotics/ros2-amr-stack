from setuptools import setup
import os
from glob import glob

package_name = 'amr_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jin Wei Lim',
    maintainer_email='jinweilim22@hotmail.com',
    description='Navigation configuration and launch files for the AMR stack',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
