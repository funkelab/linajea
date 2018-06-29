from setuptools import setup
import subprocess

setup(
        name='linajea',
        version='0.1',
        description='Lineage Tracking in 4D Microscopy Volumes.',
        url='https://github.com/funkey/linajea',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'linajea',
            'linajea.gp',
        ],
        install_requires=[
            "gunpowder"
        ]
)
