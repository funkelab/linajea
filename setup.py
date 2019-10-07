from setuptools import setup

setup(
        name='linajea',
        version='1.1.1',
        description='Lineage Tracking in 4D Microscopy Volumes.',
        url='https://github.com/funkelab/linajea',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'linajea',
            'linajea.gunpowder',
            'linajea.tensorflow',
            'linajea.tracking',
            'linajea.evaluation',
            'linajea.process_blockwise',
        ]
)
