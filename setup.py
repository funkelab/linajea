from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

setup(
        name='linajea',
        version='0.1',
        description='Lineage Tracking in 4D Microscopy Volumes.',
        url='https://github.com/funkelab/linajea',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'linajea',
            'linajea.gp',
        ],
        ext_modules=[
            Extension(
                'linajea.target_counts',
                sources=['linajea/target_counts.pyx'],
                extra_compile_args=['-O3'])
        ],
        cmdclass={'build_ext': build_ext}
)
