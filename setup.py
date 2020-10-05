from imagerie import __VERSION__
import setuptools

long_description = '# Imagerie lite'

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='imagerie',
    version=__VERSION__,
    author='Ibragim Abubakarov',
    author_email='ibragim.ai95@gmail.com',
    maintainer='Ibragim Abubakarov',
    maintainer_email='ibragim.ai95@gmail.com',
    description='Python package grouping together common useful functions and operations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ibragim64/imagerie/tree/lite',
    packages=['imagerie', 'imagerie.ndimage', 'imagerie.operations', 'imagerie._lib'],
    install_requires=['opencv-python', 'pillow', 'numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia'
    ]
)
