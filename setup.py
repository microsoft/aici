from setuptools import setup

setup(
    name='pyaici',
    version='0.1',
    packages=['pyaici'],
    entry_points={
        'console_scripts': [
            'aici = pyaici.cli:main'
        ]
    },
)
