from setuptools import setup, find_packages

setup(
    name='xedata',
    packages=['xedata'],
    version='1.0.0',
    author='Carlo Fuselli',
    author_email='cfuselli@nikhef.nl',
    description='A package for processing and saving XENONnT data',
    entry_points={
    'console_scripts': [
        'xedata = xedata.submitter:main',
        'xd = xedata.submitter:main'
    ]},
    scripts=['bin/sqq']
)
