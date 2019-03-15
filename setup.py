from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    setup(name='horsepics',
        version='0.1',
        description='Image processing library for equines (and people too)',
        url='https://github.com/JimBoonie/HorsePics.git',
        author='Mason McGough',
        author_email='mcgough.mason@gmail.com',
        license='MIT',
        packages=find_packages(),
        install_requires = list(f.read().splitlines()),
        zip_safe=False)