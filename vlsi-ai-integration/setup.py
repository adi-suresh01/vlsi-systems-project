from setuptools import setup, find_packages

setup(
    name='vlsi-ai-integration',
    version='0.1.0',
    author='Aditya Suresh',
    author_email='adi22tya@gmail.com',
    description='A project for integrating VLSI hardware with AI inference pipelines.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow==2.11.0',
        'numpy',
        'fastapi',
        'uvicorn',
        'matplotlib',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'train-model=scripts.train_model:main',
            'run-simulation=scripts.run_simulation:main',
            'benchmark=scripts.benchmark:main',
        ],
    },
)