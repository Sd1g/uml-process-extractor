from setuptools import setup, find_packages

setup(
    name="uml-process-extractor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask==2.3.3',
        'Flask-CORS==4.0.0',
        'numpy==1.24.3',
        'plotly==5.14.1',
        'gunicorn==20.1.0',
    ],
    python_requires='>=3.6',
)