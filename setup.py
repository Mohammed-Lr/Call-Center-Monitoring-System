from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="call-center-monitoring-system",
    version="1.0.0",
    author="Your Name",
    description="AI-powered call center compliance monitoring system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "face-recognition>=1.3.0",
        "ultralytics>=8.0.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        'console_scripts': [
            'ccms-monitor=scripts.run_full_monitoring:main',
        ],
    },
)