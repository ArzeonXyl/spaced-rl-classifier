from setuptools import setup, find_packages

setup(
    name="spaced_rl_classifier",
    version="0.1.0",
    author="ArzeonXyl",
    description="Spaced Retrieval + RL Scheduler classifier library",
    packages=find_packages(),   # otomatis cari folder/package
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "scikit-learn>=1.2",
        "pandas>=2.0",
        "numpy>=1.24",
    ],
)
