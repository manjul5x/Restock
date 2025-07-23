from setuptools import setup, find_packages

setup(
    name="forecaster",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.3.1",
        "numpy>=2.0.2",
        "plotly>=6.2.0",
        "prophet>=1.1.7",
        "flask>=3.1.1",
        "scikit-learn>=1.6.1",
        "statsmodels>=0.14.5",
        "matplotlib>=3.9.4",
        "seaborn>=0.13.2",
        "tqdm>=4.67.1",
        "holidays>=0.77",
        "cmdstanpy>=1.2.5",
        "narwhals>=1.48.0"
    ],
    author="Your Name",
    description="Inventory Forecasting & Simulation Suite",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
        'console_scripts': [
            'run-backtest=run_customer_backtest:main',
            'run-safety-stocks=run_safety_stock_calculation:main',
            'run-simulation=run_simulation:main',
        ],
    },
    include_package_data=True,
    package_data={
        'forecaster': [
            'data/dummy/*.csv',
            'webapp/templates/*.html',
            'webapp/static/*'
        ],
    },
)
