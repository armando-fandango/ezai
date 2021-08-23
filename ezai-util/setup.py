import setuptools

with open('../version.txt', 'r') as vf:
    __version__ = vf.read().strip()

# TODO: Replace the long description comments once cleared
# with open("../README.md", "r", encoding="utf-8") as fh:
#    long_description = fh.read()
long_description = " "

setuptools.setup(
    name="ezai_util",
    version=__version__,
    author="Armando Fandango",
    author_email="armando@neurasights.com",
    description="EZAI Utilities and Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    include_package_data=True,
    # data_files=[('n',['n/version.txt'])],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'opencv-python',
        'pytest',
        'click',
        #'gym>=0.18.0',
        #'gym_unity==0.27.0'
    ],
    entry_points={
        "console_scripts": [
#            "n_env_test=n_envs.env_tst:main",
        ]
    },
)
