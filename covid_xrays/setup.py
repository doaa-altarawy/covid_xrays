import setuptools


def read_requirements():
    """parses requirements from requirements.txt"""

    with open('requirements.txt') as f:
        req = f.readlines()

    return req


if __name__ == "__main__":
    setuptools.setup(
        name='covid_xrays',
        version="0.1.0",
        description='Screening for COVID 19 from x-ray images.',
        author='Doaa Altarawy',
        author_email='doaa.altarawy@gmail.com',
        url="https://github.com/doaa-altarawy/covid_xrays.git",
        license='MIT',

        packages=setuptools.find_packages(),

        install_requires=read_requirements(),

        include_package_data=True,

        extras_require={
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'codecov',
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
        },

        tests_require=[
            'codecov',
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
