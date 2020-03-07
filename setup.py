import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='qisquick',
    packages=['qisquick'],
    include_package_data=True,
    version='0.0.5',
    license='MIT',
    description='Utility library for automating running and analyzing transpiler experiments with IBM qiskit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Brandon K Kamaka',
    author_email='brandon.kamaka@gmail.com',
    url='https://github.com/ParanoydAndroid/qisquick',
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',  # I explain this later on
    keywords=['transpiler', 'qiskit', 'quantum'],
    install_requires=[
        'qiskit',
        'numpy',
        'matplotlib',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Environment :: Win32 (MS Windows)'
    ],
    python_rewuires='>=3.6'
)
