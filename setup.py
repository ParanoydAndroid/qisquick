import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='qisquick',
    packages=['qisquick'],
    version='0.0a5',
    license='MIT',
    description='Utility library for automating running and analyzing transpiler experiments with IBM qiskit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Brandon K Kamaka',
    author_email='brandon.kamaka@gmail.com',
    url='https://github.com/ParanoydAndroid/qisquick',
    keywords=['transpiler', 'qiskit', 'quantum'],
    install_requires=[
        'qiskit',
        'numpy',
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
    python_requires='>=3.6',
    data_files=[('docs', ['qisquick_documentation.pdf', 'README.md'])]
)
