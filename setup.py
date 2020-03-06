from distutils.core import setup

setup(
    name='qisquick',
    packages=['qisquick'],
    version='0.0.5',
    license='MIT',
    description='Utility library for automating running and analyzing transpiler experiments with IBM qiskit',
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
        'Intended Audience :: Researchers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
