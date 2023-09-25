from setuptools import setup

setup(
    name='Compyct',
    version='0.0.1',
    packages=['compyct'],
    package_dir={'compyct':'src'},
    url='',
    license='',
    author='Samuel James Bader',
    author_email='samuel.james.bader@gmail.com',
    description='Compact model fitting',
    install_requires=['panel','numpy','scipy','verilogae'],
    extras_require={
        'spectre':['pynut','pyspectre','pexpect'],
        'ngspice':['PySpice'],
    }
)

