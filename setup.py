import pathlib
from setuptools import setup

exec(open('DeepSaki/version.py').read())

# https://setuptools.pypa.io/en/latest/references/keywords.html

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='DeepSaki',
    version=__version__,    
    description='Custom deep learning code for TensorFlow',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/sascha-kirch/DeepSaki',
    author='Sascha Kirch',
    author_email='susch130993@googlemail.com',
    license='MIT',
    packages=['DeepSaki',
              'DeepSaki.activations',
              'DeepSaki.optimizer',
              'DeepSaki.layers',
              'DeepSaki.layers.helper',
              'DeepSaki.models',
              'DeepSaki.initializer',
              'DeepSaki.initializer.helper',
              'DeepSaki.loss',
              'DeepSaki.utils',
              'DeepSaki.regularization',
             ],
    install_requires=['tensorflow >= 2.8',
                      'numpy',
                      'tensorflow_addons',
                      'pandas',
                      ],
    extras_require={
        'extras': ['matplotlib',],
        },
    keywords = ['deeplearning', 'machinelearning', 'tensorflow','TPU'],
  
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
