from setuptools import setup, find_packages

setup(
    name='titanic-survival-predictor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn'
        
    ],
    author='Ghodake Shreyas',
    author_email='shreyasghodake999@gmail.com',
    description='Predicting Titanic Passenger Survival',
    url='https://github.com/shreyas-codes25/CodSoft-Internship',
    
)