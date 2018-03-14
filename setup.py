from setuptools import setup

setup(
    name='trackAnalyzer',    # This is the name of your PyPI-package.
    version='0.1',                          # Update the version number for new releases
    scripts=['trackAnalyzer']                  # The name of your scipt, and also the command you'll be using for calling it
	description = 'functions to analyze 3D tracks of animals',
	author = 'Jason Duguay',
	author_email = 'duguay.jason@gmail.com',
	url = 'https://github.com/Fishified/trackAnalyzer', # use the URL to the github repo
	download_url = 'https://github.com/Fishified/trackAnalyzer/archive/0.1.tar.gz', # I'll explain this in a second
	
	)