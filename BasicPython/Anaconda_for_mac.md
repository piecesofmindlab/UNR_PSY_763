# Setting up anaconda (principally for mac)

* installed homebrew:
ruby -e "$(curl...)"

* Make sure you are using bash shell. Some machines configured for particular purposes may default to other shell environments (e.g., if you use Freesurfer, your machine may default to t-c-shell or tcsh)

* installed wget w/ homebrew
`wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-MacOSX-x86_64.sh`

for imac1: 

WARNING:
    You currently have a PYTHONPATH environment variable set. This may cause
    unexpected behavior when running the Python interpreter in Anaconda3.
    For best results, please verify that your PYTHONPATH only points to
    directories of packages that are compatible with the Python interpreter
    in Anaconda3: /Users/corebadmin/anaconda3
Do you wish the installer to prepend the Anaconda3 install location
to PATH in your /Users/corebadmin/.bash_profile ? [yes|no]

[Said yes. Unclear if this was the right thing to do.]