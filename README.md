# sqooler-example
This is a template for a number of quantum simulators that can be used with the `qiskit-cold-atom` and the `qlued` interface.

We are proud to be currently sponsored by the *Unitary Fund*. It enables us to set up a good test environment and make it as straight-forward as possible to integrate cold atoms with circuits. This documentation will improve as the good goes beyond the initial piloting status. 

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund) 

To enable this work-flow, the simulator has to follow a few rules on how to parse the json files etc. This is what we have started to standardize and simplify as much as possible. In the following we documented each module its purpose and look forward to your contributions.

## Getting started

The simplest way to use the package is to deploy it to `heroku`. This directly starts the `maintainer.py` in a loop, because it is defined like that in the `Procfile`.  However, you will also need to have the following credentials of the Dropbox:

- `APP_KEY`, `APP_SECRET` and `REFRESH_TOKEN`. Please head over to the documentation of `qlued` to see how they might be set up.
- They should be all defined `Settings` > `Config Vars`. 
- Now your system  should automatically look for jobs that are under `Backend_files/Queued_Jobs`, process them and safe the result under `Backend_files/Finished_Jobs`.


## Getting started locally
    
> :warning: This part of the documentiation needs a lot of love. Feel free to help us making it more understandable.

If you would like to write some new simulator, extend it etc, you will need to deploy the code locally. Then you will need to:

- clone or fork the repo.
- pip install the `requirements.txt`.
- define `APP_KEY`, `APP_SECRET` and `REFRESH_TOKEN` in the `.env` file that is you should create in the root directory.
- run the maintainer with `python maintainer.py`.