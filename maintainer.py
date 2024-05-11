"""
The module that contains all the necessary logic for processing jobs in the database queue.
"""

import logging

from decouple import config

# import the storage provider that you would like to use
# currently we have dropbox and mongodb
from sqooler.storage_providers.mongodb import MongodbProvider
from sqooler.schemes import MongodbLoginInformation
from sqooler.utils import update_backends, main

from singlequdit.config import spooler_object as sq_spooler
from multiqudit.config import spooler_object as mq_spooler
from fermions.config import spooler_object as f_spooler
from rydberg.config import spooler_object as ryd_spooler
# configure the backends
backends = {
    "singlequdit": sq_spooler,
    "multiqudit": mq_spooler,
    "fermions": f_spooler,
    "rydberg": ryd_spooler,
}

# configure the storage provider

mongodb_username = config("MONGODB_USERNAME")
mongodb_password = config("MONGODB_PASSWORD")
mongodb_database_url = config("MONGODB_DATABASE_URL")
login_dict = {
    "mongodb_username": mongodb_username,
    "mongodb_password": mongodb_password,
    "mongodb_database_url": mongodb_database_url,
}
mongodb_login = MongodbLoginInformation(**login_dict)
storage_provider = MongodbProvider(mongodb_login)

logging.basicConfig(level=logging.INFO)

logging.info("Update")
update_backends(storage_provider, backends)
logging.info("Now run as usual.")
main(storage_provider, backends)
