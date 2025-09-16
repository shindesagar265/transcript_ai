%run ./01_Initialize

import requests
import json
import os
import glob
from pyspark.sql.functions import max, udf, collect_list, concat_ws, lit, current_timestamp
