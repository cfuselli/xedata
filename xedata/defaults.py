import os, glob

USER = os.getenv('USER', default=None)

MY_PATH = f'/dali/lgrandi/{USER}/'
OUTPUT_FOLDER = f'/dali/lgrandi/{USER}/test_xedata/'

XEDATA_PATH = os.path.join(MY_PATH, 'software/xedata')

