# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os

try:
    PROJECT_ROOT_DIR = os.environ.get("PROJECT_ROOT_DIR") 
except:
    PROJECT_ROOT_DIR = '/data1/PycharmProjects/multitask'


class MyPath(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/data2/dataset/'
        db_names = {'PASCAL_MT', 'NYUD_MT', 'nyuv2'}

        if database in db_names:
            return os.path.join(db_root, database)
        
        elif not database:
            return db_root
        
        else:
            raise NotImplementedError

    @staticmethod
    def seism_root():
        return '/path/to/seism/'
