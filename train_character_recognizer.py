# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:36:16 2021

@author: alest
"""

with open("gzip/emnist-bymerge-train-images-idx3-ubyte", "rb") as f:
    byte = f.read(1)
    while byte:
        # Do stuff with byte.
        byte = f.read(1)