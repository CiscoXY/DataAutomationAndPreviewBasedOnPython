# -*- coding: utf-8 -*-
from os import walk
from os import path as os_path
from sys import path,argv,exit
for root, dirs, files in walk(os_path.dirname(os_path.dirname(os_path.realpath(__file__)))):
    path.append(root) #* 完成路径的添加，路径包括：该文件所在路径，该文件所在文件夹的所有子文件夹的路径（子文件夹的子文件夹也会添加）