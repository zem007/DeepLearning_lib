# -*- encoding: utf-8 -*-
''' utils funtions for graph based models

Author: Ze Ma
Date: Feb 2020

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import logging
logging.basicConfig(level = logging.INFO)
import numpy as np
import tensorflow as tf
import math
from utils.pointcloud_processing_utils import *
from data_io.dataframe_io import DataFrameIo
from data_io.stl_io import StlIo
from data_io.ela_io import ElaIo
from utils.data_io_utils import split_dataset
from sklearn import preprocessing
import vtk

PointTypeName = 'PointType'
AnIDName = 'AnID'
PointIDName = 'PointID'
vesselType = 0;
unRuptureAnType = 1;
ruptureAnType = 2;

def load_stl_with_ela_info(stl_filename, ela_filename):
    """load stl data and its labels(in .ela file)
    """
    stl = StlIo.load(stl_filename)
    ela = ElaIo.load(ela_filename)

    point_types = vtk.vtkUnsignedCharArray.SafeDownCast(stl.GetPointData().GetArray(PointTypeName))
    an_ids = vtk.vtkUnsignedCharArray.SafeDownCast(stl.GetPointData().GetArray(AnIDName))
    point_ids = vtk.vtkUnsignedLongArray.SafeDownCast(stl.GetPointData().GetArray(PointIDName))

    ###########Process Point Type################
    if point_types == None:
        point_types = vtk.vtkUnsignedCharArray()
        point_types.SetName(PointTypeName)
        point_types.SetNumberOfComponents(1)
        point_types.SetNumberOfTuples(stl.GetNumberOfPoints())
        stl.GetPointData().AddArray(point_types)
    for i in range(len(ela)):
        point_types.InsertTuple1(i, ela[i][1])

    ###########Process Aneury ID################
    if an_ids == None:
        an_ids = vtk.vtkUnsignedCharArray()
        an_ids.SetName(AnIDName)
        an_ids.SetNumberOfComponents(1)
        an_ids.SetNumberOfTuples(stl.GetNumberOfPoints())
        stl.GetPointData().AddArray(an_ids)
    for i in range(len(ela)):
        an_ids.InsertTuple1(i, ela[i][2])

    ###########Process Point ID################
    if point_ids == None:
        point_ids = vtk.vtkUnsignedLongArray()
        point_ids.SetName(PointIDName);
        point_ids.SetNumberOfComponents(1)
        point_ids.SetNumberOfTuples(stl.GetNumberOfPoints())
        stl.GetPointData().AddArray(point_ids)
    for i in range(len(ela)):
        point_ids.InsertTuple1(i, ela[i][0])

    return stl

def read_all_info(df):
    """ read all info from a df that contains all necessary data for this project
    """
    x = df.iloc[:, -3].tolist()
    x = np.array(x)
    labels = df.iloc[:, -2].tolist()
    labels = np.array(labels)
    adj = df.iloc[:, -1].tolist()
    adj = np.array(adj)
    info = df.iloc[:, 4: -3].values
    return x, labels, adj, info

# this is for multiple an
# pid shape= N*2, column1:labels(0-vessel,1-unruptured, 2-ruptured), column2:an_labels(0-no an, 1-an#1, 2-an#2, ...)
def get_aneurysm_points(data, pid):
    """ get a list that contains all aneurym points from the point cloud,
       each point is a list contains xyz coordinates location. eg. [[1,2,3], [2, 3, 10], ...]
    """
    an_nums = np.max(pid[:, 1])
    ids = range(pid.shape[0])
    an_list = []
    for i in range(an_nums):
        print('an #: ', (i+1))
        one_an = []
        for id in ids:
            if pid[id, 0] == 1 or pid[id, 0] == 2:
                if pid[id, 1] == (i+1):
                    one_an.append(data[id, :])
        one_an = np.asarray(one_an)
        an_list.append(one_an)
    return an_list
    
# decimate_ratio controls the vertex density
def generate_point_data(df, decimate_ratio, num_point = 1024, save = False, filename = None):
    """ generator the pointcloud, labels, and adj_matrix from the stl and ela data,
       then merged with the clinical data, return to a new df with all that info
    """
    for i in range(len(df)):
        print('----------------------------------------------')
        print(df.iloc[i]['file_name'][:-4])
        elapoly = load_stl_with_ela_info(df.iloc[i]['stl'], df.iloc[i]['ela'])
        decimated_polydata = decimate_polydata(elapoly, decimate_ratio, preserve_topology = True)  #stl
        
        ### 从down sampling之后的vtk文件中获取stl，ela，graph dictionary ###
        stldata, eladata, graph = get_info_from_polydata(decimated_polydata)
        
        # processing anone by one
        an_list = get_aneurysm_points(stldata, eladata[:, 1:])
        for j in range(len(an_list)):
            an_data = an_list[j]
            num_an_points = an_data.shape[0] # 动脉瘤点的总数量
            if num_an_points == 0: 
                break
            print('the nums of #%d an is: '%(j+1), str(num_an_points))
            bb = get_bounding_box(an_data)
            an_center=[0,0,0]
            bb.GetCenter(an_center)

            temp_sampled_points, temp_sampled_ids = get_n_points(stldata, an_center, num_point) #output two arrays
            sampled_points = temp_sampled_points   # np.array of shape [sample, 3]
            sampled_ids = list(temp_sampled_ids)
            sampled_ela = [eladata[i,1] for i in sampled_ids]  # list of len = sample, 0 for vessel, 1 for unruptured AN, 2 for ruptured AN
            assert(sampled_points.shape == (num_point, 3))

            # pointcloud normalization
            sampled_points = normlize_pointcloud(sampled_points)

            # get adj_normalized
            adj_normalized = get_adj_normalized(graph, sampled_ids)

            df.set_value(i+j, 'points_arr', sampled_points)
            df.set_value(i+j, 'labels_list', sampled_ela)
            df.set_value(i+j, 'adj_normalized', adj_normalized)
            
    if save == True:
        DataFrameIo.save_df(filename, df)
    return df

def generate_triangle_data(df, decimate_ratio, num_triangle = 1024, save = False, filename = None):
    """ generator the triangle pointcloud, labels, and adj_matrix from the stl and ela data,
       then merged with the clinical data, return to a new df with all that info
    """
    for w in range(len(df)):
        print('----------------------------------------------')
        print(df.iloc[w]['file_name'][:-4])
        elapoly = load_stl_with_ela_info(df.iloc[w]['stl'], df.iloc[w]['ela'])
        decimated_polydata = decimate_polydata(elapoly, decimate_ratio, preserve_topology = True)  #stl
        point_dic, tirangle_point_dic, triangle_neighbor_dic = get_polydata_triangle_neighbors(decimated_polydata)

        ### 从down sampling之后的vtk文件中获取stl ###
        stldata = np.array(list(point_dic.values()))   # N*3
        ### 从down sampling之后的vtk文件中获取对应的ela ###
        eladata = get_ela_from_polydata(decimated_polydata)
        eladata = eladata.astype(int)  # N*3
        # processing an one by one
        an_list = get_aneurysm_points(stldata, eladata[:, 1:])
        an_data = an_list[0]
        num_an_points = an_data.shape[0] # 动脉瘤点的总数量
        print('the nums of #%d an is: '%(1), str(num_an_points))
        bb = get_bounding_box(an_data)
        an_center=[0,0,0]
        bb.GetCenter(an_center)

        # want 1024 triangles, each one has 6 vertex points, so that form a 1024*(6*3) array
        # label is a len=1024 list. if any vertex in that triangle belongs to an, label = 1 or 2
        nearest_ids = find_nearest_ids(stldata, tirangle_point_dic, an_center, num_triangle) 
        sampled_points, sampled_labels = get_sampled_labels_by_ids(nearest_ids, triangle_neighbor_dic, tirangle_point_dic, eladata)
        assert(len(sampled_labels) == num_triangle)
        final_matrix = get_sampled_coords(sampled_points, point_dic)
        assert(final_matrix.shape == (num_triangle, 18))
        
        df.set_value(w, 'points_arr', final_matrix)
        df.set_value(w, 'labels_list', sampled_labels)
    if save == True:
        DataFrameIo.save_df(filename, df)
    return df

def convert_to_cls_labels(y_array):
    """
    Args: 
        y_array: [N, sample_nums] array;
    Return:
        labels_array: 1d array of shape[N,], the classification label for each sample
    """
    labels_list = []
    for i in range(len(y_array)):
        if max(y_array[i]) == 1 or max(y_array[i]) == 0:
            labels_list.append(0)
        elif max(y_array[i]) == 2:
            labels_list.append(1)
#         else:
#             print(str(max(y_array[i])))
#             print(str(i))
    labels_array = np.array(labels_list)
    return labels_array

def convert_to_seg_labels(y_array):
    """
    Args: 
        y_array: [N, sample_nums] array;
    Return:
        y_array: [N, 1024] array, only contains 0 and 1 values for vessel and aneuryms
    """
    assert(len(y_array.shape) == 2)
    for i in range(y_array.shape[0]):
        for j in range(y_array.shape[1]):
            if y_array[i,j] == 2 or y_array[i,j] == 1:
                y_array[i,j] = 1

    return y_array


