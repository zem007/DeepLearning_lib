"""processing utils for point cloud data type
"""
import numpy as np
from common.variables import *
import pandas as pd
import vtk
import networkx as nx
import scipy.sparse as sp


def decimate_polydata(polydata, reduction, preserve_topology=True):
    """ Reduce the number of triangles in a triangle mesh.
        Args:
            polydata: vtkPolyData data format
            reduction: The desired reduction in the total number of polygons, 
                        e.g., if TargetReduction is set to 0.9, this filter will 
                        try to reduce the data set to 10% of its original size.
            preserveTopology: Whether preserve the topology of polydata
    """
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(reduction)
    decimate.SetPreserveTopology(preserve_topology)
    if not preserve_topology:     
        decimate.SetBoundaryVertexDeletion(1)
        decimate.SplittingOn()
        decimate.SetMaximumError(vtk.VTK_DOUBLE_MAX)
    decimate.Update()
    return decimate.GetOutput()

def get_polydata_triangle_neighbors(polydata):
    """ Get the triangle neighbors of polydata.
        Args:
            polydata: vtkPolyData data format
        Returns:
            pointDic: points dictionary, key is pointID and value is point location
            tiranglePointDic: triangle dictionary, key is triangleID and value is pointID of the three vertex
            triangleNeighborDic: triangle neighbor dictionary, key is triangleID and value is triangleID of neighbor triangles.
                                The neighbor triangle number could be 3 or 2 or 1
    """  
    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(polydata)
    triangleFilter.Update()
    polydata = triangleFilter.GetOutput()
    points = polydata.GetPoints()
    NumberOfTriangles = polydata.GetNumberOfCells()
    triangleNeighborDic={}
    tiranglePointDic={}
    pointDic={}
    for pointId in range(points.GetNumberOfPoints()):
        point=[0,0,0]
        points.GetPoint(pointId,point)
        pointDic[pointId] = point
    for cellId in range(NumberOfTriangles):
        cellPointIds = vtk.vtkIdList()
        polydata.GetCellPoints(cellId,cellPointIds)
        neighbors = []
        points = []
        for i in range(cellPointIds.GetNumberOfIds()):
            points.append(cellPointIds.GetId(i))
            idList = vtk.vtkIdList()
            idList.InsertNextId(cellPointIds.GetId(i))
            if (i + 1 == cellPointIds.GetNumberOfIds()):
                idList.InsertNextId(cellPointIds.GetId(0))
            else:
                idList.InsertNextId(cellPointIds.GetId(i + 1))
            neighborCellIds = vtk.vtkIdList()
            polydata.GetCellNeighbors(cellId,idList,neighborCellIds)
            for j in range(neighborCellIds.GetNumberOfIds()):
                neighbors.append(neighborCellIds.GetId(j))
        triangleNeighborDic[cellId] = neighbors
        tiranglePointDic[cellId] = points
    return pointDic,tiranglePointDic,triangleNeighborDic

def get_ela_from_polydata(polydata):
    """ Get the ela labels from polydata.
        Args:
            polydata: vtkPolyData data format
        Returns:
            labels_array: N*3 array
    """  
    pointTypes = vtk.vtkUnsignedCharArray.SafeDownCast(polydata.GetPointData().GetArray('PointType'))
    anIds = vtk.vtkUnsignedCharArray.SafeDownCast(polydata.GetPointData().GetArray('AnID'))
    pointIds = vtk.vtkUnsignedLongArray.SafeDownCast(polydata.GetPointData().GetArray('PointID'))
    
    if pointTypes == None:
        raise Exception("No point type array in polydata!")
    
    if anIds == None:
        raise Exception("No aneury id array in polydata!")
    
    if pointIds == None:
        raise Exception("No point id array in polydata!")
        
    if ((pointTypes.GetNumberOfTuples() != anIds.GetNumberOfTuples()) or 
        (pointTypes.GetNumberOfTuples() != pointIds.GetNumberOfTuples()) or 
        (anIds.GetNumberOfTuples() != pointIds.GetNumberOfTuples())):
        raise Exception("Mismatch array size !")
        
    ela = []
    for i in range(pointIds.GetNumberOfTuples()):
        onePoint = [pointIds.GetTuple1(i),pointTypes.GetTuple1(i),anIds.GetTuple1(i)]
        ela.append(onePoint)
        
    return np.array(ela)

def get_bounding_box(points):
    """ storing all points coordinates in vtk bounding box
        Args:
            points: list, storing the x y z coordinates of each point
        Returns:
            bound_box: vtk bounding box
    """ 
    bound_box = vtk.vtkBoundingBox()
    [bound_box.AddPoint(p[0],p[1],p[2]) for p in points]
    return bound_box


def get_polydata_as_graph(polydata):
    """ Get the adjacent info for each node, and store it in a dict. eg. {node#1: [node#3, node#5, ...], node#2: [...]}
        Args:
            polydata: vtkPolyData data format
        Returns:
            graph: an adjacent info dict 
    """ 
    points = polydata.GetPoints()
    NumberOfPoints = points.GetNumberOfPoints()
    graph={}
    for id in range(NumberOfPoints):
        graph[id] = [];
        cellIds = vtk.vtkIdList()
        polydata.GetPointCells(id,cellIds)
        for cid in range(cellIds.GetNumberOfIds()):
            cell = polydata.GetCell(cellIds.GetId(cid))
            for i in range(cell.GetNumberOfEdges()):
                edge = cell.GetEdge(i)
                pointIdList = edge.GetPointIds()
                isConnected = False
                for p in range(pointIdList.GetNumberOfIds()):
                    if id == pointIdList.GetId(p):
                        isConnected = True
                if isConnected:
                    for p in range(pointIdList.GetNumberOfIds()):
                        neighbourPointID = pointIdList.GetId(p)
                        if (id != neighbourPointID) and (neighbourPointID not in graph[id]):
                            graph[id].append(neighbourPointID)
    return graph

def get_info_from_polydata(decimated_polydata):
    """ Get stl, ela, adj info from decimated polydata
        Args:
            decimated_polydata: vtkPolyData data format
        Returns:
            stldata: N*3 array, 3d coordinates of each point
            eladata: N*3 array, label of each point
            graph: an adjacent info dict
    """ 
    ### 从down sampling之后的vtk文件中获取stl ###
    pt_list = [[p for p in decimated_polydata.GetPoint(j)] for j in range(decimated_polydata.GetNumberOfPoints())]
    stldata = np.array(pt_list)
    ### 从down sampling之后的vtk文件中获取对应的ela ###
    eladata = get_ela_from_polydata(decimated_polydata)
    eladata = eladata.astype(int)
    ### 得到down sampling之后的dict
    graph = get_polydata_as_graph(decimated_polydata)
    assert(len(graph) == len(stldata))
    
    return stldata, eladata, graph

def convert_points_to_vtk_point_set(points):
    """ convert points to vtk point set
        Args:
            points: list of each point 3d coordinates
        Returns:
            polydata: vtkPolyData data format
    """  
    vtkpoints = vtk.vtkPoints()
    vertices = vtk.vtkCellArray();
    [vtkpoints.InsertNextPoint(p[0],p[1],p[2]) for p in points]
    polydata = vtk.vtkPolyData();
    polydata.SetPoints(vtkpoints)
    return polydata

def get_n_points(points, center, numberOfPoints):
    """ convert points to vtk point set
        Args:
            points: list of each point 3d coordinates
            center: list, 3d coord of center point of aneurysm
            numberOfPoints: int, eg, 1024, 2048 ...
        Returns:
            pointArray: array, output array of target nums of points
            pointIDArray: array, output array of target points' ID
    """  
    locator = vtk.vtkPointLocator()
    idList = vtk.vtkIdList()
    polydata = convert_points_to_vtk_point_set(points)
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    locator.FindClosestNPoints(numberOfPoints,center,idList)
    outPoints = vtk.vtkPoints()
    polydata.GetPoints().GetPoints(idList,outPoints)
    outPointList = [[p for p in outPoints.GetPoint(i)] for i in range(outPoints.GetNumberOfPoints())]
    outIDList = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
    return np.array(outPointList),np.array(outIDList)

def normlize_pointcloud(sampled_points):
    """ norm target sampled points
        Args:
            sampled_points: array, original points 3d coords
        Returns:
            sampled_points: normlized points 3d coords
    """  
    center = np.mean(sampled_points, axis = 0)
    sampled_points -= center
    furthest_distance = np.max(np.sqrt(np.sum(abs(sampled_points)**2,axis=-1)))
    sampled_points /= furthest_distance
    return sampled_points

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_adj_normalized(graph, sampled_ids):
    """ read adj info from "graph" dict, and convert to adj normlized array
        Args:
            graph: dict, storing all the adj info for all points
            sampled_ids: target points from all points
        Returns:
            adj_normalized: normlized adj matrix for target points
    """  
    sampled_graph = {i: graph[i] for i in sampled_ids}
    sampled_graph_new = {}
    # remove unsampled edges
    for key in sampled_graph:
        v = sampled_graph[key]
        v_new = [v[j] for j in range(len(v)) if v[j] in sampled_ids]
        sampled_graph_new.update({key: v_new})
    assert(len(sampled_graph_new) == len(sampled_graph))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(sampled_graph_new))
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized = adj_normalized.toarray()
    
    return adj_normalized

def find_nearest_ids(stldata, tiranglePointDic, anCenter, wanted_tris):
    """ find nearest tirangle points ids for target point
        Args:
            stldata: list, contains each point coords
            tiranglePointDic: dict, key: target point ID, value: target point's adj points IDs
            anCenter: the center point of target point cloud
            wanted_tirs: int, eg. 1024, 2048...
        Returns:
            nearest_ids: list of nearest points' ID
    """  
    dist_list = []
    for i in range(len(stldata)):
        point = stldata[i]
        dist = np.sqrt(np.sum(np.square(point - anCenter)))
        dist_list.append(dist)
    tiranglePointDist = []
    for key in tiranglePointDic:
        value = tiranglePointDic[key]
        dist_sum = dist_list[value[0]] + dist_list[value[1]] + dist_list[value[2]]
        tiranglePointDist.append(dist_sum)
    
    argsort = np.argsort(tiranglePointDist)
    nearest_ids = argsort[: wanted_tris] # 1d array
    return nearest_ids

def get_sampled_labels_by_ids(nearest_ids, triangle_neighbor_dic, tirangle_point_dic, eladata):
    """ find nearest tirangle points ids for target point
        Args:
            nearest_ids: list of nearest points' ID
            tirangle_point_dic: dict, key: target point ID, value: target point's adj points IDs
            eladata: array, labels of all points
        Returns:
            sampled_labels:list, sampled points label
    """  
    sampled_tri = {}
    for i in range(len(nearest_ids)):
        key = nearest_ids[i]
        value = triangle_neighbor_dic[nearest_ids[i]]
        sampled_tri.update({key: value})
    #     print(sampled_tri)
    sampled_points = {}
    sampled_labels = {}
    for key in sampled_tri:
        value = sampled_tri[key]
        point_ids = []
        for tri_id in value:
            point_ids += tirangle_point_dic[tri_id]
        point_ids = list(set(point_ids))
        # 整理labels
        labels = [eladata[i,1] for i in point_ids]
        labels = int(max(labels))
        sampled_labels.update({key: labels})
        sampled_points.update({key: point_ids})
    sampled_labels = list(sampled_labels.values())
    
    return sampled_points, sampled_labels

def get_sampled_coords(sampled_points, point_dic):
    """ get the feature matrix of sampled points
        Args:
            sampled_points: list, contains coords of each sampled points
            point_dic: dict, key-target point ID, value-target point's adj points IDs
        Returns:
            final_matrix:array, feature matrix of sampled points N * (3*6)
    """  
    sampled_coords = {}
    for key in sampled_points:
        value = sampled_points[key]
        coordinates = []
        for point_id in value:
            coordinates += point_dic[point_id]
        # 保证每个N都是18列
        ori_len = len(coordinates)
        if ori_len < 18:
            for i in range(int((18-ori_len)/3)):
                coordinates.extend(coordinates[-3:])
        elif ori_len > 18:
            coordinates = coordinates[:18]

        sampled_coords.update({key: coordinates})
    #     print(sampled_coords)
    final_matrix = np.array(list(sampled_coords.values()))
    
    return final_matrix
