import numpy as np
import random

class GeneratorPointCloud():
    
    def __init__(self, rotate, jitter, random_scale, scale_low, scale_high, rotate_perturbation, shift):
        self.rotate = rotate
        self.jitter = jitter
        self.random_scale = random_scale
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.rotate_perturbation = rotate_perturbation
        self.shift = shift
    
    def flow(self, data, labels, batch_size, seed = 0):
        if self.rotate != False:
            data = self.rotate_point_cloud(data, self.rotate)
        if self.shift != False:
            data = self.shift_point_cloud(data, self.shift)
        if self.random_scale == True:
            data = self.random_scale_point_cloud(data, self.scale_low, self.scale_high)
        if self.jitter == True:
            data = self.jitter_point_cloud(data)
        if self.rotate_perturbation == True:
            data = self.rotate_perturbation_point_cloud(data)
            
        random.seed(seed)
        while True:
            idx = random.sample(range(0, len(data)), batch_size)
            batch_data = data[idx]
            batch_labels = labels[idx]
            yield(batch_data, batch_labels)
        
        
    def rotate_point_cloud(self, data, rotation_angle = False):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(data.shape, dtype=np.float32)
        for k in range(data.shape[0]):
            if rotation_angle == False:
                rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        B, N, C = data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += data
        return jittered_data

    def random_scale_point_cloud(self, data, scale_low=0.8, scale_high=1.25):
        """ Randomly scale the point cloud. Scale is per point cloud.
           Input:
              BxNx3 array, original batch of point clouds
           Return:
              BxNx3 array, scaled batch of point clouds
        """
        B, N, C = data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            data[batch_index,:,:] *= scales[batch_index]
        return data

    def rotate_perturbation_point_cloud(self, data, angle_sigma=0.06, angle_clip=0.18):
        """ Randomly perturb the point clouds by small rotations
           Input:
           BxNx3 array, original batch of point clouds
           Return:
           BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(data.shape, dtype=np.float32)
        for k in range(data.shape[0]):
            angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array([[1,0,0],
                     [0,np.cos(angles[0]),-np.sin(angles[0])],
                     [0,np.sin(angles[0]),np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                     [0,1,0],
                     [-np.sin(angles[1]),0,np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                     [np.sin(angles[2]),np.cos(angles[2]),0],
                     [0,0,1]])
            R = np.dot(Rz, np.dot(Ry,Rx))
            shape_pc = data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return rotated_data

    def shift_point_cloud(self, data, shift_range=0.1):
        """ Randomly shift point cloud. Shift is per point cloud.
           Input:
           BxNx3 array, original batch of point clouds
           Return:
           BxNx3 array, shifted batch of point clouds
        """
        B, N, C = data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B,3))
        for batch_index in range(B):
            data[batch_index,:,:] += shifts[batch_index,:]
        return data
