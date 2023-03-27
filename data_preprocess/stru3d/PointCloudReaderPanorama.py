import cv2
import open3d as o3d
import os
from sklearn.preprocessing import normalize
import json
import matplotlib.pyplot as plt
import numpy as np

NUM_SECTIONS = -1

class PointCloudReaderPanorama():

    def __init__(self, path, resolution="full", random_level=0, generate_color=False, generate_normal=False):
        self.path = path
        self.random_level = random_level
        self.resolution = resolution
        self.generate_color = generate_color
        self.generate_normal = generate_normal
        sections = [p for p in os.listdir(os.path.join(path, "2D_rendering"))]
        self.depth_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", self.resolution, "depth.png"]) for p in sections]
        self.rgb_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", self.resolution, "rgb_coldlight.png"]) for p in sections]
        self.normal_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", self.resolution, "normal.png"]) for p in sections]
        self.camera_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", "camera_xyz.txt"]) for p in sections]
        self.camera_centers = self.read_camera_center()
        self.point_cloud = self.generate_point_cloud(self.random_level, color=self.generate_color, normal=self.generate_normal)

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            with open(self.camera_paths[i], 'r') as f:
                line = f.readline()
            center = list(map(float, line.strip().split(" ")))
            camera_centers.append(np.asarray([center[0], center[1], center[2]]))
        return camera_centers

    def generate_point_cloud(self, random_level=0, color=False, normal=False):
        coords = []
        colors = []
        normals = []
        points = {}

        # Getting Coordinates
        for i in range(len(self.depth_paths)):
            depth_img = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            x_tick = 180.0/depth_img.shape[0]
            y_tick = 360.0/depth_img.shape[1]

            rgb_img = cv2.imread(self.rgb_paths[i])
            rgb_img = cv2.cvtColor(rgb_img, code=cv2.COLOR_BGR2RGB)
            normal_img = cv2.imread(self.normal_paths[i])

            for x in range(0, depth_img.shape[0]):
                for y in range(0, depth_img.shape[1]):
                    # need 90 - -09
                    alpha = 90 - (x * x_tick)
                    beta = y * y_tick -180

                    depth = depth_img[x,y] + np.random.random()*random_level

                    if depth > 500.:
                        z_offset = depth*np.sin(np.deg2rad(alpha))
                        xy_offset = depth*np.cos(np.deg2rad(alpha))
                        x_offset = xy_offset * np.sin(np.deg2rad(beta))
                        y_offset = xy_offset * np.cos(np.deg2rad(beta))
                        point = np.asarray([x_offset, y_offset, z_offset])
                        coords.append(point + self.camera_centers[i])
                        colors.append(rgb_img[x, y])
                        # normals.append(normalize(normal_img[x, y].reshape(-1, 1)).ravel())
            # break

        coords = np.asarray(coords)
        colors = np.asarray(colors) / 255.0
        # normals = np.asarray(normals)

        coords[:,:2] = np.round(coords[:,:2] / 10) * 10.
        coords[:,2] = np.round(coords[:,2] / 100) * 100.
        unique_coords, unique_ind = np.unique(coords, return_index=True, axis=0)

        coords = coords[unique_ind]
        colors = colors[unique_ind]
        # normals = normals[unique_ind]


        points['coords'] = coords
        points['colors'] = colors
        # points['normals'] = normals

        # if color:
        #     # Getting RGB color
        #     for i in range(len(self.rgb_paths)):
        #         rgb_img = cv2.imread(self.rgb_paths[i])
        #         rgb_img = cv2.cvtColor(rgb_img, code=cv2.COLOR_BGR2RGB)
        #         for x in range(0, rgb_img.shape[0], 2):
        #             for y in range(0, rgb_img.shape[1], 2):
        #                 colors.append(rgb_img[x, y])
        #     points['colors'] = np.asarray(colors)/255.0
        # if normal:
        #     # Getting Normal
        #     for i in range(len(self.normal_paths)):
        #         normal_img = cv2.imread(self.normal_paths[i])
        #         for x in range(0, normal_img.shape[0], 2):
        #             for y in range(0, normal_img.shape[1], 2):
        #                 normals.append(normalize(normal_img[x, y].reshape(-1, 1)).ravel())
        #     points['normals'] = normals

        print("Pointcloud size:", points['coords'].shape[0])
        return points

    def get_point_cloud(self):
        return self.point_cloud

    def generate_density(self, width=256, height=256):

        ps = self.point_cloud["coords"] * -1
        ps[:,0] *= -1
        ps[:,1] *= -1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ps)
        pcd.estimate_normals()

        # zs = np.round(ps[:,2] / 100) * 100
        # zs, zs_ind = np.unique(zs, return_index=True, axis=0)
        # ps_ind = ps[:, :2] ==
        # print("Generate density...")

        image_res = np.array((width, height))

        max_coords = np.max(ps, axis=0)
        min_coords = np.min(ps, axis=0)
        max_m_min = max_coords - min_coords

        max_coords = max_coords + 0.1 * max_m_min
        min_coords = min_coords - 0.1 * max_m_min

        normalization_dict = {}
        normalization_dict["min_coords"] = min_coords
        normalization_dict["max_coords"] = max_coords
        normalization_dict["image_res"] = image_res


        # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
        coordinates = \
            np.round(
                (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
        coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                    image_res - 1)

        density = np.zeros((height, width), dtype=np.float32)

        unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
        # print(np.unique(counts))
        # counts = np.minimum(counts, 1e2)

        unique_coordinates = unique_coordinates.astype(np.int32)

        density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
        density = density / np.max(density)
        # print(np.unique(density))

        normals = np.array(pcd.normals)
        normals_map = np.zeros((density.shape[0], density.shape[1], 3))

        import time
        start_time = time.time()
        for i, unique_coord in enumerate(unique_coordinates):
            # print(normals[unique_ind])
            normals_indcs = np.argwhere(np.all(coordinates[::10] == unique_coord, axis=1))[:,0]
            normals_map[unique_coordinates[i, 1], unique_coordinates[i, 0], :] = np.mean(normals[::10][normals_indcs, :], axis=0)

        print("Time for normals: ", time.time() - start_time)

        normals_map = (np.clip(normals_map,0,1) * 255).astype(np.uint8)

        # plt.figure()
        # plt.imshow(normals_map)
        # plt.show()

        return density, normals_map, normalization_dict

    def visualize(self, export_path=None):
        pcd = o3d.geometry.PointCloud()

        points = self.point_cloud['coords']

        print(np.max(points, axis=0))
        indices = np.where(points[:, 2] < 2000)

        points = points[indices]
        points[:,1] *= -1
        points[:,:] /= 1000
        pcd.points = o3d.utility.Vector3dVector(points)

        if self.generate_normal:
            normals = self.point_cloud['normals']
            normals = normals[indices]
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if self.generate_color:
            colors = self.point_cloud['colors']
            colors = colors[indices]
            pcd.colors = o3d.utility.Vector3dVector(colors)


        with open("/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/Datasets/Structured_3D_dataset/Structured3D/Structured3D_0/Structured3D/train/scene_00015/annotation_3d.json") as file:
            annos = json.load(file)



        # wireframe_geo_list = visualize_wireframe(annos, vis=False, ret=True)
        # o3d.visualization.draw_geometries([pcd] + wireframe_geo_list)
        # o3d.visualization.draw_geometries([pcd])

        pcd.estimate_normals()

        # radii = 0.01
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

        # alpha = 0.1
        # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)

        o3d.visualization.draw_geometries([pcd])

        if export_path is not None:

            o3d.io.write_point_cloud(export_path, pcd)

        # o3d.visualization.draw_geometries([pcd])

    def export_ply(self, path):
        '''
        ply
        format ascii 1.0
        comment Mars model by Paul Bourke
        element vertex 259200
        property float x
        property float y
        property float z
        property uchar r
        property uchar g
        property uchar b
        property float nx
        property float ny
        property float nz
        end_header
        '''
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %d\n" % self.point_cloud['coords'].shape[0])
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if self.generate_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            if self.generate_normal:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            f.write("end_header\n")
            for i in range(self.point_cloud['coords'].shape[0]):
                normal = []
                color = []
                coord = self.point_cloud['coords'][i].tolist()
                if self.generate_color:
                    color = list(map(int, (self.point_cloud['colors'][i]*255).tolist()))
                if self.generate_normal:
                    normal = self.point_cloud['normals'][i].tolist()
                data = coord + color + normal
                f.write(" ".join(list(map(str,data)))+'\n')
