import os
import os.path as osp
import pickle

import cv2
import copy
import numpy as np
import trimesh
from PIL import Image
import torch
from torch.utils.data import Dataset
from smplx import create

import lib.if_nerf_data_utils as if_nerf_dutils
from lib.if_nerf_data_utils import get_rays, get_bound_2d_mask, get_near_far


class Paths:
    __slots__ = [
        "scan",
        "view",
        "smpl_param",
        "smplx_param",
        "T_normal_F",
        "T_position_F",
        "normal_F",
        "depth_F",
        "depth_F_visual",
        "calib",
        "image",
        "T_normal_F_folder",
        "T_position_F_folder",
        "normal_F_folder",
        "depth_F_folder",
        "depth_F_visual_folder",
        "calib_folder",
        "image_folder",
    ]


class BasePointHumanDataset(Dataset):

    def __init__(self,
                 data_folder,
                 render_folder=None,
                 dataset_type="thuman2",
                 global_scale=None,
                 is_perspective=None,
                 num_views=None,
                 num_rotations=None,
                 H=None,
                 W=None,
                 ratio=None,
                 ):
        self.data_folder = data_folder
        self.dataset_type = dataset_type
        if num_rotations is not None:
            assert 360 % num_rotations == 0
            num_views = num_rotations
        self.global_scale = 1. if global_scale is None else global_scale
        self.is_perspective = True if is_perspective is None else is_perspective
        self.num_views = 36 if num_views is None else num_views
        self.num_rotations = num_rotations
        self.H = 512 if H is None else H
        self.W = 512 if W is None else W
        self.ratio = 1.0 if ratio is None else ratio
        if render_folder is None:
            render_folder = f"{dataset_type}_{self.num_views}views"
        self.render_folder = osp.join(self.data_folder, render_folder)
    
    def get_view_name(self, view_idx):
        if self.num_rotations is not None:
            rotation = view_idx * 360 // self.num_rotations
            return f"{rotation:03d}"
        else:
            return f"{view_idx:03d}"
    
    def get_paths(self, scan_id, view_name=None, view_idx=None):
        paths = Paths()
        if self.dataset_type == "thuman2":
            paths.scan = osp.join(
                self.data_folder, "scans", scan_id, f"{scan_id}.obj")
        elif self.dataset_type == "cape":
            paths.scan = osp.join(
                self.data_folder, "scans", f"{scan_id}.obj")
        paths.smpl_param = osp.join(
            self.data_folder, "smpl", f"{scan_id}.pkl")
        paths.smplx_param = osp.join(
            self.data_folder, "smplx", f"{scan_id}.pkl")
        paths.T_normal_F_folder = osp.join(
            self.render_folder, scan_id, "T_normal_F")
        paths.T_position_F_folder = osp.join(
            self.render_folder, scan_id, "T_position_F")
        paths.normal_F_folder = osp.join(
            self.render_folder, scan_id, "normal_F")
        paths.depth_F_folder = osp.join(
            self.render_folder, scan_id, "depth_F")
        paths.depth_F_visual_folder = osp.join(
            self.render_folder, scan_id, "depth_F")
        paths.calib_folder = osp.join(
            self.render_folder, scan_id, "calib")
        paths.image_folder = osp.join(
            self.render_folder, scan_id, "render")
        if view_name is not None or view_idx is not None:
            view_name = self.get_view_name(view_idx) \
                if view_name is None else view_name
            paths.view = view_name
            paths.T_normal_F = osp.join(
                self.render_folder, scan_id, "T_normal_F", f"{view_name}.png")
            paths.T_position_F = osp.join(
                self.render_folder, scan_id, "T_position_F", f"{view_name}.png")
            paths.normal_F = osp.join(
                self.render_folder, scan_id, "normal_F", f"{view_name}.png")
            paths.depth_F = osp.join(
                self.render_folder, scan_id, "depth_F", f"{view_name}.npy")
            paths.depth_F_visual = osp.join(
                self.render_folder, scan_id, "depth_F", f"{view_name}.png")
            paths.calib = osp.join(
                self.render_folder, scan_id, "calib", f"{view_name}.txt")
            paths.image = osp.join(
                self.render_folder, scan_id, "render", f"{view_name}.png")
        return paths    
    
    def load_file(self, path:str, force_exist=True):
        if force_exist and not os.path.exists(path):
            raise ValueError(f"{path} does not exist.")
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".png") or path.endswith(".jpg"):
            image = Image.open(path).convert("RGB")
            image = np.array(image).astype(np.float32)
            image = image / 255.
            return image
        elif path.endswith(".txt"):
            return np.loadtxt(path)
        elif path.endswith(".npy") or path.endswith(".npz"):
            return np.load(path, allow_pickle=True)
        elif path.endswith(".obj"):
            return trimesh.load_mesh(path, process=False)
        else:
            raise NotImplementedError

class PointHumanDataset(BasePointHumanDataset):
    def __init__(self,
                 data_folder:str,
                 split:str,
                 render_folder: str = None,
                 dataset_type: str = "thuman2",
                 global_scale: int = None,
                 is_perspective: bool = True,
                 num_views: int = None,
                 num_rotations: int = None,
                 big_box: bool = True,
                 H: int = None,
                 W: int = None,
                 ratio: float = None,
                 mask_bkgd: bool = True,
                 white_bkgd: bool = False,
                 body_sample_ratio=0.8,
                 num_ref_views=3,
                 N_rand: int = 1024,
                 **kwargs,
                 ):
        """_summary_

        Args:
            data_folder (_type_): _description_
            split (_type_): _description_
            render_folder (str, optional): _description_. Defaults to None.
            dataset_type (str, optional): _description_. Defaults to "thuman2".
            global_scale (int, optional): _description_. Defaults to None.
            is_perspective (bool, optional): _description_. Defaults to True.
            num_views (int, optional): _description_. Defaults to None.
            num_rotations (int, optional): _description_. Defaults to None.
            big_box (bool, optional): _description_. Defaults to True.
            H (int, optional): _description_. Defaults to None.
            W (int, optional): _description_. Defaults to None.
            ratio (float, optional): _description_. Defaults to None.
            N_rand (int, optional): _description_. Defaults to 1024.
            mask_bkgd (bool, optional): _description_. Defaults to True.
            white_bkgd (bool, optional): _description_. Defaults to False.
            body_sample_ratio (float, optional): _description_. Defaults to 0.9.
            num_ref_views (int, optional): _description_. Defaults to 3.
        """
        super(PointHumanDataset, self).__init__(
            data_folder=data_folder,
            render_folder=render_folder,
            dataset_type=dataset_type,
            global_scale=global_scale,
            is_perspective=is_perspective,
            num_views=num_views,
            num_rotations=num_rotations,
            H=H,
            W=W,
            ratio=ratio,
        )

        self.split = split
        self.smplx_model_folder = "data/smpl_related/models"
        self.big_box = big_box
        self.mask_bkgd = mask_bkgd
        self.white_bkgd = white_bkgd
        self.body_sample_ratio = body_sample_ratio
        self.num_ref_views = num_ref_views
        self.nrays = N_rand
        self.train_view = [0, 1, 2, 3] # Warning! This is just a placeholder. It's not used.

        self.is_train, self.is_test, self.is_infer = False, False, False

        scan_ids = open(osp.join(self.data_folder, f'{self.split}.txt')).readlines()
        scan_ids = [scan_id.strip() for scan_id in scan_ids if scan_id.strip() != '']
        scan_ids = list(sorted(scan_ids))
        scan_ids = [scan_id.split('/')[-1] for scan_id in scan_ids]
        self.scan_ids = scan_ids

        if self.split == "train" or self.split == "val":
            self.initialize_for_train()

        model_init_params = dict(
            gender='male',
            model_type='smplx',
            model_path=self.smplx_model_folder,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_expression=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_transl=False,
            num_pca_comps=12) 
        self.smplx = create(**model_init_params)
        # model_init_params['use_pca'] = False
        model_init_params['flat_hand_mean'] = True
        self.dapose_smplx = create(**model_init_params)

    @classmethod
    def from_config(cls, dataset_cfg, data_split, cfg):
        ''' Creates an instance of the dataset.

        Args:
            dataset_cfg (dict): input configuration.
            data_split (str): data split (`train` or `val`).
        '''
        assert data_split in ['test']
        # test novel view on thuman2
        # dataset_cfg = {
        #     "data_folder": "data/thuman2",
        #     "data_root": "data/thuman2",
        #     "render_folder": "thuman2_perspective_nolight_36views",
        #     "dataset_type": "thuman2",
        #     "split": "test",
        #     "num_rotations": 36,
        # }
        # dataset = cls(**dataset_cfg)
        # dataset.initialize_for_test(test_type="novel_view")

        # # test novel view on zju_mocap
        # dataset_cfg = {
        #     "data_folder": "data/zju_mocap",
        #     "data_root": "data/zju_mocap",
        #     "render_folder": "zju_mocap_9views",
        #     "dataset_type": "zju_mocap",
        #     "split": "test",
        #     "num_views": 9,
        #     "H": 512,
        #     "W": 512,
        #     "ratio": 1.0,
        # }
        # dataset = cls(**dataset_cfg)
        # dataset.initialize_for_test(test_type="novel_view")

        # # # test novel view on h36m
        # dataset_cfg = {
        #     "data_folder": "data/h36m",
        #     "data_root": "data/h36m",
        #     "render_folder": "h36m_4views",
        #     "dataset_type": "h36m",
        #     "split": "test",
        #     "num_views": 4,
        #     "H": 512,
        #     "W": 512,
        #     "ratio": 1.0
        # }
        # dataset = cls(**dataset_cfg)
        # dataset.initialize_for_test(test_type="novel_view")

        # # # test novel pose on zju_mocap
        # dataset_cfg = {
        #     "data_folder": "data/zju_mocap",
        #     "data_root": "data/zju_mocap",
        #     "render_folder": "zju_mocap_9views",
        #     "dataset_type": "zju_mocap",
        #     "split": "test",
        #     "num_views": 9,
        #     "H": 512,
        #     "W": 512,
        #     "ratio": 1.0,
        # }
        # dataset = cls(**dataset_cfg)
        # dataset.initialize_for_test(test_type="novel_pose")

        # # # test novel pose on h36m
        dataset_cfg = {
            "data_folder": "data/h36m",
            "data_root": "data/h36m",
            "render_folder": "h36m_4views",
            "dataset_type": "h36m",
            "split": "test",
            "num_views": 4,
            "H": 512,
            "W": 512,
            "ratio": 1.0
        }
        dataset = cls(**dataset_cfg)
        dataset.initialize_for_test(test_type="novel_pose")

        return dataset
    
    def initialize_for_train(self):
        self.stage = 'train'
        self.is_train = True
        self.is_test = False
        self.is_infer = False
        # meshes = np.load(
        #     osp.join(self.data_folder, 'meshes.npz'), allow_pickle=True)
        # self.meshes = {}
        # for scan_id in tqdm(self.scan_ids, desc="Loading meshes..."):
        #     self.meshes[scan_id] = meshes[scan_id]
        
        self.views = [
            self.get_view_name(view_idx) for view_idx in range(self.num_views)
        ]
        
        assert self.num_views % self.num_ref_views == 0

    def initialize_for_test(self, test_type="novel_view"):
        self.is_train = False
        self.is_test = True
        self.is_infer = False
        self.stage = 'test'
        self.test_type = test_type

        if test_type == "novel_view":
            if self.dataset_type == "thuman2":
                test_ref_views = []
                ref_view_interval = self.num_views // self.num_ref_views
                for i in range(0, ref_view_interval, 3):
                    test_ref_views.append([
                        self.get_view_name(view_idx * ref_view_interval + i)
                        for view_idx in range(self.num_ref_views)
                    ])
                test_tgt_views = [
                    self.get_view_name(view_idx)
                    for view_idx in range(self.num_rotations)
                ]
                self.test_ref_views = test_ref_views
                self.test_tgt_views = test_tgt_views
            elif self.dataset_type == "zju_mocap":
                self.test_ref_views = [[
                    self.get_view_name(view_idx)
                    for view_idx in [0, 1, 2]]]
                self.test_tgt_views = [
                    self.get_view_name(view_idx)
                    for view_idx in [3, 4, 5, 6, 7, 8]]
            elif self.dataset_type == "h36m":
                self.test_ref_views = [[
                    self.get_view_name(view_idx)
                    for view_idx in [0, 1, 2]]]
                self.test_tgt_views = [
                    self.get_view_name(view_idx)
                    for view_idx in [3]]
        elif test_type == "novel_pose":
            if self.dataset_type == "zju_mocap":
                self.test_ref_views = [[
                    self.get_view_name(view_idx)
                    for view_idx in [0, 1, 2]]]
                self.test_tgt_views = [
                    self.get_view_name(view_idx)
                    for view_idx in [3, 4, 5, 6, 7, 8]]
                novel_pose_interval = 3
            elif self.dataset_type == "h36m":
                self.test_ref_views = [[
                    self.get_view_name(view_idx)
                    for view_idx in [0, 1, 2]]]
                self.test_tgt_views = [
                    self.get_view_name(view_idx)
                    for view_idx in [3]]
                novel_pose_interval = 1
            
            ref_scan_ids = []
            tgt_scan_ids = []
            for i in range(len(self.scan_ids)):
                if i + novel_pose_interval < len(self.scan_ids):
                    ref_scan_id = self.scan_ids[i]
                    tgt_scan_id = self.scan_ids[i + novel_pose_interval]
                    if '_'.join(ref_scan_id.split('_')[:-1]) == \
                        '_'.join(tgt_scan_id.split('_')[:-1]):
                        ref_scan_ids.append(ref_scan_id)
                        tgt_scan_ids.append(tgt_scan_id)
            self.ref_scan_ids = ref_scan_ids
            self.tgt_scan_ids = tgt_scan_ids
                
    
    def initialize_for_infer(self):
        pass

    @torch.no_grad()
    def load_smpl(self, smplx_params, dapose=False):
        device = 'cpu'
        smplx_input = {}
        for k in smplx_params:
            smplx_input[k] = torch.as_tensor(
                smplx_params[k], dtype=torch.float32, device=device)[None, ...]
        # Important! We compute rays in canonical space of smplx.
        # Note that global_orient should not influence the position of root joint.
        smplx_input["global_orient"] = torch.zeros_like(smplx_input["global_orient"])
        if dapose:
            smplx_input["left_hand_pose"] = torch.zeros_like(smplx_input["left_hand_pose"])
            smplx_input["right_hand_pose"] = torch.zeros_like(smplx_input["right_hand_pose"])
            smplx_input["jaw_pose"] = torch.zeros_like(smplx_input["jaw_pose"])
            smplx_input["leye_pose"] = torch.zeros_like(smplx_input["leye_pose"])
            smplx_input["reye_pose"] = torch.zeros_like(smplx_input["reye_pose"])
            dapose_body_pose = torch.zeros_like(smplx_input["body_pose"])
            dapose_body_pose[:, 0:3] = torch.tensor(
                [0, 0, np.deg2rad(30.)], dtype=dapose_body_pose.dtype, device=device)
            dapose_body_pose[:, 3:6] = torch.tensor(
                [0, 0, -np.deg2rad(30.)], dtype=dapose_body_pose.dtype, device=device)
            smplx_input["body_pose"] = dapose_body_pose
            smplx_output = self.dapose_smplx(
                **smplx_input,
                return_verts=True,
                return_full_pose=True,
                pose2rot=True,
            )
        else:
            smplx_output = self.smplx(
                **smplx_input,
                return_verts=True,
                return_full_pose=True,
                pose2rot=True,
            )
        root_joint = smplx_output.joints[0, 0].detach().cpu().numpy()
        return root_joint, smplx_output
    
    def load_calibration_of_smplx(
        self, intrinsic, extrinsic, Th, scale, global_orient, root_joint):

        H, W = self.H * self.ratio, self.W * self.ratio
        

        Th = np.array(
            [
                [1, 0, 0, Th[0]],
                [0, 1, 0, Th[1]],
                [0, 0, 1, Th[2]],
                [0, 0, 0, 1,]
                
            ]
        )
        scale = np.array(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1,]
            ]
        )
        back_root_joint = np.array(
            [
                [1., 0., 0., -root_joint[0]],
                [0., 1., 0., -root_joint[1]],
                [0., 0., 1., -root_joint[2]],
                [0., 0., 0., 1.],
                
            ]
        )
        go_root_joint = np.array(
            [
                [1., 0., 0., root_joint[0]],
                [0., 1., 0., root_joint[1]],
                [0., 0., 1., root_joint[2]],
                [0., 0., 0., 1.],
                
            ]
        )
        global_orient_, _ = cv2.Rodrigues(global_orient)
        global_orient = np.eye(4)
        global_orient[:3, :3] = global_orient_
        smplx2scan = Th @ scale @ go_root_joint @ global_orient @ back_root_joint
        calib = intrinsic @ extrinsic @ smplx2scan
        temp = cv2.decomposeProjectionMatrix(calib[:3])
        K, camera_R, camera_t = temp[0], temp[1], temp[2]
        camera_t = camera_t[:3] / camera_t[3:]
        camera_t = -(camera_R @ camera_t)
        camera_t = camera_t.ravel()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = camera_R
        extrinsic[:3, 3] = camera_t

        rot_x_180 = np.eye(4)
        rot_x_180[1, 1] = -1.
        rot_x_180[2, 2] = -1.
        extrinsic = rot_x_180 @ extrinsic
        camera_R = extrinsic[:3, :3]
        camera_t = extrinsic[:3, 3]

        to_pixel = np.eye(3)
        to_pixel[0, 0] = W / 2.
        to_pixel[1, 1] = H / 2.
        to_pixel[0, 2] = W / 2.
        to_pixel[1, 2] = H / 2.
        # to_pixel = np.eye(3)
        # to_pixel[0, 0] = W / 2.
        # to_pixel[1, 1] = -H / 2.
        # to_pixel[0, 2] = -W / 2.
        # to_pixel[1, 2] = -H / 2.
        # to_pixel[2, 2] = -1.
        K = to_pixel @ K

        return calib, K, camera_R, camera_t, smplx2scan


    
    def _prepare_ref_input_for_train(self, scan_id, view):
        # scan_id, rotation = self.get_metadata_from_index(index)
        paths = self.get_paths(scan_id, view)

        # Read rendered image.
        img = self.load_file(paths.image)
        # Read camera calibration parameters and depth image.
        calib = self.load_file(paths.calib)
        extrinsic = calib[:4]
        intrinsic = calib[4:]
        intrinsic[1, 1] = -intrinsic[1, 1]

        mask = img.sum(-1) > 0.

        H, W = int(self.H * self.ratio), int(self.W * self.ratio)
        # Resize the image and depth image.
        if self.ratio != 1.:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
        if self.mask_bkgd:
            if self.white_bkgd:
                img[np.logical_not(mask)] = 1.
            else:
                img[np.logical_not(mask)] = 0.

        # Get the fitted smplx parameters.
        smplx_param = self.load_file(paths.smplx_param)
        smplx_input = {
            'betas': smplx_param['betas'][0],
            'global_orient': smplx_param['global_orient'][0],
            # 'global_orient': np.zeros_like(smplx_param['global_orient'][0]),
            'body_pose': smplx_param['body_pose'][0],
            'left_hand_pose': smplx_param['left_hand_pose'][0],
            'right_hand_pose': smplx_param['right_hand_pose'][0],
            'expression': smplx_param['expression'][0],
            'jaw_pose': smplx_param['jaw_pose'][0],
            'leye_pose': smplx_param['leye_pose'][0],
            'reye_pose': smplx_param['reye_pose'][0],
            # 'scale': smplx_param['scale'][0] * self.global_scale,
            # 'Th': smplx_param['Translation'][0] * self.global_scale,
        }
        global_orient = smplx_param['global_orient'][0]
        scale = float(smplx_param['scale'])
        Th = np.array(smplx_param['translation'])
        root_joint, smplx_output = self.load_smpl(smplx_input)
        calib, K, camera_R, camera_t, smplx2scan = self.load_calibration_of_smplx(
            intrinsic, extrinsic,
            Th * self.global_scale,
            scale * self.global_scale,
            global_orient, root_joint
        )

        smplx_xyzs = smplx_output.vertices[0].detach().cpu().numpy()
        # obtain the original bounds for point sampling
        min_xyz = np.min(smplx_xyzs, axis=0)
        max_xyz = np.max(smplx_xyzs, axis=0)
        if self.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_neubody_batch(
                img, mask.astype(np.uint8), K, camera_R, camera_t[:, None], bounds, self.nrays, self.split, ratio=self.body_sample_ratio)

        ret = {}
        ret["img_all"] = np.transpose(img, axes=(2, 0, 1))
        ret["o_img_all"] = np.transpose(img, axes=(2, 0, 1))
        ret["msk_all"] = mask
        # ret["msk_cihp_all"] = mask_cihp
        ret["K_all"] = K
        ret["R_all"] = camera_R
        ret["T_all"] = camera_t[:, None]
        ret["rgb_all"] = rgb
        ret["ray_o_all"] = ray_o
        ret["ray_d_all"] = ray_d
        ret["near_all"] = near
        ret["far_all"] = far
        ret["mask_at_box_all"] = mask_at_box
        ret["bkgd_msk_all"] = bkgd_msk
        ret["img_ray_d_all"] = img_ray_d
        return ret
    
    def _prepare_tgt_input_for_train(self, scan_id, view):
        paths = self.get_paths(scan_id, view)

        # Read rendered image.
        img = self.load_file(paths.image)
        # Read camera calibration parameters and depth image.
        calib = self.load_file(paths.calib)
        extrinsic = calib[:4]
        intrinsic = calib[4:]
        intrinsic[1, 1] = -intrinsic[1, 1]

        mask = img.sum(-1) > 0.

        H, W = int(self.H * self.ratio), int(self.W * self.ratio)
        # Resize the image and depth image.
        if self.ratio != 1.:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
        if self.mask_bkgd:
            if self.white_bkgd:
                img[np.logical_not(mask)] = 1.
            else:
                img[np.logical_not(mask)] = 0.

        # Get the fitted smplx parameters.
        smplx_param = self.load_file(paths.smplx_param)
        smplx_input = {
            'betas': smplx_param['betas'][0],
            'global_orient': smplx_param['global_orient'][0],
            # 'global_orient': np.zeros_like(smplx_param['global_orient'][0]),
            'body_pose': smplx_param['body_pose'][0],
            'left_hand_pose': smplx_param['left_hand_pose'][0],
            'right_hand_pose': smplx_param['right_hand_pose'][0],
            'expression': smplx_param['expression'][0],
            'jaw_pose': smplx_param['jaw_pose'][0],
            'leye_pose': smplx_param['leye_pose'][0],
            'reye_pose': smplx_param['reye_pose'][0],
            # 'scale': smplx_param['scale'][0] * self.global_scale,
            # 'Th': smplx_param['Translation'][0] * self.global_scale,
        }
        global_orient = smplx_param['global_orient'][0]
        scale = float(smplx_param['scale'])
        Th = np.array(smplx_param['translation'])
        root_joint, smplx_output = self.load_smpl(smplx_input)
        calib, K, camera_R, camera_t, smplx2scan = self.load_calibration_of_smplx(
            intrinsic, extrinsic,
            Th * self.global_scale,
            scale * self.global_scale,
            global_orient, root_joint
        )

        # Get the fitted smplx mesh.
        smplx_xyzs = smplx_output.vertices[0].detach().cpu().numpy()
        vertices = smplx_xyzs.copy()
        feature = vertices.copy()

        # obtain the original bounds for point sampling
        min_xyz = np.min(smplx_xyzs, axis=0)
        max_xyz = np.max(smplx_xyzs, axis=0)
        if self.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        # construct the coordinate in smpl space
        xyz = smplx_xyzs
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_neubody_batch(
                img, mask.astype(np.uint8), K, camera_R, camera_t[:, None], bounds, self.nrays, self.split, ratio=self.body_sample_ratio)

        _, dapose_smplx_output = self.load_smpl(smplx_input, dapose=True)
        t_vertices = dapose_smplx_output.vertices[0].detach().cpu().numpy()
        t_feature = t_vertices.copy()

        xyz = t_vertices
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        t_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        t_coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        t_out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        t_out_sh = (t_out_sh | (x - 1)) + 1
        

        # bounds_2d = np.array([
        #     [bounds[0, 0], bounds[0, 1], bounds[0, 2]],
        #     [bounds[0, 0], bounds[0, 1], bounds[1, 2]],
        #     [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
        #     [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
        #     [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
        #     [bounds[1, 0], bounds[0, 1], bounds[1, 2]],
        #     [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
        #     [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
        # ])
        # bounds_2d = bounds_2d @ camera_R.T + camera_t
        # bounds_2d = bounds_2d @ K.T
        # bounds_2d[..., :2] /= bounds_2d[..., 2:]

        # joints = smplx_output.joints[0, :55].detach().cpu().numpy()
        # joints_2d = joints @ camera_R.T + camera_t
        # joints_2d = joints_2d @ K.T
        # joints_2d[..., :2] /= joints_2d[..., 2:]


        ret = {}

        ret["o_img_all"] = np.transpose(img, axes=(2, 0, 1))
        ret["msk_all"] = mask
        # ret["msk_cihp_all"] = mask_cihp

        ret["rgb_all"] = rgb
        ret["ray_o_all"] = ray_o
        ret["ray_d_all"] = ray_d
        ret["near_all"] = near
        ret["far_all"] = far
        ret["mask_at_box_all"] = mask_at_box
        ret["bkgd_msk_all"] = bkgd_msk

        params = smplx_param
        params['global_orient'] = np.zeros_like(params['global_orient'])
        params['Th'] = np.zeros((1, 3)).astype(np.float32)
        params['R'] = np.eye(3).astype(np.float32)
        params['Rh'] = np.zeros((1, 3)).astype(np.float32)
        params['shapes'] = params['betas']
        del params['scale']
        del params['translation']
        ret['params'] = params

        ret["vertices"] = vertices
        ret["feature"] = feature
        ret["bounds"] = bounds
        ret["coord"] = coord
        ret["out_sh"] = out_sh

        ret["t_vertices"] = t_vertices
        ret["t_feature"] = t_feature
        ret["t_bounds"] = t_bounds
        ret["t_coord"] = t_coord
        ret["t_out_sh"] = t_out_sh
        return ret
    
    def concat_inputs(self, inputs):
        ret = {}
        for key in inputs[0]:
            ret[key] = np.stack(
                [inp[key] for inp in inputs], axis=0)
        return ret
    
    def add_prefix_to_key(self, d, prefix):
        ret = {}
        for key in d:
            ret[prefix + key] = d[key]
        return ret
    
    def get_metadata_from_index(self, index):
        if self.is_train:
            scan_idx = index // len(self.scan_ids)
            scan_id = self.scan_ids[scan_idx]
            tgt_view = self.views[index % len(self.views)]
            ref_views = np.random.choice(
                self.views,
                self.num_ref_views,
                replace=False
            )
            ref_scan_id = scan_id
            tgt_scan_id = scan_id
        elif self.is_test:
            if self.test_type == "novel_view":
                num_test_ref_views = len(self.test_ref_views)
                num_test_tgt_views = len(self.test_tgt_views)

                tgt_view_idx = index % num_test_tgt_views
                tgt_view = self.test_tgt_views[tgt_view_idx]

                ref_views_idx = (index // num_test_tgt_views) % num_test_ref_views
                ref_views = self.test_ref_views[ref_views_idx]

                scan_idx = index // (num_test_ref_views * num_test_tgt_views)
                scan_id = self.scan_ids[scan_idx]

                ref_scan_id = scan_id
                tgt_scan_id = scan_id

            elif self.test_type == 'novel_pose':

                num_test_ref_views = len(self.test_ref_views)
                num_test_tgt_views = len(self.test_tgt_views)

                tgt_view_idx = index % num_test_tgt_views
                tgt_view = self.test_tgt_views[tgt_view_idx]

                ref_views_idx = (index // num_test_tgt_views) % num_test_ref_views
                ref_views = self.test_ref_views[ref_views_idx]

                scan_idx = index // (num_test_ref_views * num_test_tgt_views)
                ref_scan_id = self.ref_scan_ids[scan_idx]
                tgt_scan_id = self.tgt_scan_ids[scan_idx]


        else:
            raise NotImplementedError
        return ref_scan_id, tgt_scan_id, ref_views, tgt_view

    def __getitem__(self, index):
        torch.set_default_tensor_type(torch.FloatTensor)
        ref_scan_id, tgt_scan_id, ref_views, tgt_view = self.get_metadata_from_index(index)
        ret_ref = []
        for ref_view in ref_views:
            ret = self._prepare_ref_input_for_train(ref_scan_id, ref_view)
            ret_ref.append(ret)
        ret_ref = self.concat_inputs(ret_ref)
        # Get target view.
        ret_tgt = self._prepare_tgt_input_for_train(tgt_scan_id, tgt_view)
        # ret_tgt = self.add_prefix_to_key(ret_tgt, "tgt_")

        # images = np.concatenate(
        #     [ret_tgt["images"][None, ...], ret_ref["images"]], axis=0)
        # images_masks = np.concatenate(
        #     [ret_tgt["images_masks"][None, ...], ret_ref["images_masks"]], axis=0)
        # K = np.concatenate(
        #     [ret_tgt["K"][None, ...], ret_ref["K"]], axis=0)
        # Rt = np.concatenate(
        #     [ret_tgt["Rt"][None, ...], ret_ref["Rt"]], axis=0)

        ret = {}
        ret["pose_index"] = 0
        ret["instance_index"] = 0
        ret["gender"] = 1
        ret["R"] = np.eye(3).astype(np.float32)
        ret["Th"] = np.zeros(3).astype(np.float32)

        ret["params"] = ret_tgt["params"]
        ret["vertices"] = ret_tgt["vertices"]
        ret["feature"] = ret_tgt["feature"]
        ret["bounds"] = ret_tgt["bounds"]
        ret["coord"] = ret_tgt["coord"]
        ret["out_sh"] = ret_tgt["out_sh"]
        ret["t_vertices"] = ret_tgt["t_vertices"]
        ret["t_feature"] = ret_tgt["t_feature"]
        ret["t_bounds"] = ret_tgt["t_bounds"]
        ret["t_coord"] = ret_tgt["t_coord"]
        ret["t_out_sh"] = ret_tgt["t_out_sh"]
        
        ret["img_all"] = ret_ref["img_all"]
        ret["img_ray_d_all"] = ret_ref["img_ray_d_all"]
        ret["K_all"] = ret_ref["K_all"]
        ret["R_all"] = ret_ref["R_all"]
        ret["T_all"] = ret_ref["T_all"]

        ret["o_img_all"] = np.concatenate(
            [ret_tgt["o_img_all"][None], ret_ref["o_img_all"]], axis=0)
        ret["msk_all"] = np.concatenate(
            [ret_tgt["msk_all"][None], ret_ref["msk_all"]], axis=0)
        ret["rgb_all"] = np.concatenate(
            [ret_tgt["rgb_all"][None], ret_ref["rgb_all"]], axis=0)
        ret["ray_o_all"] = np.concatenate(
            [ret_tgt["ray_o_all"][None], ret_ref["ray_o_all"]], axis=0)
        ret["ray_d_all"] = np.concatenate(
            [ret_tgt["ray_d_all"][None], ret_ref["ray_d_all"]], axis=0)
        ret["near_all"] = np.concatenate(
            [ret_tgt["near_all"][None], ret_ref["near_all"]], axis=0)
        ret["far_all"] = np.concatenate(
            [ret_tgt["far_all"][None], ret_ref["far_all"]], axis=0)
        ret["mask_at_box_all"] = np.concatenate(
            [ret_tgt["mask_at_box_all"][None], ret_ref["mask_at_box_all"]], axis=0)
        ret["bkgd_msk_all"] = np.concatenate(
            [ret_tgt["bkgd_msk_all"][None], ret_ref["bkgd_msk_all"]], axis=0)

        if ref_scan_id == tgt_scan_id:
            ret["scan_id"] = ref_scan_id
        else:
            ret["scan_id"] = '-'.join([ref_scan_id, tgt_scan_id])
        ret["ref_views"] = '_'.join(ref_views)
        ret["tgt_view"] = tgt_view

        # for key in ret:
        #     if isinstance(ret[key], np.ndarray):
        #         ret[key] = ret[key].astype(np.float32)
        #     elif isinstance(ret[key], torch.Tensor):
        #         ret[key] = ret[key].float()
        #     elif isinstance(ret[key], float):
        #         ret[key] = np.float32(ret[key])
        return ret

    def __len__(self):
        if self.is_train:
            return int(len(self.scan_ids) * self.num_views)
        elif self.is_test:
            if self.test_type == "novel_view":
                num_scans = len(self.scan_ids)
                num_test_ref_views = len(self.test_ref_views)
                num_test_tgt_views = len(self.test_tgt_views)
                return int(num_scans * num_test_ref_views * num_test_tgt_views)
            elif self.test_type == "novel_pose":
                num_scans = len(self.ref_scan_ids)
                num_test_ref_views = len(self.test_ref_views)
                num_test_tgt_views = len(self.test_tgt_views)
                return int(num_scans * num_test_ref_views * num_test_tgt_views)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    dataset = PointHumanDataset(
        data_folder="data/thuman2",
        render_folder="thuman2_perspective_nolight_36views",
        dataset_type="thuman2",
        split="train",
    )
    data = dataset[1]
    for k in data:
        if isinstance(data[k], torch.Tensor):
            print(k, data[k].shape)
        elif isinstance(data[k], np.ndarray):
            print(k, data[k].shape)
        else:
            print(k, data[k])
