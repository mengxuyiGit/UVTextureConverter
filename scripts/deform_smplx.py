import os
import smplx
import trimesh
import pickle
import torch
from glob import glob
import numpy as np


model_init_params = dict(
    gender='male',
    model_type='smplx',
    model_path='/home/liu-compute/Data',
    # model_path=SMPLX().model_dir,
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

smpl_model = smplx.create(**model_init_params)

# pickle_folder = '/home/david/Datasets/THuman2.0/THUman2.0__Smpl-X/'
# pickle_files = glob(os.path.join(pickle_folder, '*/*.pkl'))
# pickle_files = ["/home/liu-compute/Repo/UVTextureConverter/test_data/0000_smplx_pkl/smplx_param.pkl"]
pickle_files = ["/home/liu-compute/Repo/UVTextureConverter/test_data/0480_smplx_pkl/smplx_param.pkl"]

for pickle_filename in pickle_files:
    
    param = np.load(pickle_filename, allow_pickle=True)
    for key in param.keys():
        param[key] = torch.as_tensor(param[key]).to(torch.float32)

    model_forward_params = dict(betas=param['betas'],
                                global_orient=param['global_orient'],
                                body_pose=param['body_pose'],
                                left_hand_pose=param['left_hand_pose'],
                                right_hand_pose=param['right_hand_pose'],
                                jaw_pose=param['jaw_pose'],
                                leye_pose=param['leye_pose'],
                                reye_pose=param['reye_pose'],
                                expression=param['expression'],
                                return_verts=True)

    smpl_out = smpl_model(**model_forward_params)

    smpl_verts = (
        (smpl_out.vertices[0] * param['scale'] + param['translation'])).detach()
    
    smpl_verts /= param['scale']
    
    # print("scale: ", param['scale'])
    # smpl_verts = (smpl_out.vertices[0] + param['translation'] * 2).detach()
    smpl_mesh = trimesh.Trimesh(smpl_verts,
                                smpl_model.faces,
                                process=False, 
                                maintain_order=True)
    
    mesh_fname = pickle_filename.replace('.pkl', '_myvis_scaled_back.obj')
    smpl_mesh.export(mesh_fname)