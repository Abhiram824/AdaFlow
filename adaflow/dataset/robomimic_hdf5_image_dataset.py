from robomimic.utils.dataset import SequenceDataset
from robomimic.macros import LANG_EMB_KEY
from adaflow.dataset.base_dataset import BaseImageDataset
from adaflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from threadpoolctl import threadpool_limits
from adaflow.common.pytorch_util import dict_apply

import torch
import numpy as np
from typing import Dict, List

from adaflow.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

from adaflow.dataset.robomimic_replay_image_dataset import normalizer_from_stat


class RobomimicHDF5ImageDataset(SequenceDataset,BaseImageDataset):

    """
    Dataset class for sampling directly from Robomimic HDF5 dataset with image observations.

    Same args as RobomimicReplayImageDataset for now.
    """
    def __init__(self, 
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0 # validation not implemented yet
        ):

        assert not abs_action, "abs_action not supported"
        obs_dict= shape_meta["obs"].copy()

        # sequence dataset does not consider language embeddings as part of obs
        obs_dict.pop(LANG_EMB_KEY, None)
        obs_keys = obs_dict.keys()

        # convert horizon and n_obs_steps to frame_stack and seq_length
        frame_stack = n_obs_steps
        seq_length = horizon - frame_stack + 1
        action_keys = ['actions'] if not abs_action else ["actions_abs"]
        dataset_keys = action_keys

        # normalization will be done outside in the dp codebase, so turn off normalization in sequence dataset
        action_config = {action_keys[0]: {'normalization': None}}
        hdf5_normalize_obs = False

        # assert pad_before and pad_after equal to nobs and action length
        # init sequence dataset with kwargs
        SequenceDataset.__init__(self, 
            hdf5_path=dataset_path, 
            obs_keys=obs_keys,
            action_keys=action_keys,
            dataset_keys=dataset_keys, 
            action_config=action_config,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=True,
            pad_seq_length=True,
            load_next_obs=False,
            hdf5_normalize_obs=hdf5_normalize_obs,
        )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.action_key = action_keys[0]
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_size = self.shape_meta['action']["shape"][0]


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # threadpool_limits(1)

        # super call to get data
        data = SequenceDataset.__getitem__(self, idx)

        # Rest is same as RobomimicReplayImageDataset
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        # print(data["obs"].keys())
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data["obs"][key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
        for key in self.lowdim_keys:
            obs_dict[key] = data["obs"][key][T_slice].astype(np.float32)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['actions'][:, :self.action_size].astype(np.float32))
        }

        return torch_data
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self._get_all_data(self.action_key).astype(np.float32))
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self._get_all_data("obs/"+key).astype(np.float32))

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == LANG_EMB_KEY:
                # don't normalize language embeddings
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        self.close_and_delete_hdf5_handle()
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        return self._get_all_data(self.action_key)[:, :self.action_size]

    def _get_all_data(self, key):
        """
        Get all data for a given key across all episodes.
        """
        if LANG_EMB_KEY in key:
            return np.array(list(self._demo_id_to_demo_lang_emb.values()))
        return np.concatenate([self.hdf5_file["data/{}/{}".format(ep, key)][()] for ep in self.demos], axis=0)
