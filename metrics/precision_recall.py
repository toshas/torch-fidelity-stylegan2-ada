# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Precision/Recall (PR) from the paper "Improved Precision and Recall
Metric for Assessing Generative Models". Matches the original implementation
by Kynkaanniemi et al. at
https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py"""
import copy

import torch

import dnnlib
from . import metric_utils

#----------------------------------------------------------------------------

def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_pr(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    detector_kwargs = dict(return_features=True)

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float16).to(opts.device)

    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
    return results['precision'], results['recall']

#----------------------------------------------------------------------------

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, max_items):
        self.dataset = dataset
        self.real_len = min(len(dataset), max_items)

    def __len__(self):
        return self.real_len

    def __getitem__(self, i):
        assert i < self.real_len
        item = self.dataset[i]
        item = torch.from_numpy(item[0])
        return item


def compute_pr_fidelity(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):

    from torch_fidelity import calculate_metrics, KEY_METRIC_PRECISION, KEY_METRIC_RECALL, KEY_METRIC_F_SCORE, GenerativeModelBase

    class G_wrapper(GenerativeModelBase):
        def __init__(self, G, G_kwargs, crop, coerce_fakes_dtype):
            super(G_wrapper, self).__init__()
            self.G = copy.deepcopy(G)
            self.G_kwargs = G_kwargs
            self.crop = crop
            self.coerce_fakes_dtype = coerce_fakes_dtype

        def forward(self, z, c):
            if c.dim() == 1:
                c = torch.nn.functional.one_hot(c, self.G.c_dim)

            w = self.G.mapping(z=z, c=c)

            # Randomize noise buffers.
            for name, buf in self.G.named_buffers():
                if name.endswith('.noise_const'):
                    buf.copy_(torch.randn_like(buf))

            # Generate images.
            img = self.G.synthesis(ws=w, noise_mode='const', force_fp32=True, **self.G_kwargs)

            # Center crop.
            if self.crop:
                assert img.shape[2] == img.shape[3]
                c = img.shape[2] // 8
                img = img[:, :, c * 3: c * 7, c * 2: c * 6]

            # Downsample to 256x256 if larger (smaller, such as cifar-10, will stay as is)
            factor = self.G.img_resolution // 256
            if factor > 1:
                img = img.reshape(
                    [-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

            # Scale dynamic range from [-1,1] to [0,255].
            img = (img + 1) * (255 / 2)
            if self.G.img_channels == 1:
                img = img.repeat([1, 3, 1, 1])

            # coerce dtyoe to uint8 (so the predictions are judged exactly how they would be read from an image file)
            if self.coerce_fakes_dtype:
                img = img.clamp(0., 255.)
                img = img.to(torch.uint8)

            return img

        @property
        def z_size(self):
            return self.G.z_dim

        @property
        def z_type(self):
            return 'normal'

        @property
        def num_classes(self):
            return self.G.c_dim

    assert row_batch_size == col_batch_size

    if opts.rank > 0:
        # keeps torch-fidelity busy on one gpu with rank=0
        return float('nan')

    g_wrapper = G_wrapper(opts.G, opts.G_kwargs, crop=False, coerce_fakes_dtype=True)
    g_wrapper.eval().requires_grad_(False).to(opts.device)

    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    dataset = DatasetWrapper(dataset, max_real)

    out = calculate_metrics(
        prc=True,
        input1=dataset,
        input2=g_wrapper,
        input2_model_z_type='normal',
        input2_model_z_size=opts.G.z_dim,
        input2_model_num_classes=opts.G.c_dim,
        input2_model_num_samples=num_gen,
        prc_neighborhood=nhood_size,
        prc_batch_size=col_batch_size,
    )

    return out[KEY_METRIC_PRECISION], out[KEY_METRIC_RECALL]

#----------------------------------------------------------------------------
