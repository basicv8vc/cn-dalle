from tqdm.notebook import tqdm

import torchvision.transforms as T

import webdataset as wds

import jax
import braceexpand
from pathlib import Path

Google_Cloud_Storage_Bucket_URL = "webdatasets/wukong-{000000..001071}.tar"
shards = Google_Cloud_Storage_Bucket_URL
encoded_output = Path("encoded_data")  # where we will save our encoded data

VQGAN_REPO, VQGAN_COMMIT_ID = (
    "dalle-mini/vqgan_imagenet_f16_16384",
    "85eb5d3b51a1c62a0cc8f4ccdee9882c0d0bd384",
)

# defaults for a TPU v2-8, 128 is large
batch_size = 64  #  128  # Per device
num_workers = 8  # For parallel processing
total_bs = batch_size * jax.device_count()  # You can use a smaller size while testing
save_frequency = 128  # Number of batches to create a new file (180MB for f16 and 720MB for f8 per file)

print("total_bs: {}".format(total_bs))
shards = list(
    braceexpand.braceexpand(shards)
)  # better display for tqdm with known length


ds = (
    wds.WebDataset(shards, handler=wds.warn_and_continue)
    .decode("rgb", handler=wds.warn_and_continue)
    .to_tuple("jpg", "json")  # assumes image is in `jpg` and caption in `txt`
    .batched(total_bs)  # load in batch per worker (faster)
)


# images, captions = next(iter(ds))
dl = (
    wds.WebLoader(ds, batch_size=None, num_workers=8).unbatched().batched(total_bs)
)  # avoid partial batch at the end of each worker


from vqgan_jax.modeling_flax_vqgan import VQModel
from flax.jax_utils import replicate

# vqgan = VQModel.from_pretrained("flax-community/vqgan_f16_16384")
vqgan = VQModel.from_pretrained(VQGAN_REPO)
vqgan_params = replicate(vqgan.params)
print(type(vqgan_params))  # dict
# import json
# json.dump(vqgan_params, open("kkk.dt", "w"))

from collections.abc import MutableMapping

def flatten_dict(d, parent_key  = '', sep ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
flatten_params = flatten_dict(vqgan_params)
for k, v in flatten_params.items():
    print(k, type(v))

from flax.training.common_utils import shard
from functools import partial


@partial(jax.pmap, axis_name="batch")
def p_encode(batch, params):
    # Not sure if we should `replicate` params, does not seem to have any effect
    _, indices = vqgan.encode(batch, params=params)
    return indices

import pandas as pd


def encode_dataset(dataloader, output_dir, save_frequency):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_captions = []
    all_encoding = []
    all_urls = []
    n_file = 1
    for idx, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.numpy()
        n = len(images) // 8 * 8
        if n != len(images):
            # get the max number of images we can (multiple of 8)
            print(f"Different sizes {n} vs {len(images)}")
            images = images[:n]
            captions = captions[:n]
        if not len(captions):
            print(f"No images/captions in batch...")
            continue
        images = shard(images)
        encoded = p_encode(images, vqgan_params)
        encoded = encoded.reshape(-1, encoded.shape[-1])
        all_captions.extend([cap["caption"] for cap in captions])
        all_urls.extend([cap["url"] for cap in captions])
        all_encoding.extend(encoded.tolist())

        # save files
        if (idx + 1) % save_frequency == 0:
            print(f"Saving file {n_file}")
            batch_df = pd.DataFrame.from_dict(
                {"caption": all_captions, "encoding": all_encoding, "url":all_urls}
            )
            batch_df.to_parquet(f"{output_dir}/{n_file:03d}.parquet")
            all_captions = []
            all_encoding = []
            all_urls = []
            n_file += 1

    if len(all_captions):
        print(f"Saving final file {n_file}")
        batch_df = pd.DataFrame.from_dict(
            {"caption": all_captions, "encoding": all_encoding, "url": all_urls}
        )
        batch_df.to_parquet(f"{output_dir}/{n_file:03d}.parquet")
encode_dataset(dl, output_dir=encoded_output, save_frequency=save_frequency)
