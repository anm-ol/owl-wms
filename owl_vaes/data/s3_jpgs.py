import boto3
import threading
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import io
import time
from PIL import Image

class RandomizedQueue:
    def __init__(self):
        self.items = []

    def add(self, item):
        idx = random.randint(0, len(self.items))
        self.items.insert(idx, item)

    def pop(self):
        if not self.items:
            return None
        idx = random.randint(0, len(self.items) - 1)
        return self.items.pop(idx)

class S3CoDDataset(IterableDataset):
    def __init__(self, bucket_name="cod-raw-360p-30fs", prefix="depth-and-raw", rank=0, world_size=1, include_flow=False, include_depth=False, target_size=(360, 640)):
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.include_flow = include_flow
        self.include_depth = include_depth
        self.target_size = target_size

        # Queue parameters
        self.max_data = 1000

        # Initialize queue
        self.data_queue = RandomizedQueue()

        # Setup S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
        )

        # List all jpg files (not .tar) in the bucket under the specified prefix
        self.jpg_files = self.list_all_jpgs()
        random.shuffle(self.jpg_files)

        print(self.jpg_files[:10])

        # Partition jpg_files for distributed training
        self.jpg_files = self.jpg_files[self.rank::self.world_size]

        # Start background thread to load data
        self.data_thread = threading.Thread(target=self.background_load_data, daemon=True)
        self.data_thread.start()

    def list_all_jpgs(self):
        """List all jpg files in the bucket under the specified prefix, excluding .depth.jpg and .flow.jpg for main images"""
        jpg_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.jpg') and not key.endswith('.depth.jpg') and not key.endswith('.flow.jpg'):
                        jpg_files.append(key)
        return jpg_files

    def process_image_file(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            if self.target_size is not None:
                image = image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {key}: {e}")
            return None

    def process_depth_file(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data))
            if self.target_size is not None:
                image = image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            # print(f"Error loading depth image {key}: {e}")
            return None

    def process_flow_file(self, key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            if self.target_size is not None:
                image = image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            # print(f"Error loading flow image {key}: {e}")
            return None

    def background_load_data(self):
        idx = 0
        n_files = len(self.jpg_files)
        while True:
            if len(self.data_queue.items) < self.max_data:
                # Randomly sample a jpg file
                key = self.jpg_files[idx % n_files]
                idx += 1

                image = self.process_image_file(key)
                if image is None:
                    continue

                result = [image]

                if self.include_depth:
                    depth_key = key[:-4] + '.depth.jpg'
                    depth_image = self.process_depth_file(depth_key)
                    if depth_image is None:
                        continue
                    result.append(depth_image)

                if self.include_flow:
                    flow_key = key[:-4] + '.flow.jpg'
                    flow_image = self.process_flow_file(flow_key)
                    if flow_image is None:
                        continue
                    result.append(flow_image)

                if len(result) > 1:
                    self.data_queue.add(tuple(result))
                else:
                    self.data_queue.add(result[0])
            else:
                time.sleep(1)

    def __iter__(self):
        while True:
            item = self.data_queue.pop()
            if item is not None:
                yield item
            else:
                time.sleep(0.1)

def collate_fn(batch):
    # Convert PIL images to tensors and normalize to [-1, 1]
    tensors = []
    for img in batch:
        # Convert PIL image to tensor and normalize to [0, 1]
        tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
        # Convert [0, 1] to [-1, 1]
        tensor = (tensor * 2) - 1
        tensors.append(tensor)
    return torch.stack(tensors)

def collate_fn_depth(batch):
    # Each item in batch is (image, depth)
    tensors = []
    for img, depth in batch:
        # Convert RGB image to tensor and normalize to [-1, 1]
        img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor * 2) - 1

        # Convert grayscale depth to tensor and normalize to [-1, 1]
        depth_tensor = torch.FloatTensor(np.array(depth))
        if len(depth_tensor.shape) == 3:
            depth_tensor = depth_tensor[:, :, 0]  # Take first channel if RGB
        depth_tensor = depth_tensor.unsqueeze(0) / 255.0  # Add channel dim
        depth_tensor = (depth_tensor * 2) - 1

        # Concatenate on channel dimension
        combined = torch.cat([img_tensor, depth_tensor], dim=0)
        tensors.append(combined)
    return torch.stack(tensors)

def collate_fn_flow(batch):
    # Each item in batch is (image, flow)
    tensors = []
    for img, flow in batch:
        # Convert RGB image to tensor and normalize to [-1, 1]
        img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor * 2) - 1

        # Convert RGB flow to tensor and normalize to [-1, 1]
        flow_tensor = torch.FloatTensor(np.array(flow)).permute(2, 0, 1) / 255.0
        flow_tensor = (flow_tensor * 2) - 1

        # Concatenate on channel dimension
        combined = torch.cat([img_tensor, flow_tensor], dim=0)
        tensors.append(combined)
    return torch.stack(tensors)

def collate_fn_depth_and_flow(batch):
    # Each item in batch is (image, depth, flow)
    tensors = []
    for img, depth, flow in batch:
        # Convert RGB image to tensor and normalize to [-1, 1]
        img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor * 2) - 1

        # Convert grayscale depth to tensor and normalize to [-1, 1]
        depth_tensor = torch.FloatTensor(np.array(depth))
        if len(depth_tensor.shape) == 3:
            depth_tensor = depth_tensor[:, :, 0]  # Take first channel if RGB
        depth_tensor = depth_tensor.unsqueeze(0) / 255.0  # Add channel dim
        depth_tensor = (depth_tensor * 2) - 1

        # Convert RGB flow to tensor and normalize to [-1, 1]
        flow_tensor = torch.FloatTensor(np.array(flow)).permute(2, 0, 1) / 255.0
        flow_tensor = (flow_tensor * 2) - 1

        # Concatenate on channel dimension
        combined = torch.cat([img_tensor, depth_tensor, flow_tensor], dim=0)
        tensors.append(combined)
    return torch.stack(tensors)

def get_loader(batch_size, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    include_flow = data_kwargs.get('include_flow', False)
    include_depth = data_kwargs.get('include_depth', False)

    ds = S3CoDDataset(rank=rank, world_size=world_size, **data_kwargs)

    if include_flow and include_depth:
        collate = collate_fn_depth_and_flow
    elif include_flow:
        collate = collate_fn_flow
    elif include_depth:
        collate = collate_fn_depth
    else:
        collate = collate_fn

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)

if __name__ == "__main__":
    import time
    loader = get_loader(4, include_depth=True, bucket_name='yt-jpegs', prefix="")

    start = time.time()
    batch = next(iter(loader))
    print(batch.shape)
    exit()
    end = time.time()
    first_time = end - start

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    second_time = end - start

    print(batch.shape)