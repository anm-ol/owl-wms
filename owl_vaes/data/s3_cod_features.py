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
import tarfile
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
    def __init__(self, bucket_name="cod-raw-360p-30fs", prefix = "depth-and-raw", rank=0, world_size=1, include_flow=False, include_depth=False, target_size=(360,640)):
        super().__init__()
        
        self.rank = rank
        self.world_size = world_size
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.include_flow = include_flow
        self.include_depth = include_depth
        self.target_size = target_size

        # Queue parameters
        self.max_tars = 2
        self.max_data = 1000

        # Initialize queues
        self.tar_queue = RandomizedQueue()
        self.data_queue = RandomizedQueue()

        # Setup S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
        )

        self.tar_files = self.list_all_tars()
        random.shuffle(self.tar_files)
        
        # Start background threads
        self.tar_thread = threading.Thread(target=self.background_download_tars, daemon=True)
        self.data_thread = threading.Thread(target=self.background_load_data, daemon=True)
        self.tar_thread.start()
        self.data_thread.start()

    def list_all_tars(self):
        """List all tar files in the bucket under the specified prefix"""
        tar_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.tar'):
                        tar_files.append(obj['Key'])
        return tar_files

    def random_sample_prefix(self):
        """Randomly select a tar file from the available tar files"""
        return random.choice(self.tar_files)

    def background_download_tars(self):
        while True:
            if len(self.tar_queue.items) < self.max_tars:
                tar_path = self.random_sample_prefix()
                try:
                    # Download tar directly to memory
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=tar_path)
                    tar_data = response['Body'].read()
                    self.tar_queue.add(tar_data)
                except Exception as e:
                    print(f"Error downloading tar {tar_path}: {e}")
            else:
                time.sleep(1)

    def process_image_file(self, tar, image_name):
        try:
            f = tar.extractfile(image_name)
            if f is not None:
                image_data = f.read()
                image = Image.open(io.BytesIO(image_data))
                if self.target_size is not None:
                    image = image.resize((self.target_size[1],self.target_size[0]), Image.Resampling.LANCZOS)
                return image
        except:
            return None
        return None

    def background_load_data(self):
        while True:
            if len(self.data_queue.items) < self.max_data:
                tar_data = self.tar_queue.pop()
                if tar_data is None:
                    time.sleep(1)
                    continue

                try:
                    tar_file = io.BytesIO(tar_data)
                    with tarfile.open(fileobj=tar_file) as tar:
                        members = tar.getmembers()
                        
                        # Process all jpg files
                        for member in members:
                            if member.name.endswith('.jpg') and not member.name.endswith(('.depth.jpg', '.flow.jpg')):
                                image = self.process_image_file(tar, member.name)
                                if image is None:
                                    continue
                                    
                                result = [image]
                                
                                if self.include_depth:
                                    depth_name = member.name.replace('.jpg', '.depth.jpg')
                                    depth_image = self.process_image_file(tar, depth_name)
                                    if depth_image is None:
                                        continue
                                    result.append(depth_image)
                                    
                                if self.include_flow:
                                    flow_name = member.name.replace('.jpg', '.flow.jpg') 
                                    flow_image = self.process_image_file(tar, flow_name)
                                    if flow_image is None:
                                        continue
                                    result.append(flow_image)
                                    
                                if len(result) > 1:
                                    self.data_queue.add(tuple(result))
                                else:
                                    self.data_queue.add(result[0])

                except Exception as e:
                    print(f"Error processing tar: {e}")
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
    
    # Stack all tensors
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
            depth_tensor = depth_tensor[:,:,0] # Take first channel if RGB
        depth_tensor = depth_tensor.unsqueeze(0) / 255.0 # Add channel dim
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
            depth_tensor = depth_tensor[:,:,0] # Take first channel if RGB
        depth_tensor = depth_tensor.unsqueeze(0) / 255.0 # Add channel dim
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
    loader = get_loader(4, include_depth = True, bucket_name='cod-raw-360p-30fs',prefix='depth-and-raw')

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