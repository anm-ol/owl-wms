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
    def __init__(self, bucket_name="cod-yt-audio", rank=0, world_size=1, window_length=88200):
        super().__init__()
        
        self.rank = rank
        self.world_size = world_size
        self.bucket_name = bucket_name
        self.window_length = window_length

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
        """List all tar files in the bucket"""
        tar_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name):
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

    def process_audio_file(self, tar, audio_name):
        try:
            f = tar.extractfile(audio_name)
            if f is not None:
                audio_data = f.read()
                # Load audio using numpy
                audio = np.frombuffer(audio_data, dtype=np.int16)
                # Reshape to (channels, samples)
                audio = audio.reshape(-1, 2)
                # Convert to float32 and normalize to [-1, 1]
                audio = audio.astype(np.float32) / 32768.0

                # If audio is longer than window_length, randomly sample a window
                if len(audio) > self.window_length:
                    start_idx = random.randint(0, len(audio) - self.window_length)
                    audio = audio[start_idx:start_idx + self.window_length]
                else:
                    # If audio is shorter, pad with zeros
                    pad_length = self.window_length - len(audio)
                    audio = np.pad(audio, (0, pad_length))
                    
                return audio
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
                        
                        # Process all wav files
                        for member in members:
                            if member.name.endswith('.wav'):
                                audio = self.process_audio_file(tar, member.name)
                                if audio is not None:
                                    self.data_queue.add(audio)

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
    # Convert numpy arrays to tensors
    tensors = [torch.from_numpy(audio).float() for audio in batch]
    return torch.stack(tensors).transpose(-1,-2) # -> [b,n,d]

def get_loader(batch_size, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = S3CoDDataset(rank=rank, world_size=world_size, **data_kwargs)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
    import time
    loader = get_loader(4)

    start = time.time()
    batch = next(iter(loader))
    print(batch.shape)