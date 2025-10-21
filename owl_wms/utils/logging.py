import torch.distributed as dist
import wandb
import torch

import einops as eo

import numpy as np
from .vis import draw_frames
from .vis_tekken import draw_tekken_frames
from moviepy.editor import ImageSequenceClip, CompositeVideoClip
from moviepy.audio.AudioClip import AudioArrayClip

from .vis import draw_frames

import os
import pathlib


class LogHelper:
    """
    Helps get stats across devices/grad accum steps

    Can log stats then when pop'd will get them across
    all devices (averaged out).
    For gradient accumulation, ensure you divide by accum steps beforehand.
    """
    def __init__(self):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self.data = {}

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().item()
        val = data / self.world_size
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [val]

    def log_dict(self, d):
        for (k,v) in d.items():
            self.log(k,v)

    def pop(self):
        reduced = {k : sum(v) for k,v in self.data.items()}

        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, reduced)

            final = {}
            for d in gathered:
                for k,v in d.items():
                    if k not in final:
                        final[k] = v
                    else:
                        final[k] += v
        else:
            final = reduced

        self.data = {}
        return final

@torch.no_grad()
def to_wandb(x, actions, format='mp4', gather = False, max_samples = 8, fps=30):
    # x is [b,n,c,h,w]
    x = x.clamp(-1, 1)
    x = x[:max_samples]

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        x = torch.cat(gathered, dim=0)

    # Get labels on them
    b, _ = actions.shape
    temporal_compression = x.size(1) // actions.size(1) + 1
    actions = actions.unsqueeze(-1).repeat(1, 1, temporal_compression).view(b, -1)
    x = draw_tekken_frames(x, actions) # -> [b,n,c,h,w] [0,255] uint8 np

    if max_samples == 8:
        x = eo.rearrange(x, '(r c) n d h w -> n d (r h) (c w)', r = 2, c = 4)

    return wandb.Video(x, format=format, fps=fps)

@torch.no_grad()
def to_wandb_pose(x, actions, format='mp4', gather = False, max_samples = 8, fps=30):
    # x is [b,n,4,h,w]
    rgb, pose = torch.split(x, [3, 1], dim=2)
    print(f"RGB video shape: {rgb.shape}")
    rgb_videos = to_wandb(rgb, actions, format=format, gather=gather, max_samples=max_samples, fps=fps)
    pose = pose.repeat(1, 1, 3, 1, 1)
    print(f"Pose video shape: {pose.shape}")
    pose = pose.clamp(-1, 1)
    pose = pose[:max_samples]

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, pose)
        pose = torch.cat(gathered, dim=0)

    # Get labels on them
    b, _ = actions.shape
    temporal_compression = pose.size(1) // actions.size(1) + 1 # this is ugly logic needs to be fixed
    #added another ugly logic to fix temporal_compression when there is none
    if temporal_compression == 2:
        temporal_compression = 1

    actions = actions.unsqueeze(-1).repeat(1, 1, temporal_compression).view(b, -1)
    pose = draw_tekken_frames(pose, actions) # -> [b,n,c,h,w] [0,255] uint8 np

    if max_samples == 8:
        pose = eo.rearrange(pose, '(r c) n d h w -> n d (r h) (c w)', r = 2, c = 4)

    return rgb_videos, wandb.Video(pose, format=format, fps=fps)

def to_wandb_gif(x, actions, max_samples = 4, format='mp4', fps=16):
    x = x.clamp(-1, 1)
    x = (x + 1) * 127.5
    x = x.to(torch.uint8)
    x = x[:max_samples]
    x = eo.rearrange(x, 'b n c h w -> n c h (b w)' )
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    return wandb.Video(x, format=format, fps=fps)

@torch.no_grad()
def to_wandb_av(x, audio, batch_mouse, batch_btn, gather = False, max_samples = 8):
    # x is [b,n,c,h,w]
    # audio is [b,n,2]
    x = x.clamp(-1, 1)
    x = x[:max_samples].cpu().float()

    if False: #x.shape[2] > 3:
        depth = x[:,:,3:4]
        flow = x[:,:,4:7]
        x = x[:,:,:3]

        depth_gif = to_wandb_gif(depth)
        flow_gif = to_wandb_gif(flow)

        feat = True
    else:
        feat = False

    if audio is not None:
        audio = audio[:max_samples].cpu().float().detach().numpy()

    if dist.is_initialized() and gather:
        gathered_x = [None for _ in range(dist.get_world_size())]
        gathered_audio = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_x, x)
        dist.all_gather(gathered_audio, audio)
        x = torch.cat(gathered_x, dim=0)
        if audio is not None:
            audio = torch.cat(gathered_audio, dim=0)

    # Get labels on frames
    x = draw_frames(x, batch_mouse, batch_btn) # -> [b,n,c,h,w] [0,255] uint8 np

    # Convert both to list of [n,h,w,c] and [n,2] numpy arrays
    x = [np.moveaxis(x[i], 1, -1) for i in range(len(x))]
    if audio is not None:
        audio = [audio[i] for i in range(len(audio))]

    os.makedirs("temp_vids", exist_ok = True)
    paths = [f'temp_vids/temp_{i}.mp4' for i in range(len(x))]
    for i, path in enumerate(paths):
        write_video_with_audio(path, x[i], audio[i] if audio is not None else None)

    if feat:
        return [wandb.Video(path, format='mp4') for path in paths], depth_gif, flow_gif
    else:
        return [wandb.Video(path, format='mp4') for path in paths]


@torch.no_grad()
def to_wandb_samples(video, mouse, btn):
    video = video.clamp(-1, 1).cpu().float()          # [B, T, C, H, W]

    depth_gif = flow_gif = None
    if video.shape[2] > 3:                            # depth
        depth_gif = to_wandb_gif(video[:, :, 3:4])
    if video.shape[2] > 4:                            # flow
        flow_gif = to_wandb_gif(video[:, :, 4:7])
    video = video[:, :, :3]                           # keep RGB only

    video = draw_frames(video, mouse, btn)            # overlay labels – uint8

    out_dir = pathlib.Path("temp_vids")
    out_dir.mkdir(exist_ok=True)
    samples = []
    for i, clip in enumerate(video):
        path = out_dir / f"{i:04}.mp4"
        write_video_with_audio(
            str(path),
            np.moveaxis(clip, 1, -1),                 # [T, H, W, C]
            audio=None,
        )
        samples.append(wandb.Video(str(path), format="mp4"))

    artefacts = {"samples": samples}
    if depth_gif is not None:
        artefacts["depth_gif"] = depth_gif
    if flow_gif is not None:
        artefacts["flow_gif"] = flow_gif
    return artefacts


def write_video_with_audio(path, vid, audio, fps=60,audio_fps=44100):
    """
    Writes videos with audio to a path at given fps and sample rate

    :param video: [n,h,w,c] [0,255] uint8 np array
    :param audio: [n,2] stereo audio as np array norm to [-1,1]
    """
    # Create video clip from image sequence
    video_clip = ImageSequenceClip(list(vid), fps=fps)

    if audio is not None:
        # Create audio clip from array
        audio_clip = AudioArrayClip(audio, fps=audio_fps)
        # Combine video with audio
        video_clip = video_clip.set_audio(audio_clip)

    # Write to file
    video_clip.write_videofile(
        path,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
