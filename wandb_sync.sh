#!/bin/bash

while true; do
    wandb sync wandb/latest-run
    sleep 60
done