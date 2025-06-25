import torch
import os
import sys
import boto3
import dotenv

dotenv.load_dotenv()

def extract_checkpoint(path, output_path):
    d = torch.load(path, map_location='cpu', weights_only = False)
    prefix = ""
    if 'ema' in d: 
        d = d['ema']
        prefix += "ema_model."
    if any('.module.' in k for k in d.keys()):
        prefix += "module."
    
    d = {k[len(prefix):] : v for (k,v) in d.items() if k.startswith(prefix)}
    
    if output_path.startswith('s3://'):
        # Save to temporary file first
        tmp_path = 'tmp_checkpoint.pt'
        torch.save(d, tmp_path)
        
        # Parse bucket and key from s3 path
        parts = output_path[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        # Upload to S3
        s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
        )
        s3_client.upload_file(tmp_path, bucket, key)
        os.remove(tmp_path)
    else:
        torch.save(d, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("This script extracts model weights from a checkpoint file, removing any EMA or DataParallel prefixes.")
        print("Usage: python extract_checkpoint.py <input_checkpoint> <output_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input checkpoint '{input_path}' does not exist")
        sys.exit(1)
        
    try:
        extract_checkpoint(input_path, output_path)
        print(f"Successfully extracted weights to {output_path}")
    except Exception as e:
        print(f"Error processing checkpoint: {str(e)}")
        sys.exit(1)