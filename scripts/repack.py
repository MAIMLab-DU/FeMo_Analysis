import os
import tarfile
import argparse
from femo.logger import LOGGER


def is_within_directory(directory, target):         
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)

            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory

def safe_extract(tar_filepath, path="."):
    with tarfile.open(tar_filepath) as tar:
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(path)

def repack_joblib_files(work_dir, output_tar):
    # Collect all .joblib files in the work_dir
    joblib_files = []
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith(".joblib"):
                joblib_files.append(os.path.join(root, file))

    # Create a tar file and add all .joblib files
    with tarfile.open(output_tar, "w:gz") as tar:
        for joblib_file in joblib_files:
            tar.add(joblib_file, arcname=os.path.relpath(joblib_file, work_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained classifier file (.tar.gz)")
    parser.add_argument("--pipeline", type=str, required=True, help="Path to data pipeline object (.tar.gz)")
    parser.add_argument("--processor", type=str, required=True, help="Path to data processor object (.tar.gz)")
    parser.add_argument("--metrics", type=str, required=True, help="Path to evaluation metrics object (.tar.gz)")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args


def main(args):
     LOGGER.info("Starting model repacking...")

     safe_extract(args.model, args.work_dir)
     safe_extract(args.pipeline, args.work_dir)
     safe_extract(args.processor, args.work_dir)
     safe_extract(args.metrics, args.work_dir)
     # Repack all .joblib files into a single tar.gz file
     os.makedirs(os.path.join(args.work_dir, "repack"), exist_ok=True)
     output_tar = os.path.join(args.work_dir, "repack", "model.tar.gz")
     repack_joblib_files(args.work_dir, output_tar)

     LOGGER.info(f"Model repacked to {output_tar}")


if __name__ == "__main__":
     args = parse_args()
     main(args)