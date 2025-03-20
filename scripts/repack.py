import os
import shutil
import tarfile
import argparse
import tempfile
from femo.logger import LOGGER


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract(tar_filepath: str, temp_dir: str):
    """
    Safely extract tar.gz or copy joblib files to a temporary directory.
    """
    LOGGER.info(f"{tar_filepath = }")

    # If file is not a tar.gz, simply copy it
    if not tar_filepath.endswith(".tar.gz"):
        try:
            if tar_filepath.endswith(".joblib") or tar_filepath.endswith(".h5"):
                shutil.copy(tar_filepath, temp_dir)
            return
        except Exception as e:
            LOGGER.error(f"Error copying {tar_filepath}: {e}")
            raise

    # If file is a tar.gz, extract it safely
    with tarfile.open(tar_filepath) as tar:
        for member in tar.getmembers():
            member_path = os.path.join(temp_dir, member.name)
            if not is_within_directory(temp_dir, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(temp_dir)


def repack_joblib_files(work_dir, output_tar):
    """
    Repack all .joblib files from work_dir into a tar.gz archive.
    """
    # Collect all .joblib files
    joblib_files = []
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith(".joblib") or file.endswith(".h5"):
                joblib_files.append(os.path.join(root, file))

    # Create a tar.gz file containing all .joblib files
    with tarfile.open(output_tar, "w:gz") as tar:
        for joblib_file in joblib_files:
            tar.add(joblib_file, arcname=os.path.relpath(joblib_file, work_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, required=True, help="Path to classifier object file (.tar.gz or .joblib)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.tar.gz or .joblib or .h5)")
    parser.add_argument("--pipeline", type=str, required=True, help="Path to data pipeline object (.tar.gz or .joblib)")
    parser.add_argument("--processor", type=str, required=True, help="Path to data processor object (.tar.gz or .joblib)")
    parser.add_argument("--metrics", type=str, required=True, help="Path to evaluation metrics object (.tar.gz or .joblib)")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()
    return args


def main(args):
    LOGGER.info("Starting model repacking...")
    temp_dir = tempfile.mkdtemp(dir=args.work_dir)

    try:
        # Extract or copy files into the temporary directory
        safe_extract(args.classifier, temp_dir)
        safe_extract(args.model, temp_dir)
        safe_extract(args.pipeline, temp_dir)
        safe_extract(args.processor, temp_dir)
        safe_extract(args.metrics, temp_dir)

        # Repack all .joblib files into a single tar.gz file
        os.makedirs(os.path.join(args.work_dir, "repack"), exist_ok=True)
        output_tar = os.path.join(args.work_dir, "repack", "model.tar.gz")
        repack_joblib_files(temp_dir, output_tar)

        LOGGER.info(f"Model repacked to {output_tar}")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
