#!/bin/bash
set -e

VIRTUAL_ENV=.venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to handle SIGTERM signal
terminate_script() {
    echo "Received SIGTERM, terminating the script gracefully..."
    exit 0
}

# Trap SIGTERM signal and call the terminate_script function
trap terminate_script SIGTERM

function usage {
    echo "usage: build_and_push.sh [-i account_id] [-a algorithm_name] [options]"
    echo "Required:"
    echo "  -r        AWS Region with access to ECR repository."
    echo "  -a        Algorithm/Image name that will be pushed."
    exit 1
}

algorithm_flag=false
region_flag=false

while getopts "cmdr:a:h" opt; do
    case $opt in
        r  ) region_flag=true; region=$OPTARG;;
        a  ) algorithm_flag=true; algorithm_name=$OPTARG;;
        h  ) usage; exit;;
        \? ) echo "Unknown option: -$OPTARG" >&2; exit 1;;
        :  ) echo "Missing option argument for -$OPTARG" >&2; exit 1;;
        *  ) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    esac
done

if  ! $region_flag
then
    echo "The region flag (-r) must be included for a build to run" >&2
fi

if  ! $algorithm_flag
then
    echo "The algorithm name (-a) must be included for a build to run" >&2
fi

account=$(aws sts get-caller-identity --query Account --output text)

# If the repository doesn't exist in ECR, return.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    exit 1
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com > /dev/null 2>&1

# Get the latest version of 'feme' package
# Set up virtual env
virtualenv -p python3 $VIRTUAL_ENV
. $VIRTUAL_ENV/bin/activate 
#Install requirements
pip install $SCRIPT_DIR/../. -q
version=$(pip show "femo" | grep "Version" | awk '{print $2}')
deactivate

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${version}"
latest_fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t "${algorithm_name}:${version}" -f Dockerfile $SCRIPT_DIR/..
docker tag "${algorithm_name}:${version}" ${fullname}
docker tag "${algorithm_name}:${version}" ${latest_fullname}

docker push ${fullname}
docker push ${latest_fullname}
