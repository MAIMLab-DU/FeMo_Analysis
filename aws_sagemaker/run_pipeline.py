"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import sys
from femo.logger import LOGGER
from utils import get_pipeline_driver, convert_struct

def get_model_package_name(pipeline_steps):
    for step in pipeline_steps:
        if step["StepName"] == "RegisterModel-RegisterModel":
            return step["Metadata"]["RegisterModel"]["Arn"]

def main():
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)
    args.kwargs = convert_struct(args.kwargs)

    try:
        LOGGER.info(f"Local mode: {args.kwargs.get('local_mode', False)}")
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        LOGGER.info("Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        LOGGER.info(json.dumps(parsed, indent=4, sort_keys=True))

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=tags
        )
        LOGGER.info("Created/Updated SageMaker Pipeline: Response received:")
        LOGGER.info(upsert_response)

        execution = pipeline.start()
        if not args.kwargs.get('local_mode', False):
            LOGGER.info(f"Execution started with PipelineExecutionArn: {execution.arn}")
            LOGGER.info("Waiting for the execution to finish...")
            execution.wait()
            LOGGER.info("Execution completed. Execution step details:")

        pipeline_steps = execution.list_steps()
        LOGGER.info(f"{pipeline_steps = }")

        if not args.kwargs.get('local_mode', False):
            model_package_name = get_model_package_name(pipeline_steps)
            with open("pipelineExecution.json", "w") as f:
                json.dump({"arn": model_package_name}, f, indent=2)
        # out_file = open("pipelineExecutionArn", "w")
        # out_file.write(model_package_name)
        # out_file.close()
                
    except Exception as e:
        LOGGER.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()