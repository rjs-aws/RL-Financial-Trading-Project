import sagemaker
import pandas as pd
import numpy as np
import os
import sys
import sys
import boto3
from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
import datetime
import random
import math


# ======================================
#
#    Read arguments from command line
#
# ======================================

if len(sys.argv) < 3:
    print("Train Usage: python run_rl_agent.py <your-outdir-name> train")
    print("Evaluate Usage: python run_rl_agent.py <your-other-outdir> evaluate <your-outdir-name>")
    exit(1)

# Local output filename
localout = sys.argv[1] 
mode = sys.argv[2]

# Optional third argument: name of checkpoint model folder name.
# For training, invoke with three arguments 
if len(sys.argv) == 4:
    # use the directory from the model training for
    # the checkpoint here.
    checkpointmodel = sys.argv[3]


if mode not in ("train", "evaluate"):
    print("Invaid Argument: Must be `train` or `evaluate`")
    exit(1)
    
# ======================================
#
#    Train or evaluate an RL agent
#
# ======================================

today = datetime.date.today()
print('Today is {}, let\'s get going!'.format(today.strftime('%b %d')))

sess = sagemaker.session.Session()
s3_bucket = sess.default_bucket()
s3_output_path = 's3://{}/'.format(s3_bucket)
job_name_prefix = 'rl-trading'
local_mode = False
role = sagemaker.get_execution_role()


# Determine the mode based on command line args.
# Different scripts are used depending on if the mode is training or evaluation (both reside in src dir)
# see SageMaker docs for further clarification
if mode == 'train': 
    entryfile = "train-coach.py"

# Move checkpointed model (pre-trained RL agent) to S3 (required in SageMaker RL)
elif mode == 'evaluate': 
    checkpoint_dir = '{}/checkpoint'.format(checkpointmodel) 
    print("Starting from pre-trained agent checkpointed at {}".format(checkpoint_dir))
    checkpoint_path = "{}checkpoint/".format(s3_output_path)
    
    # throw if dir provided doesn't exist
    if not os.listdir(checkpoint_dir):
        raise FileNotFoundError("Checkpoint files not found under the path")
    
    os.system("aws s3 rm --recursive {}".format(checkpoint_path))    
    os.system("aws s3 cp --recursive {} {}".format(checkpoint_dir, checkpoint_path))
    print("S3 checkpoint file path: {}".format(checkpoint_path))
    entryfile = "evaluate-coach.py"          

    

# Estimator requires the following:
# The source directory where the environment, presets, and training code are uploaded.
# path to the training script
# The RL toolkit and deep learning framework you want to use.
# The training parameters, such as the instance count, job name, and S3 path for output. 
estimator = RLEstimator(source_dir='src',
                        entry_point=entryfile,
                        dependencies=["common/sagemaker_rl"],
                        toolkit=RLToolkit.COACH,
                        toolkit_version='0.11.0',
                        framework=RLFramework.MXNET,
                        role=role,
                        train_instance_count=1,
                        train_instance_type='ml.c5.9xlarge',
                        output_path=s3_output_path,
                        base_job_name=job_name_prefix,
                        hyperparameters = {"RLCOACH_PRESET" : "preset-trading"})

# fit from scratch
if mode == 'train': 
    estimator.fit()
# fit from the pre-trained agent 
elif mode == 'evaluate': 
    estimator.fit({'checkpoint': checkpoint_path})

    
# Copy RL output from S3 to local output folder and uncompress files
job_name = estimator.latest_training_job.job_name
os.system('aws s3 cp {}{}/output/output.tar.gz ./{}/'.format(s3_output_path, job_name, localout))
os.system('cd {}; tar xzvf output.tar.gz; rm output.tar.gz; cd ..'.format(localout))

print('DRL reviews complete. \n')
print('Output location on S3: {}{}/output/'.format(s3_output_path, job_name))


# =====================================
#
#   Send email and SMS notfications
#
# =====================================
    
# Publish completion to SNS topic
sns = boto3.client('sns', region_name='us-west-2')
sns.publish(
            TopicArn = 'arn:aws:sns:us-west-2:576954791978:RLTopic',
            Subject = 'RL Agent Trading - Job Complete',
            Message = 'Ready to analyze. S3 output folder: {}{}/output and localfolder: ./{}'.format(s3_output_path, job_name, localout)    
           )
