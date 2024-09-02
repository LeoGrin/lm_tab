import submitit
import os

def setup_submitit_executor_a100(name, gpus_per_node=1, cpus_per_task=8, time=10):
    executor = submitit.AutoExecutor(folder="logs")  # Logs will be stored in this folder
    # Set the parameters for the job
    executor.update_parameters(
        name=name,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_node=gpus_per_node,  # Number of GPUs per node
        cpus_per_task=cpus_per_task,  # Number of CPUs for A100 in gpu_p5 (1/8 of an 8-GPU node)
        time=time,  # Max time in minutes
        #slurm_output="gpu_job_%j.out",  # Slurm output file
        #slurm_error="gpu_job_%j.err",   # Slurm error file
        slurm_additional_parameters={"constraint": "a100",
                                    "account": "ptq@a100",
                                    "hint": "nomultithread",
                                    "partition": "gpu_p5"},
    )

    # Load necessary modules for A100
    os.system("module purge")
    os.system("module load cpuarch/amd")
    os.system("module load pytorch-gpu/py3/2.2.0")  # Adjust to your required PyTorch version
    print("Loaded modules")
    return executor
