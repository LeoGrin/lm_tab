import submitit
import torch
import os
def run_job():
    # Load PyTorch and print the number of available GPUs
    print(f"PyTorch version: {torch.__version__}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

def main():
    # Create a Slurm executor
    executor = submitit.AutoExecutor(folder="log_gpu_jobs")  # Logs will be stored in this folder

    # Set the parameters for the job
    executor.update_parameters(
        name="submitit_gpu_job",
        nodes=1,
        ntasks_per_node=1,
        gpus_per_node=1,  # Number of GPUs per node
        cpus_per_task=8,  # Number of CPUs for A100 in gpu_p5 (1/8 of an 8-GPU node)
        time=10,  # Max time in minutes
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

    # Submit the job
    job = executor.submit(run_job)

    # Print job ID and output location
    print(f"Submitted job with ID {job.job_id}")
    print(f"Job logs will be saved in {executor.folder}")

if __name__ == "__main__":
    main()
