import submitit
executor = submitit.AutoExecutor(folder="logs")
executor.update_parameters(timeout_min=1000, slurm_partition='parietal,normal', slurm_array_parallelism=200, cpus_per_task=2,
                            exclude="margpu009")
def f(a, b):
    print(a, b)
    return a + b

jobs = []
for a in range(10):
    for b in range(10):
        job = executor.submit(f, a, b)
        jobs.append(job)

for job in jobs:
    print(job.result())
    
