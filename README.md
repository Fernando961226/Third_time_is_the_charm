## MMWhale
This repository is a tracked, in-place modification of the [MMDetection codebase](https://github.com/open-mmlab/mmdetection#readme) intended specifically for collaboration on the VIP Lab whale detection project. New model configurations and customizations, data pipelines, and pre-/post-processing algorithms should be integrated into this codebase, following the best practices outlined below.  

The MMDetection code included here was copied from the MMDetection repository at release 2.25.1 (commit hash [3b72b12](https://github.com/open-mmlab/mmdetection/commit/3b72b12fe9b14de906d1363982b9fba05e7d47c1)), on August 24, 2022. It should be noted that this code has been severed from the MMDetection repository, so updating it to match later versions will take a bit more work than a simple `git pull`.

### Running code on compute canada
This repository tracks code on the shared filesystem of a compute canada (CC) instance. Running python code in this environment looks a bit different than on a local filesystem. Instead of just running a script in a pre-configured python environment, on CC we submit a 'job' that:
1. specifies the computational and memory resources required for the task 
2. creates and configures the python environment (installs necessary dependencies, etc.)
3. invokes the python script to be run.

An example of such a script can be seen in [`inference_sample.sh`](https://git.uwaterloo.ca/vip-whale-detection/mmwhale/-/blob/main/inference_sample.sh).

Instead of running the job script directly on the CC filesystem, we submit it to the scheduler using `sbatch`; e.g. `sbatch inference_sample.sh`.

Information on [job scheduling](https://docs.alliancecan.ca/wiki/Running_jobs), [python environment configuration](https://docs.alliancecan.ca/wiki/Python) and more can be found on the [CC Docs](https://docs.alliancecan.ca/wiki/Technical_documentation).

### Best Practices
As this code is intended to be hosted and run on a shared filesystem (compute canada) with multiple collaborators, it is envisioned that some shared version-control and code-structuring practices will help manage the chaos. (e.g. - 'accidentally' deleting someone else's files, modifying some core code and breaking things for everyone else, creating inconsistent file structures, etc.).

#### Branching
When working on a new model, data setup or other adventure, it is probably a good idea to do so in a separate branch named to reflect this undertaking.

1. Sync to the most current version of the `master` branch (`git fetch origin` & `git pull` on master)
2. Create a new branch from master with the following naming convention: `{your_name}/{short-description}`, e.g. `neil/retina-net-baseline`
3. Do some work, commit your changes relatively frequently, and push to a corresponding remote branch (`git push -u origin/{branch-name}`)
4. If anyone adds changes to the `master` branch while you work, update your branch to build on those changes (e.g. `git merge master` or `git rebase master`). Trying to merge a branch that changes the same files someone else also just changed never ends in happy times.
5. When you've finished and tested your code / have given up and want to move on to something new, create a merge request to master on GitLab. If everyone's happy and you don't have too many code conflicts with the master branch, run the merge request to incorporate your branch into master.

#### Code Structure
In general, try to follow the MMDetection way of doing things while keeping your files isolated from others'. This probably looks like, e.g., putting new model configuration files in a folder like mmwhale/configs/custom/{name_of_your_model}, customized model code in mmwhale/mmdet/models/necks/custom/{name_of_your_neck}, etc. Having a 'custom' folder where each component lives signals to other team members that it was developed by one of us, and having separate folders within this for each customization keeps our code separate from each other. If we must cooperate on the same code files, having git in place will facilitate that.
