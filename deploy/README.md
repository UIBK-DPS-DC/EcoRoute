# Deployment on Chameleon Cloud

## Connect to the [Chameleon Jupyter Notebook](https://chameleoncloud.org/experiment/jupyter-notebooks/)

Upload the notebook `deploy/chameleon/leases_instances.ipynb`. Configure the Codeblock under `Create leases` by 

- setting start and end dates for the leases
- setting the amount of nodes to reserve (leave unchanged for minimal deployment of one router and one model on each site)

Run the codeblock under `Create instances` to start all instances. You can use the codeblock under `Check instance status` to display the current status of all instances. If any of the instances don't have the status `ACTIVE` after about 10 minutes or even have the status `ERROR`, delete the instances through the Chameleon UI for the site under e.g. `Experiment -> CHI@UC` and run `Create instances` again.

Once all instances are active, run `Create inventory.ini for Ansible` which associates the floating IPS with the instances and creates the content of `inventory.ini` for Ansible. Copy the last lines looking like the example below and paste them into `deploy/ansible/inventory.ini`.

``` ini
[router]
router_uc ansible_host=192.5.86.227 ansible_user=cc role=uc
router_tacc ansible_host=129.114.109.237 ansible_user=cc role=tacc
router_edge ansible_host=129.114.34.229  ansible_user=cc role=edge

[model_nvidia]
model_uc_1 ansible_host=192.5.87.206 ansible_user=cc role=uc

[model_amd]
model_tacc_1 ansible_host=129.114.109.72 ansible_user=cc role=tacc

[model_cpu]
model_edge_1 ansible_host=129.114.34.198 ansible_user=cc role=edge
```

## Configure the device specific variables in `ansible/host_vars`.

For each router and model, create a `yml`-file (e.g. `router_uc.yml` for the `router_uc` definition in `ansible/inventory.ini`). 
- For each router define the keys `router` and `nats`. They point to the router container and nats server container respectively (in `docker-compose.yml`), which will be started on the instance. (This step will be automated in the future)
- For each model in site `CHI@UC`, define the key `model` with the HuggingFace model you want to use on this instance (e.g. `model: "Qwen/Qwen2.5-3B-Instruct"`)
- For each model in site `CHI@TACC` and `CHI@Edge`, define the key `model` with the HuggingFace model you want to use on this instance. These sites deploy the models using `Ollama` instead of `vLLM`. Hence, the models need to be in `GGUF`-format (e.g. `model: "Qwen/Qwen2.5-3B-Instruct-GGUF"`). Additionally, the key `model_file` needs to be defined with the filename of quantization you want to use for the model (e.g. `qwen2.5-3b-instruct-q4_k_m.gguf`)

If you want to use HuggingFace models that require gated access, you must provide your HuggingFace Access token in a `secrets.yml` file. E.g. if models using NVIDIA GPUs shall use such models create the file `deploy/ansible/group_vars/model_nvidia/secrets.yml` using the Ansible vault: 

`ansible-vault create group_vars/model_nvidia/secrets.yml`

Enter the following line with your HuggingFace Access token: `hf_token: "TOKEN"`

Save the vault password to a location on your file system for easier deployment. E.g. `~/.vault_pass.txt`

## Deployment 

Either connect to all instances once manually to accept the host key for them or run the script `utils/prep_ssh_hosts.sh` to do this automatically by writing the IPs into `/.ssh/known_hosts`. Note that the script does not check the host keys. Hence, the script does not protect against potential man-in-the-middle attacks.

Deploy routers and models using the following command:

`ansible-playbook -i inventory.ini deploy.yml --skip-tags url --vault-password-file ~/.vault_pass.txt`

The playbook also contains a task which sets aliases for the floating IPs of all instances in `/etc/hosts` such that the ssh connection to the instances is easier. E.g. `ssh cc@router_edge` for the edge router. If you want this as well, you can run the following command. Note that you need to provide your sudo password for ansible to be allowed to write to `/etc/hosts`.

`ansible-playbook -i inventory.ini deploy.yml -K --tags url --vault-password-file ~/.vault_pass.txt`


If you have not installed `ansible` yet, install it for example with pip: `pip install ansible`

## Evaluation

For evaluation you need to download the datasets. Execute the script `utils/prepare_datasets.py` to download the datasets to disk. Adjust the variable `save_dir` for your desired download location. Then run the script `utils/generate_datasets_for_evaluation.py`. Set the variable `hf_dir` to `save_dir` from the previous script. The prepared datasets are then stored at `save_dir/targets` Note that this requires approximately 4.6 GB of disk space.

You can run evaluation runs using the following command where `-n` gives the number of queries per seconds that are sent to the router under `http://router_edge` and `-t` is the duration of the run in minutes:

`python eval.py http://router_edge -n 1 -p save_dir/targets -t 10`

To download the DuckDB databases of each router containing all metrics collected during the run, run the following command. First adjust the variable `save_dir` under `vars` in `deploy/ansible/metrics.yml` to the location where the databases shall be saved on your disk.

`ansible-playbook -i inventory.ini metrics.yml`

You can use the scipts `utils/query_db.py` and `utils/db_check_times.py` which create some plots using the latest downloaded database. These plots are still work in progress.
