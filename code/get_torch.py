import subprocess

torch_version_suffix = ""

subprocess.call(['pip', 'install', 'torch-cpu==1.7.1' + torch_version_suffix, 'torchvision-cpu==0.8.2' + torch_version_suffix, '-f', 'https://download.pytorch.org/whl/torch_stable.html', 'ftfy', 'regex'])