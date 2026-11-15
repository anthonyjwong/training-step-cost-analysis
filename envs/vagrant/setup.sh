#!/bin/bash
set -euo pipefail

# Require root
if [ "$EUID" -ne 0 ]; then
  echo "Error: this script must be run with sudo" >&2
  exit 1
fi

# Update the package list
apt-get update

# === Install required packages ===
apt-get install -y --no-install-recommends python3-venv python3-pip

# === Setup python venv ===
sudo -u vagrant python3 -m venv /home/vagrant/venv
sudo -u vagrant /home/vagrant/venv/bin/pip install --upgrade pip
sudo -u vagrant /home/vagrant/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision # gpu is not available on VM
sudo -u vagrant /home/vagrant/venv/bin/pip install python-dotenv

# === Setup .bashrc for login items ===
grep -q 'python venv' /home/vagrant/.bashrc || echo '# python venv
source /home/vagrant/venv/bin/activate
cd /home/vagrant/proj1' >> /home/vagrant/.bashrc
