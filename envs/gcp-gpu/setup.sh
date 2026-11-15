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

# === Setup python venv (as the user who ran sudo) ===
sudo -u "$SUDO_USER" python3 -m venv "/home/$SUDO_USER/venv"
sudo -u "$SUDO_USER" "/home/$SUDO_USER/venv/bin/pip" install --upgrade pip
sudo -u "$SUDO_USER" "/home/$SUDO_USER/venv/bin/pip" install torch torchvision
sudo -u "$SUDO_USER" "/home/$SUDO_USER/venv/bin/pip" install python-dotenv

# === Setup .bashrc for login items ===
grep -q 'python venv' "/home/$SUDO_USER/.bashrc" || echo "# python venv
source /home/$SUDO_USER/venv/bin/activate
cd /home/$SUDO_USER/proj1" >> "/home/$SUDO_USER/.bashrc"
