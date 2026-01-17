#!/bin/bash
set -e  # Exit on error

echo "Updating apt package list..."
apt-get update

echo "Installing unzip..."
apt-get install -y unzip

echo "Installing AWS CLI..."
# Install AWS CLI v2 using the official installer
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

echo "Installing huggingface-cli..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying installations..."
echo "AWS CLI version:"
aws --version

echo "Hugging Face CLI version:"
huggingface-cli --help

git config user.name Stevada
git config user.email stevexu247@gmail.com
git config --list

mkdir dataset/
mkdir logs/
mkdir output/

echo "Setup completed successfully!"
