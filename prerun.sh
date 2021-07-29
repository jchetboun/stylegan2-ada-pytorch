# support git lfs
apt-get update -y
apt-get install git-lfs -y
git lfs install

# For a more compact command line
echo "export PS1='\w$ '" >> ~/.bashrc
# Add root dir to python path
echo "export PYTHONPATH=$PWD" >> ~/.bashrc

# Prerequisite to use cv2 on cnvrg
apt-get install -y libgl1-mesa-glx

apt-get install gettext-base
envsubst < /cnvrg/clearml_template.conf > /cnvrg/clearml.conf
mv /cnvrg/clearml.conf /root
