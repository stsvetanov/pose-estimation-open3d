# Pose estimation using Open3D
This project demonstrate pose estimation in 3D space using data from depth camera.

## Project Struction
This is the breakdown of a project.
```
pose-estimation-open3d
│   README.md
│  .gitignore
│   requirements.txt
│   ...
│
│
└───data
│   │   ...
│   │
└───icp_data
│   │   ...
│   │
└───poses
    │   ...

```

## Deployment
Let's walk through setting up the development environment and deployment

1. Install Python, pip, and virtualenv
```
sudo apt-get install python3
sudo apt install python3-pip
sudo apt install python3-virtualenv
```

2. Clone this repo and CD into the projects directory
```
git clone git@github.com:stsvetanov/pose-estimation-open3d.git
cd pose-estimation-open3d
```

3. Create and activate a virtualenv
```
virtualenv venv
source venv/bin/activate
```

4. Install packages
```
pip install -r requirements.txt
```

5. Create a new dataset
```
python3 collect_data_from_live_capture.py
```

6. Run pose estimation
```
python3 run_icp_alignment.py
```