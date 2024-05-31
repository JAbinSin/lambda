# lambda
Research Project: Face Detection and Recognition Door Lock System Using Image Processing with Behavioral Analysis

## Installation

Create an [Environment](https://flask.palletsprojects.com/en/3.0.x/installation/#create-an-environment) And [Activate](https://flask.palletsprojects.com/en/3.0.x/installation/#activate-the-environment)

Install the requirements use:

`pip3 install -r requirements.txt`

PyTorch requirements vary by operating system and CUDA requirements, so it's recommended to install PyTorch first following instructions at https://pytorch.org/get-started/locally.

## How to run

If you have a virtual environment activate it

`.venv\Scripts\activate`

To run the web application use the following command

`flask --app server run --host=0.0.0.0 --debug`


## Requirements

### Hardware Requirement

| Hardware | Minimum | Recommended |
| :---:   | :---: | :---: |
| Processor | Intel Core i5 11th Gen / AMD Ryzen 5 5th Gen | Intel Core i7 11th Gen / AMD Ryzen 7 5th Gen |
| RAM | 8GB | 16GB |
| GPU | NVIDIA GTX 1050 | NVIDIA RTX 2080 |

### Minimum Camera Required

| Hardware | Minimum | Recommended |
| :---:   | :---: | :---: |
| Resolution | 720p (1280 x 720 pixels) | 1080p (1920 x 1080 pixels) |
| Framerate | 30fps (frame per seconds) | 60fps (frame per seconds) |
| Connection | USB 2.0 | USB 3.0 |

### Software Requirements

- [Python >= 3.9](http://docs.python-guide.org/en/latest/starting/installation/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- python3-dev `sudo apt-get install python3-dev  # for python3.x installs`
- [virtualenv](https://flask.palletsprojects.com/en/3.0.x/installation/#) (Recommended)
