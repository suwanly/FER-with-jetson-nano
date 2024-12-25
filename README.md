# FER with Jetson Nano

## Introduction
Facial Emotion Recognition (FER) is a technology that uses AI to identify human emotions from facial expressions. This project leverages the Jetson Nano, a powerful edge computing device, to perform FER efficiently. The focus is on optimizing the model through techniques like pruning and quantization to ensure it runs smoothly on the Jetson Nano.

## Features
- **Pruning**: Reduces the size of the model by removing unnecessary weights, improving performance without significant loss of accuracy.
- **Quantization**: Converts the model to use lower precision, reducing memory usage and increasing inference speed.

## Requirements
- **Hardware**: Jetson Nano or Apple silicon (i used M4)
- **Software**:
  - Python 3.8
  - TensorFlow
  - OpenCV
  - NumPy

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/FER-with-jetson-nano.git
   ```
2. **Navigate to the project directory**
3. **Install the required packages**

## Usage
1. **Prepare your dataset**: Ensure your dataset is in the correct format and path as specified in the configuration file.
2. **Train the model**:
    please use ipynb file to train the model,
    or you can use the trained model in the model folder
3. **Run the FER model**:
   Also you can use this code in jetson nano to run the model.

## Contributing
We welcome contributions! If you have suggestions or improvements, please fork the repository and submit a pull request. Ensure your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
Special thanks to the open-source community and contributors who have made this project possible.

