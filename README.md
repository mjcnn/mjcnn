M's Java Convolutional Neural Network (MJCNN)

This is a Java CNN implementation, which is somewhat optimized with Aparapi GPU support. It was not designed dreaming that it would be fast enough for training, but rather for classification with previously trained weights, to be used in apps.

MJCNN supports loading of CNN structures and of trained weights from JSON files that are compatible with data easily obtainable from VGG16 h5 files (the corresponding scripts are provided).

MJCNN supports the following types of CNN Layers:
* CONVOLUTION
* RELU perceptron
* SIGM perceptron
* TANH perceptron
* POOL MAX
* POOL AVG
* SOFTMAX
* INPUT
Note that unlike with other common architectures, in MJCNN RELU/SIGM/TANH perceptrons include a convolution.

The API is available through the class cnn.CNN while constants and configuration can be loaded not only with JSON but also based on prepared arrays of cnn.Config.LayerConfig


About the repository:


- 👋 Hi, I’m @mjcnn
- 👀 I’m interested in plenty of things
- 🌱 My net is currently planning to learn.
- 💞️ I’m looking forward!
- 📫 How to reach me?

<!---
mjcnn/mjcnn is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
