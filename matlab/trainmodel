% Load data
data = load('data.mat');

% Split data into training and testing sets
train_data = data(1:80, :);
test_data = data(81:end, :);

% Preprocess data (e.g., normalize, one-hot encoding)
train_data = normalize(train_data);
test_data = normalize(test_data);

% Define the input layer
input_layer = imageInputLayer([28 28 1], 'Name', 'input');

% Define the first convolutional layer
conv1 = convolution2dLayer(5, 20, 'Name', 'conv1');

% Define the first pooling layer
pool1 = maxPooling2dLayer(2, 'Name', 'pool1');

% Define the second convolutional layer
conv2 = convolution2dLayer(5, 50, 'Name', 'conv2');

% Define the second pooling layer
pool2 = maxPooling2dLayer(2, 'Name', 'pool2');

% Define the fully connected layer
fc = fullyConnectedLayer(10, 'Name', 'fc');

% Define the output layer
output_layer = classificationLayer('Name', 'output');

% Combine the layers into a layer array
layers = [input_layer; conv1; pool1; conv2; pool2; fc; output_layer];

% Set the training options (e.g., number of epochs, mini-batch size)
options = trainingOptions('sgdm', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.01, ...
    'Verbose', true);

% Train the deep learning model
net = trainNetwork(train_data, train_labels, layers, options);

% Predict the labels for the test data
pred_labels = classify(net, test_data);

% Calculate the accuracy of the model
accuracy = mean(pred_labels == test_labels);
fprintf('Accuracy: %0.4f\n', accuracy);
