import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        output = nn.as_scalar(self.run(x))
        if output >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        done = False
        while not done:
            done = True
            for x, y in dataset.iterate_once(batch_size=1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    done = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.hidden_size = 100
        self.learning_rate = 0.01
        self.batch_size = 50
        self.w1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)
        self.loss_threshold = 0.02
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x_w1 = nn.Linear(x, self.w1)
        relu_input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(relu_input)
        relu_b2 = nn.Linear(relu, self.w2)
        sol = nn.AddBias(relu_b2, self.b2)
        return sol

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        done = False
        loss_list = []
        params = [self.w1, self.w2, self.b1, self.b2]
        while not done:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                loss_list.append(nn.as_scalar(loss))
                gradients = nn.gradients(loss, params)
                self.w1.update(gradients[0], -self.learning_rate)
                self.w2.update(gradients[1], -self.learning_rate)
                self.b1.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            if sum(loss_list)/len(loss_list) <= self.loss_threshold:
                done = True

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.hidden_size = 100
        self.learning_rate = 0.01
        self.batch_size = 50
        self.w1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, int(self.hidden_size/2))
        self.b2 = nn.Parameter(1, int(self.hidden_size/2))
        self.w3 = nn.Parameter(int(self.hidden_size/2), int(self.hidden_size/10))
        self.b3 = nn.Parameter(1, int(self.hidden_size/10))
        self.acc_threshold = 0.975

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x_w1 = nn.Linear(x, self.w1)
        relu_input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(relu_input)
        relu_b2 = nn.Linear(relu, self.w2)
        l1l2 = nn.AddBias(relu_b2, self.b2)
        relu_l3 = nn.ReLU(l1l2)
        l3_w3 = nn.Linear(relu_l3, self.w3)
        sol = nn.AddBias(l3_w3, self.b3)
        return sol

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        done = False
        params = [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]
        while not done:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, params)
                self.w1.update(gradients[0], -self.learning_rate)
                self.w2.update(gradients[1], -self.learning_rate)
                self.w3.update(gradients[2], -self.learning_rate)
                self.b1.update(gradients[3], -self.learning_rate)
                self.b2.update(gradients[4], -self.learning_rate)
                self.b3.update(gradients[5], -self.learning_rate)
            if dataset.get_validation_accuracy() >= self.acc_threshold:
                done = True


