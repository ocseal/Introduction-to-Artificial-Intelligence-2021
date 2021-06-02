import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        self.hidden_size = 500
        self.numTrainingGames = 3000
        self.batch_size = 50
        self.learning_rate = 0.5
        self.w1 = nn.Parameter(state_dim, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, int(self.hidden_size/5))
        self.b2 = nn.Parameter(1, int(self.hidden_size/5))
        self.w3 = nn.Parameter(int(self.hidden_size/5), int(self.hidden_size/100))
        self.b3 = nn.Parameter(1, action_dim)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        s_w1 = nn.Linear(states, self.w1)
        relu_input = nn.AddBias(s_w1, self.b1)
        relu = nn.ReLU(relu_input)
        relu_b2 = nn.Linear(relu, self.w2)
        l1l2 = nn.AddBias(relu_b2, self.b2)
        relu_l3 = nn.ReLU(l1l2)
        l3_w3 = nn.Linear(relu_l3, self.w3)
        sol = nn.AddBias(l3_w3, self.b3)
        return sol

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        params = [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, params)
        self.w1.update(gradients[0], -self.learning_rate)
        self.w2.update(gradients[1], -self.learning_rate)
        self.w3.update(gradients[2], -self.learning_rate)
        self.b1.update(gradients[3], -self.learning_rate)
        self.b2.update(gradients[4], -self.learning_rate)
        self.b3.update(gradients[5], -self.learning_rate)
