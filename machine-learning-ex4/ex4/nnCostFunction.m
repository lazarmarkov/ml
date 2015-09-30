function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% since y is a [ 1 x `examples_count`]
% we need to translate it to a [`num_labels` x `examples_count`] vector
% where the `y`th item in the row has a value of 1.
y_vect = zeros(m, num_labels);
for i = 1:m
    y_vect(i,y(i)) = 1;
end

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% The h_result is a `m` x `num_labels` matrix.
% The matrix has `m` rows, where each row `i` represents
% the activation(output values) for the output layer for training example `i`.
h_result = zeros(m, num_labels);
for i = 1:m
    % get the activation for the input layer
    % (i.e. the grayscale pixel intensity in our case)
    a1 = [1 X(i,:)];
    
    % Theta1 [25 x 401] * a1' [401 x 1]
    % z [25 x 1]
    z2 = Theta1*a1';
    
    % compute the activation for the hidden layer
    % a2 [26 x 1] 
    a2 = [1; sigmoid(z2)];
    
    % Theta2 [10 x 26] * a2 [26 x 1]
    % z [10x1]
    z3 = Theta2*a2;
    
    % compute the activation for the output layer
    % a3 [10x1]
    a3 = sigmoid(z3);
    
    % store the activation result for training example `i`
    % in the `i`th row of the matrix
    h_result(i,:) = a3';
end

% the sum of all costs for all examples
all_ex_sum = 0;
for i = 1:m
    
    % the sum of the costs for example `i` 
    example_sum = 0;
    
    for k = 1:num_labels
        % compute the cost for each possible result(label) in the training example
        cost = -y_vect(i,k) * log(h_result(i,k)) - (1 - y_vect(i,k)) * log(1 - h_result(i,k));
        
        % sum the costs for all the labels
        example_sum = example_sum + cost;
    end
    
    %sum the cost for all training examples
    all_ex_sum = all_ex_sum + example_sum;
end

% get the unregulazied cost by dividing by the number of training examples.
J = all_ex_sum / m;

reg_sum = sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^2));
reg = lambda*reg_sum / (2*m);

J = J + reg;
% -------------------------------------------------------------

% =========================================================================

delta_2_acc = zeros(size(Theta1));
delta_3_acc = zeros(size(Theta2));

for t = 1:m
    
    % === get the activation for the input layer
    % (i.e. the grayscale pixel intensity in our case)
    a1 = [1 X(t,:)]';
    z2 = Theta1*a1;
    
    % === compute the activation for the hidden layer 
    a2 = [1; sigmoid(z2)];
    
    
    % === compute the delta error for the output layer 
    % do so by computing `actual` - `expected` for output layer.
    delta_3 = h_result(t,:)' - y_vect(t,:)';

    
    % === compute the delta error for the hidden layer  
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);
    % exclude the bias unit (+1)
    delta_2 = delta_2(2:end);
    
    % accumulate the hidden layer's delta error for all training examples
    delta_2_acc = delta_2_acc + delta_2 * a1';
    % accumulate the output layer's delta error for all training examples
    delta_3_acc = delta_3_acc + delta_3 * a2';
end


% ==== compute the gradient ====

% for the bias term, we don't regularize, so we just divide by m
Theta1_grad(:,1) = delta_2_acc(:,1) / m;
% for the rest of the terms in the network, we regularize:
Theta1_grad(:,2:end) = delta_2_acc(:,2:end) / m + (lambda/m)*Theta1(:,2:end);

Theta2_grad(:,1) = delta_3_acc(:,1) / m;
Theta2_grad(:,2:end) = delta_3_acc(:,2:end) / m + (lambda/m)*Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
