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

X_temp = [ones(m, 1) X];
z1 = X_temp * Theta1';
a1 = sigmoid(z1);

m2 = size(a1, 1);
a1 = [ones(m2, 1) a1];

H = a1 * Theta2';
a2 = sigmoid(H);


for i = 1:m,
	label = y(i);
	for k = 1:num_labels,
		y_val = 0;
		if label == k,
			y_val = 1;
		else,
			y_val = 0;
		end;
		first = y_val * log(a2(i, k));
		second = (1 - y_val) * log(1 - a2(i, k));
		J = J + first + second;
	end;
end;


J = -J / m;

start1 = hidden_layer_size + 1;
start2 = num_labels + 1;
Reg = sum(Theta1(start1:end) .^ 2) + sum(Theta2(start2:end) .^ 2);
Reg = Reg * lambda/(2*m);

J = J + Reg;

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

for t = 1:m,
	% Step 1: Perform Forward Propagation
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	a1 = X(t, :);
	a1 = a1(:);
	a1 = [1; a1];

	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);


	% Step 2: Find delta3
	%%%%%%%%%%%%%%%%%%%%%%%%%%%
	delta3 = zeros(num_labels, 1);
	
	label = y(t);
	for k = 1:num_labels,
		y_val = 0;
		if label == k,
			y_val = 1;
		else,
			y_val = 0;
		end;
		delta3(k) = a3(k) - y_val;
	end;

	% Step 3: Find delta2
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	delta2 = (Theta2') * delta3;
	delta2 = delta2(2:end);
	delta2 = delta2 .* sigmoidGradient(z2);

	% Step 4: Accumulate gradient
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Theta1_grad = Theta1_grad + (delta2 * a1');
	Theta2_grad = Theta2_grad + (delta3 * a2');

	
end;



% Step 5: obtain gradients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set bias terms gradient to 1'

m = size(X, 1);

%Theta1_grad(:, 1) = zeros(hidden_layer_size, 1);
%Theta2_grad(:, 1) = zeros(num_labels, 1);

Theta1_grad = Theta1_grad .* 1/m;
Theta2_grad = Theta2_grad .* 1/m;




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Reg1 = lambda/m * Theta1;
Reg2 = lambda/m * Theta2;

temp_grad1 = Theta1_grad(:, 1);
temp_grad2 = Theta2_grad(:, 1);

Theta1_grad = Theta1_grad + Reg1;
Theta2_grad = Theta2_grad + Reg2;

Theta1_grad(:, 1) = temp_grad1;
Theta2_grad(:, 1) = temp_grad2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
