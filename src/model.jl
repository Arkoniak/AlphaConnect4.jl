abstract type AbstractModel end

struct ResidualCNNMX <: AbstractModel
  optimizer
  model::MXNet.mx.SymbolicNode
end
function ResidualCNNMX(hidden_layers, learning_rate, weight_decay, momentum, output_dim)
  main_input = mx.Variable(:main_input)

  x = conv_layer(main_input, hidden_layers[1][:filters], hidden_layers[1][:kernel_size])

  if length(hidden_layers) > 1
    for i in 2:length(hidden_layers)
      x = residual_layer(x, hidden_layers[i][:filters], hidden_layers[i][:kernel_size])
    end
  end

  vh = value_head(x)
  ph = policy_head(x, output_dim)

  # Somehow combine two outputs together
  grp = mx.Group(vh, ph)
  model = mx.FeedForward(grp)
  optimizer = mx.SGD(lr = learning_rate, weight_decay = weight_decay, momentum = momentum)
  ResidualCNNMX(optimizer, model)
end

function predict(m::ResidualCNNMX, x)
  mx.predict(m.model, x)
end

function fit(m::ResidualCNNMX, states, target, batch_size, num_epoch, validation_split)
  # Prepare train/eval providers
  train_provider = ???
  eval_provider = ???
  mx.fit(m.model, m.optimizer, train_provider, n_epoch = num_epoch, eval_data = eval_provider)
end

function residual_layer(m::MXNet.mx.SymbolicNode, filters, kernel_size)
  res = @mx.chain conv_layer(m, filters, kernel_size)
            mx.Convolution(m1, no_bias = true, num_filters = filters, kernel = kernel_size)
            mx.BatchNorm()
  res = res + m
  res = mx.Activation(res, act_type = :relu)
end

function conv_layer(m::MXNet.mx.SymbolicNode, filters, kernel_size)
  @mx.chain mx.Convolution(m, no_bias = true, num_filter = filters, kernel = kernel_size)
            mx.BatchNorm()
            mx.Activation(act_type = :relu)    # No leaky relu in current version of MXNet
end

function value_head(m::MXNet.mx.SymbolicNode)
  @mx.chain mx.Convolution(m, num_filters = 1, kernel = (1, 1), no_bias = true)
            mx.BatchNorm()
            mx.Activation(act_type = :relu)
            mx.Flatten()
            mx.FullyConnected(num_hidden = 20, no_bias = true)
            mx.Activation(act_type = :relu)
            mx.FullyConnected(num_hidden = 1, no_bias = true)
            mx.Activation(act_type = :tanh)
            mx.LinearRegressionOutput()
end

function policy_head(m::MXNet.mx.SymbolicNode, output_dim)
  @mx.chain mx.Convolution(m, filters = 2, kernel = (1, 1), no_bias = true)
            mx.BatchNorm()
            mx.Activation(act_type = :leaky_relu)
            mx.Flatten()
            mx.FullyConnected(output_dim)
            mx.SoftmaxOutput()
end
