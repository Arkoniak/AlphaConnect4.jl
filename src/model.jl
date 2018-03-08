abstract type AbstractModel end

struct ResidualCNNMX <: AbstractModel
  model::MXNet
end

function residual_layer(m::MXNet)
  res = @mx.chain mx.Convolution(m, ???)
            mx.BatchNorm()
  res = res + m
  res = mx.Activation(res, act_type = :leaky_relu)
end

function conv_layer(m::MXNet, filters, kernel_size)
  @mx.chain mx.Convolution(m, no_bias = true, num_filter = filters, kernel = kernel_size)
            mx.BatchNorm()
            mx.Activation(act_type = :relu)    # No leaky relu in current version of MXNet
end

function value_head(m::MXNet)
  @mx.chain mx.Convolution(m, ???)
            mx.BatchNorm()
            mx.Activation(act_type = :leaky_relu)
            mx.Flatten()
            mx.FullyConnected()
            mx.Activation(act_type = :leaky_relu)
            mx.FullyConnected()
            mx.Activation(act_type = :tanh)
end

function policy_head(m::MXNet)
  @mx.chain mx.Convolution(m, )
            mx.BatchNorm()
            mx.Activation(act_type = :leaky_relu)
            mx.Flatten()
            mx.FullyConnected()
end
