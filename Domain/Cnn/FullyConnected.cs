using MultiLayerNet.Domain.Activations;

namespace MultiLayerNet.Domain.Cnn;

/// <summary>
/// 全結合層（Flatten ベクトル x ∈ R^N → y ∈ R^1）。
/// 数式:
///   z = Σ_i w_i x_i + b,  y = σ(z)
///   δ = ∂L/∂z = ∂L/∂y · σ′(z)
/// 勾配降下:
///   ∂L/∂w_i = δ · x_i,  ∂L/∂b = δ,  δ_prev_i = δ · w_i
/// </summary>
public sealed class FullyConnected
{
    private readonly double[] _weightVector; // w_i
    private double _bias;                    // b
    private readonly IActivationFunction _activation; // σ

    private double[] _lastInputX = default!; // x
    private double _lastPreZ;                // z

    public FullyConnected(int inputSizeN, IActivationFunction activation, Func<double> weightInit, Func<double> biasInit)
    {
        _activation = activation;
        _weightVector = new double[inputSizeN];
        for (int i = 0; i < inputSizeN; i++) _weightVector[i] = weightInit();
        _bias = biasInit();
    }

    public double Forward(double[] inputX)
    {
        _lastInputX = (double[])inputX.Clone();
        double z = 0; for (int i = 0; i < inputX.Length; i++) z += inputX[i] * _weightVector[i];
        z += _bias; _lastPreZ = z;
        return _activation.Invoke(z);
    }

    public double[] Backward(double incomingDeltaY, double learningRate)
    {
        double deltaZ = incomingDeltaY * _activation.DerivativeAt(_lastPreZ);
        var deltaX = new double[_lastInputX.Length];
        for (int i = 0; i < _weightVector.Length; i++)
        {
            deltaX[i] = deltaZ * _weightVector[i];
            _weightVector[i] -= learningRate * (deltaZ * _lastInputX[i]);
        }
        _bias -= learningRate * deltaZ;
        return deltaX;
    }
}