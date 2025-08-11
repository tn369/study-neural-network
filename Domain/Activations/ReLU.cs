namespace MultiLayerNet.Domain.Activations;

/// <summary>
/// ReLU（Rectified Linear Unit）
/// 数式: ReLU(z) = max(0, z)
/// 導関数: ReLU′(z) = 1 (z>0), 0 (z≤0)
/// </summary>
public sealed class ReLU : IActivationFunction
{
    public double Invoke(double preActivationZ) => preActivationZ > 0 ? preActivationZ : 0.0;
    public double DerivativeAt(double preActivationZ) => preActivationZ > 0 ? 1.0 : 0.0;
}