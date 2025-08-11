namespace MultiLayerNet.Domain.Activations;

/// <summary>
/// シグモイド活性化関数。
/// 数式: σ(z) = 1/(1+e^(−z)),  σ′(z) = σ(z)·(1−σ(z))
/// </summary>
public sealed class Sigmoid : IActivationFunction
{
    public double Invoke(double preActivationZ) => 1.0 / (1.0 + Math.Exp(-preActivationZ));

    public double DerivativeAt(double preActivationZ)
    {
        var sigma = Invoke(preActivationZ);      // σ(z)
        return sigma * (1 - sigma);              // σ′(z)
    }
}