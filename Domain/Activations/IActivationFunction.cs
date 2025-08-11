namespace MultiLayerNet.Domain.Activations;

/// <summary>
/// 活性化関数 σ とその導関数 σ′ を表す抽象。
/// 例: シグモイド σ(z) = 1/(1+e^(−z)), σ′(z) = σ(z)(1−σ(z))
/// </summary>
public interface IActivationFunction
{
    double Invoke(double preActivationZ);        // y = σ(z)
    double DerivativeAt(double preActivationZ);  // σ′(z)
}