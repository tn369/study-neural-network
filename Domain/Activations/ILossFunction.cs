namespace MultiLayerNet.Domain.Activations;


/// <summary>
/// 損失関数 L。デフォルト実装は二乗和誤差の半分: L = 1/2 Σ_k (y_k − t_k)²
/// </summary>
public interface ILossFunction
{
    double Loss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector);
}