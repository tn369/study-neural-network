using MultiLayerNet.Domain.Activations;

namespace NeuralNetwork.Domain.Losses;

/// <summary>
/// 平均二乗誤差（半分）
/// 数式: L = 1/2 Σ_k (y_k − t_k)²
/// </summary>
public sealed class MeanSquaredError : ILossFunction
{
    public double Loss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector)
    {
        double sumOfSquaredErrors = 0;
        for (int k = 0; k < outputVector.Count; k++)
        {
            double difference = outputVector[k] - targetVector[k]; // (y_k − t_k)
            sumOfSquaredErrors += Math.Pow(difference, 2);         // (y_k − t_k)²
        }
        return 0.5 * sumOfSquaredErrors; // 1/2 Σ (y_k − t_k)²
    }
}