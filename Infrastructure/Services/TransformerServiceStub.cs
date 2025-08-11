using MultiLayerNet.Application;

namespace MultiLayerNet.Infrastructure.Services;

/// <summary>
/// Transformer スタブ。
/// 数式例（Scaled Dot-Product Attention）:
///   Attention(Q,K,V) = softmax(QK^T / √d_k) V
/// </summary>
public sealed class TransformerServiceStub : INeuralNetService
{
    public string Name => "Transformer (stub)";
    public IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector) => inputVector; // TODO: シーケンス/埋め込み対応へ
    public void Train(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate) { /* TODO */ }
    public double CalcLoss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector) => 0; // TODO
}