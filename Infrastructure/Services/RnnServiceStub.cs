using MultiLayerNet.Application;

namespace MultiLayerNet.Infrastructure.Services;

/// <summary>
/// RNN スタブ。
/// 数式例（単純RNN）: h_t = σ(W_xh x_t + W_hh h_{t-1} + b_h),  y_t = φ(W_hy h_t + b_y)
/// </summary>
public sealed class RnnServiceStub : INeuralNetService
{
    public string Name => "RNN (stub)";
    public IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector) => inputVector; // TODO: 時系列テンソル対応へ
    public void Train(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate) { /* TODO */ }
    public double CalcLoss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector) => 0; // TODO
}