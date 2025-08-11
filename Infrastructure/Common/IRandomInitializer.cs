namespace NeuralNetwork.Infrastructure.Common;

/// <summary>
/// 重み・バイアスの初期化戦略。
/// ここでは簡易に w ∈ [-0.5, 0.5), b = 0 を返す。
/// （将来は Xavier/He 初期化などに差し替え可能）
/// </summary>
public interface IRandomInitializer
{
    double NextWeight(); // 初期重み w
    double NextBias();   // 初期バイアス b
}
