namespace MultiLayerNet.Application;

/// <summary>
/// 方式比較のための共通インターフェース。
/// 数式上は、入力ベクトル x を与えると出力ベクトル y を返し、
/// 損失 L(x, t) と勾配に基づく 1 サンプル学習を提供します。
/// </summary>
public interface INeuralNetService
{
    string Name { get; }

    /// <summary>
    /// 推論（順伝播）: y = f(x)
    /// 数式: z = W x + b（層ごとに Σᵢ wᵢ·xᵢ + b）、y = σ(z)
    /// </summary>
    IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector);

    /// <summary>
    /// 1サンプル学習（オンライン学習）
    /// 数式（勾配降下）: wᵢ ← wᵢ − η · (δ · xᵢ),  b ← b − η · δ
    /// </summary>
    void Train(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate);

    /// <summary>
    /// 損失 L を返す。デフォルトは L = 1/2 Σ_k (y_k − t_k)²（MSE/2）。
    /// </summary>
    double CalcLoss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector);
}