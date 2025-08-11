namespace MultiLayerNet.Domain.Mlp;

/// <summary>
/// 全結合のみの前向きネットワーク（MLPの核）。
/// </summary>
public sealed class FeedForwardNetwork
{
    private readonly List<Layer> _layers;

    public FeedForwardNetwork(IReadOnlyList<Layer> layers)
    {
        if (layers == null || layers.Count == 0) throw new ArgumentException("layers is empty");
        _layers = new List<Layer>(layers);
    }

    /// <summary>
    /// 推論（順伝播）: 各層で y = σ(Wx+b) を適用し、最終出力を返す。
    /// </summary>
    public IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector)
    {
        IReadOnlyList<double> currentVector = inputVector;
        foreach (var layer in _layers)
            currentVector = layer.FeedForward(currentVector);
        return currentVector;
    }

    /// <summary>
    /// 1サンプル学習（誤差逆伝播）。
    /// 手順:
    ///   1) 順伝播で最終出力 y を得る
    ///   2) 出力層の “(y−t)” を作る（後で各層で σ′(z) を掛けて δ を得る）
    ///   3) 層を逆順に辿り BackPropagate（隠れ層は Σ w·δ を計算しつつ更新）
    /// </summary>
    public void TrainOne(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate)
    {
        var outputVector = Predict(inputVector); // y

        // 出力層の “(y − t)” を準備（ここでは σ′(z) は各レイヤで掛ける）
        var deltaVector = new double[outputVector.Count];
        for (int k = 0; k < outputVector.Count; k++)
            deltaVector[k] = outputVector[k] - targetVector[k]; // (y_k − t_k)

        // 逆順に BackPropagate（各層で δ = incoming * σ′(z) を作って更新）
        for (int layerIndex = _layers.Count - 1; layerIndex >= 0; layerIndex--)
            deltaVector = (double[])_layers[layerIndex].BackPropagate(deltaVector, learningRate);
    }
}