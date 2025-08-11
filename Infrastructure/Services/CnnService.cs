using MultiLayerNet.Application;
using MultiLayerNet.Domain.Activations;
using MultiLayerNet.Domain.Cnn;
using NeuralNetwork.Infrastructure.Common;

namespace MultiLayerNet.Infrastructure.Services;

/// <summary>
/// 単純な CNN サービス実装（Conv→ReLU→MaxPool→Flatten→FC→Sigmoid）。
/// 方式比較用に <see cref="INeuralNetService"/> を実装します。
///
/// 【数式の前提と記号】
/// - 入力画像:            𝑋 ∈ ℝ^{H×W}（ここでは H=W=8）
/// - 畳み込みフィルタ:    𝐾 ∈ ℝ^{kH×kW}（ここでは kH=kW=3）, バイアス: b
/// - 畳み込み（valid, stride=1）:
///     𝑍[i,j] = Σ_u Σ_v 𝑋[i+u, j+v] · 𝐾[u,v] + b
/// - 活性化（ReLU）:
///     𝑌 = φ(𝑍),  φ(z)=max(0,z),  φ′(z)=1(z>0),0(z≤0)
/// - 最大プーリング（2×2, stride=2）:
///     𝑌_pool[p,q] = max_{(i,j)∈window(p,q)} 𝑌[i,j]
/// - Flatten 後の全結合（1出力, Sigmoid）:
///     z = Σ_i w_i x_i + b,  y = σ(z),  σ(z)=1/(1+e^{−z})
/// - 損失（MSE/2）:
///     L = 1/2 · (y − t)^2
/// - 逆伝播の要点:
///     δ_FC = ∂L/∂z = (y − t) · σ′(z)
///     ∂L/∂w_i = δ_FC · x_i,   ∂L/∂b = δ_FC,   δ_prev_i = δ_FC · w_i
///     プーリングは “最大値だった位置にのみ” 勾配を流す
///     Conv は δZ = δ ⊙ φ′(Z) を用い、∂L/∂K, ∂L/∂b, ∂L/∂X を計算
/// </summary>
public sealed class CnnService : INeuralNetService
{
    public string Name => "CNN (Conv3x3 + MaxPool2 + FC1)";

    // 入力画像サイズ（デモ用に固定: 8×8, 1ch）
    private readonly int _imageHeight = 8;
    private readonly int _imageWidth = 8;

    // レイヤ構成（Domain.Cnn に分離したミニマルな演算実装を利用）
    private readonly Conv2D _convolutionLayer;          // 3×3 畳み込み + ReLU
    private readonly MaxPool2D _maxPoolingLayer;        // 2×2 最大プーリング
    private readonly FullyConnected _fullyConnectedLayer; // Flatten → 1出力 + Sigmoid

    private readonly ILossFunction _lossFunction;       // L = 1/2 (y − t)^2

    /// <summary>
    /// 依存の注入:
    /// - <paramref name="relu"/> は Conv 後の活性 φ=ReLU
    /// - <paramref name="sigmoid"/> は FC の出力活性 σ=Sigmoid
    /// - <paramref name="loss"/> は損失関数 L（MSE/2）
    /// - <paramref name="initializer"/> は重み・バイアス初期化（w, b）
    /// </summary>
    public CnnService(ReLU relu, Sigmoid sigmoid, ILossFunction loss, IRandomInitializer initializer)
    {
        // --- Conv の構築 ---
        // 出力サイズ: conv(valid,3x3) → (H-2)×(W-2)
        _convolutionLayer = new Conv2D(
            kernelHeight: 3,
            kernelWidth: 3,
            activation: relu,
            weightInit: initializer.NextWeight,
            biasInit: initializer.NextBias
        );

        _maxPoolingLayer = new MaxPool2D(); // 2×2 stride=2

        // --- FC の入力次元を計算して構築 ---
        int convOutH = _imageHeight - 3 + 1;
        int convOutW = _imageWidth - 3 + 1;
        int pooledH = convOutH / 2;
        int pooledW = convOutW / 2;
        int flattenedLength = pooledH * pooledW; // Flatten 後の次元数 N

        _fullyConnectedLayer = new FullyConnected(
            inputSizeN: flattenedLength,
            activation: sigmoid,
            weightInit: initializer.NextWeight,
            biasInit: initializer.NextBias
        );

        _lossFunction = loss;
    }

    /// <summary>
    /// 推論（順伝播）: x → Conv→ReLU → MaxPool → Flatten → FC→Sigmoid → y
    /// 数式：
    ///   1) 𝑍 = 𝑋 * 𝐾 + b,  2) 𝑌 = φ(𝑍),
    ///   3) 𝑌_pool = MaxPool(𝑌), 4) x_flat = Flatten(𝑌_pool),
    ///   5) z = Σ_i w_i x_i + b,  y = σ(z)
    /// </summary>
    public IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector)
    {
        // 入力ベクトル（長さ H×W）を 2次元画像 X(H×W) に整形
        var inputImage = ToImage(inputVector);

        // Conv + ReLU
        var afterConvolution = _convolutionLayer.Forward(inputImage);

        // MaxPool
        var afterPooling = _maxPoolingLayer.Forward(afterConvolution);

        // Flatten → FC + Sigmoid
        var flattenedVector = Flatten(afterPooling);
        var outputScalar = _fullyConnectedLayer.Forward(flattenedVector);

        return new[] { outputScalar };
    }

    /// <summary>
    /// 学習（1サンプルの誤差逆伝播）:
    ///  1) 順伝播で y を得る
    ///  2) 出力層の (y − t) を計算
    ///  3) FC → Pool → Conv の順で逆伝播し、各パラメータを更新
    /// 数式（要所）:
    ///  - FC: δ_FC = (y − t) · σ′(z),  ∂L/∂w_i = δ_FC · x_i,  ∂L/∂b = δ_FC
    ///  - MaxPool: 勾配は “最大値だった位置” にのみ伝播
    ///  - Conv: δZ = δ ⊙ φ′(Z),  ∂L/∂K[u,v] = Σ_i Σ_j X[i+u, j+v] · δZ[i,j],  ∂L/∂b = Σ δZ
    /// </summary>
    public void Train(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate)
    {
        // --- 順伝播 ---
        var inputImage = ToImage(inputVector);
        var afterConvolution = _convolutionLayer.Forward(inputImage);
        var afterPooling = _maxPoolingLayer.Forward(afterConvolution);
        var flattenedVector = Flatten(afterPooling);
        var outputScalar = _fullyConnectedLayer.Forward(flattenedVector);

        // 出力層の “(y − t)”
        double differenceYMinusT = outputScalar - targetVector[0];

        // --- 逆伝播（FC → Pool → Conv） ---
        // FC の逆伝播: δ を前段（Flatten 後ベクトル）へ返すとともに w, b 更新
        var deltaIntoFlattenVector = _fullyConnectedLayer.Backward(differenceYMinusT, learningRate);

        // Flatten の逆変換でプール出力の形状に戻す
        var deltaOnPooledFeatureMap = Unflatten(
            vector: deltaIntoFlattenVector,
            height: afterPooling.GetLength(0),
            width: afterPooling.GetLength(1)
        );

        // MaxPool の逆伝播（最大値位置にのみ勾配を返す）
        var deltaBeforeConvolution = _maxPoolingLayer.Backward(deltaOnPooledFeatureMap);

        // Conv の逆伝播（ReLU の導関数を掛け、K と b を更新）
        _ = _convolutionLayer.Backward(deltaBeforeConvolution, learningRate);
    }

    /// <summary>
    /// 損失 L = 1/2 · (y − t)^2 を返す。
    /// </summary>
    public double CalcLoss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector)
        => _lossFunction.Loss(outputVector, targetVector);

    /// <summary>
    /// 1次元ベクトル（長さ H×W）を 2次元画像 X(H×W) に整形。
    /// 入力が 64 要素でない場合は例外。
    /// </summary>
    private double[,] ToImage(IReadOnlyList<double> inputVector)
    {
        if (inputVector.Count != _imageHeight * _imageWidth)
            throw new ArgumentException($"CNN expects {_imageHeight * _imageWidth} inputs for {_imageHeight}x{_imageWidth} image.");

        var image = new double[_imageHeight, _imageWidth];
        int k = 0;
        for (int i = 0; i < _imageHeight; i++)
            for (int j = 0; j < _imageWidth; j++)
                image[i, j] = inputVector[k++];

        return image;
    }

    /// <summary>
    /// 2次元配列を行優先で 1次元ベクトルに Flatten。
    /// </summary>
    private static double[] Flatten(double[,] matrix)
    {
        int height = matrix.GetLength(0), width = matrix.GetLength(1);
        var vector = new double[height * width];
        int k = 0;
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                vector[k++] = matrix[i, j];
        return vector;
    }

    /// <summary>
    /// Flatten の逆（ベクトル → 行列）。
    /// </summary>
    private static double[,] Unflatten(double[] vector, int height, int width)
    {
        if (vector.Length != height * width)
            throw new ArgumentException("shape mismatch");

        var matrix = new double[height, width];
        int k = 0;
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                matrix[i, j] = vector[k++];
        return matrix;
    }
}
