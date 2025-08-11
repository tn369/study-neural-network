using MultiLayerNet.Application;
using MultiLayerNet.Domain.Activations;
using MultiLayerNet.Domain.Cnn;
using NeuralNetwork.Infrastructure.Common;

namespace MultiLayerNet.Infrastructure.Services;

/// <summary>
/// 単純な CNN 実装。
/// 前向き: X(8×8) → Conv3×3(valid, stride=1, ReLU) → MaxPool2×2 → Flatten → FC(1出力, Sigmoid)
/// 損失: L = 1/2 (y − t)^2
/// </summary>
public sealed class CnnService : INeuralNetService
{
    public string Name => "CNN (Conv3x3 + MaxPool2 + FC1)";

    private readonly int _imageH = 8;
    private readonly int _imageW = 8;

    private readonly Conv2D _conv;
    private readonly MaxPool2D _pool;
    private readonly FullyConnected _fc;
    private readonly ILossFunction _loss;

    public CnnService(ReLU relu, Sigmoid sigmoid, ILossFunction loss, IRandomInitializer init)
    {
        _conv = new Conv2D(3, 3, relu, init.NextWeight, init.NextBias);
        _pool = new MaxPool2D();

        int convH = _imageH - 3 + 1, convW = _imageW - 3 + 1; // valid
        int poolH = convH / 2, poolW = convW / 2;
        int flattened = poolH * poolW;

        _fc = new FullyConnected(flattened, sigmoid, init.NextWeight, init.NextBias);
        _loss = loss;
    }

    public IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector)
    {
        var x = ToImage(inputVector);
        var y1 = _conv.Forward(x);
        var y2 = _pool.Forward(y1);
        var flat = Flatten(y2);
        var y = _fc.Forward(flat);
        return new[] { y };
    }

    public void Train(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate)
    {
        var x = ToImage(inputVector);
        var y1 = _conv.Forward(x);
        var y2 = _pool.Forward(y1);
        var flat = Flatten(y2);
        var y = _fc.Forward(flat);

        double diff = y - targetVector[0];                 // (y − t)
        var deltaFcIn = _fc.Backward(diff, learningRate);  // FC の逆伝播
        var deltaPool = Unflatten(deltaFcIn, y2.GetLength(0), y2.GetLength(1));
        var deltaBeforeConv = _pool.Backward(deltaPool);    // MaxPool 逆伝播
        _ = _conv.Backward(deltaBeforeConv, learningRate);  // Conv 逆伝播（入力側 δ は未使用）
    }

    public double CalcLoss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector)
        => _loss.Loss(outputVector, targetVector);

    private double[,] ToImage(IReadOnlyList<double> inputVector)
    {
        if (inputVector.Count != _imageH * _imageW)
            throw new ArgumentException($"CNN expects {_imageH * _imageW} inputs for {_imageH}x{_imageW} image.");
        var img = new double[_imageH, _imageW];
        int k = 0; for (int i = 0; i < _imageH; i++) for (int j = 0; j < _imageW; j++) img[i, j] = inputVector[k++];
        return img;
    }

    private static double[] Flatten(double[,] x)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        var v = new double[h * w]; int k = 0;
        for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) v[k++] = x[i, j];
        return v;
    }

    private static double[,] Unflatten(double[] v, int h, int w)
    {
        if (v.Length != h * w) throw new ArgumentException("shape mismatch");
        var x = new double[h, w]; int k = 0;
        for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) x[i, j] = v[k++];
        return x;
    }
}