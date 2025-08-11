using MultiLayerNet.Domain.Activations;

namespace MultiLayerNet.Domain.Cnn;

/// <summary>
/// 2D 畳み込み層（最小実装）。
/// 数式:
///   入力画像 X ∈ R^{H×W}, カーネル K ∈ R^{kH×kW}, バイアス b
///   前活性 Z[i,j] = Σ_u Σ_v X[i+u, j+v] · K[u,v] + b  （valid, stride=1）
///   出力 Y = φ(Z)（φ: 活性化関数, ここでは ReLU）
/// 逆伝播（δ は出力 Y に対する損失の偏微分）:
///   δZ = δ ⊙ φ′(Z)
///   ∂L/∂K[u,v] = Σ_i Σ_j X[i+u, j+v] · δZ[i,j]
///   ∂L/∂b      = Σ_i Σ_j δZ[i,j]
///   ∂L/∂X[a,b] = Σ_u Σ_v δZ[a−u, b−v] · K[u,v]
/// </summary>
public sealed class Conv2D
{
    private readonly int _kernelH;
    private readonly int _kernelW;
    private readonly IActivationFunction _activation;

    private readonly double[,] _kernel; // K[u,v]
    private double _bias;               // b

    private double[,] _lastInputX = default!;  // X
    private double[,] _lastPreZ = default!;  // Z
    private double[,] _lastY = default!;  // Y

    public Conv2D(int kernelH, int kernelW, IActivationFunction activation, Func<double> weightInit, Func<double> biasInit)
    {
        _kernelH = kernelH; _kernelW = kernelW; _activation = activation;
        _kernel = new double[kernelH, kernelW];
        for (int u = 0; u < kernelH; u++)
            for (int v = 0; v < kernelW; v++)
                _kernel[u, v] = weightInit();
        _bias = biasInit();
    }

    public double[,] Forward(double[,] inputX)
    {
        _lastInputX = (double[,])inputX.Clone();
        int h = inputX.GetLength(0), w = inputX.GetLength(1);
        int oh = h - _kernelH + 1, ow = w - _kernelW + 1;
        var Z = new double[oh, ow];
        for (int i = 0; i < oh; i++)
            for (int j = 0; j < ow; j++)
            {
                double sum = 0;
                for (int u = 0; u < _kernelH; u++)
                    for (int v = 0; v < _kernelW; v++)
                        sum += inputX[i + u, j + v] * _kernel[u, v];
                Z[i, j] = sum + _bias;
            }
        _lastPreZ = Z;
        var Y = new double[oh, ow];
        for (int i = 0; i < oh; i++)
            for (int j = 0; j < ow; j++)
                Y[i, j] = _activation.Invoke(Z[i, j]);
        _lastY = Y;
        return Y;
    }

    public double[,] Backward(double[,] incomingDeltaY, double learningRate)
    {
        int oh = _lastY.GetLength(0), ow = _lastY.GetLength(1);
        var deltaZ = new double[oh, ow];
        for (int i = 0; i < oh; i++)
            for (int j = 0; j < ow; j++)
                deltaZ[i, j] = incomingDeltaY[i, j] * _activation.DerivativeAt(_lastPreZ[i, j]);

        var gradK = new double[_kernelH, _kernelW];
        double gradB = 0.0;
        for (int u = 0; u < _kernelH; u++)
            for (int v = 0; v < _kernelW; v++)
            {
                double s = 0;
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                        s += _lastInputX[i + u, j + v] * deltaZ[i, j];
                gradK[u, v] = s;
            }
        for (int i = 0; i < oh; i++)
            for (int j = 0; j < ow; j++) gradB += deltaZ[i, j];

        int h = _lastInputX.GetLength(0), w = _lastInputX.GetLength(1);
        var deltaX = new double[h, w];
        for (int a = 0; a < h; a++)
            for (int b = 0; b < w; b++)
            {
                double s = 0;
                for (int u = 0; u < _kernelH; u++)
                    for (int v = 0; v < _kernelW; v++)
                    {
                        int i = a - u; int j = b - v;
                        if (i >= 0 && j >= 0 && i < oh && j < ow)
                            s += deltaZ[i, j] * _kernel[u, v];
                    }
                deltaX[a, b] = s;
            }

        for (int u = 0; u < _kernelH; u++)
            for (int v = 0; v < _kernelW; v++)
                _kernel[u, v] -= learningRate * gradK[u, v];
        _bias -= learningRate * gradB;

        return deltaX;
    }
}