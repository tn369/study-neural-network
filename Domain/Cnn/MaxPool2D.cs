namespace MultiLayerNet.Domain.Cnn;

/// <summary>
/// 最大プーリング（2×2, stride=2）。
/// 前向き: Y[p,q] = max_{(i,j)∈window(p,q)} X[i,j]
/// 逆伝播: 勾配は最大値の位置だけに流す。
/// </summary>
public sealed class MaxPool2D
{
    private int _inH, _inW;
    private int[,] _argMaxI = default!;
    private int[,] _argMaxJ = default!;

    public double[,] Forward(double[,] inputX)
    {
        _inH = inputX.GetLength(0); _inW = inputX.GetLength(1);
        int oh = _inH / 2, ow = _inW / 2;
        var Y = new double[oh, ow];
        _argMaxI = new int[oh, ow];
        _argMaxJ = new int[oh, ow];

        for (int p = 0; p < oh; p++)
            for (int q = 0; q < ow; q++)
            {
                int i0 = p * 2, j0 = q * 2;
                double max = double.NegativeInfinity; int mi = i0, mj = j0;
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                    {
                        double v = inputX[i0 + di, j0 + dj];
                        if (v > max) { max = v; mi = i0 + di; mj = j0 + dj; }
                    }
                Y[p, q] = max; _argMaxI[p, q] = mi; _argMaxJ[p, q] = mj;
            }
        return Y;
    }

    public double[,] Backward(double[,] incomingDeltaY)
    {
        var deltaX = new double[_inH, _inW];
        int oh = incomingDeltaY.GetLength(0), ow = incomingDeltaY.GetLength(1);
        for (int p = 0; p < oh; p++)
            for (int q = 0; q < ow; q++)
                deltaX[_argMaxI[p, q], _argMaxJ[p, q]] += incomingDeltaY[p, q];
        return deltaX;
    }
}