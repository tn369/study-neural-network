using MultiLayerNet.Application;

namespace MultiLayerNet.Infrastructure.Datasets;

/// <summary>
/// CNN向けデモデータ：8x8の単一チャネル画像を 64 要素にフラット化して返す。
/// 数式: 画像 X ∈ R^{H×W}（ここでは H=W=8）
/// フラット化: x_flat[k] = X[i,j] （i,j を1次元に並べ替え）
/// 目標 t: 1出力（Sigmoid）を想定し t ∈ [0,1]
/// </summary>
public sealed class CnnDemoDataProvider : IDemoDataProvider
{
    public ModelKind Kind => ModelKind.Cnn;

    public (IReadOnlyList<double> Input, IReadOnlyList<double> Target) GetSample()
    {
        // 例：簡単なグラデーション画像（0..1）
        var imgFlat = new List<double>(64);
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                imgFlat.Add((i + j) / 14.0);

        var t = new List<double> { 1.0 }; // 例として 1.0 を目標に
        return (imgFlat, t);
    }
}
