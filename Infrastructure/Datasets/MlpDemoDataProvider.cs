using MultiLayerNet.Application;

namespace MultiLayerNet.Infrastructure.Datasets;

/// <summary>
/// MLP向けデモデータ：固定長ベクトル x と目標 t
/// 数式: 入力ベクトル x = (x1,x2,x3), 目標 t = (t1)
/// </summary>
public sealed class MlpDemoDataProvider : IDemoDataProvider
{
    public ModelKind Kind => ModelKind.Mlp;

    public (IReadOnlyList<double> Input, IReadOnlyList<double> Target) GetSample()
    {
        var x = new List<double> { 1.0, 0.5, -1.2 }; // x
        var t = new List<double> { 0.8 };            // t
        return (x, t);
    }
}
