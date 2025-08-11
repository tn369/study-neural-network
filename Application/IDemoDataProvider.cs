namespace MultiLayerNet.Application;

/// <summary>
/// デモ/テスト用の (input, target) を供給する責務だけを持つ。
/// 返す形は「そのモデルのサービスが受け付ける形」に合わせる。
/// </summary>
public interface IDemoDataProvider
{
    ModelKind Kind { get; }

    /// <summary>
    /// 1サンプル返す。
    /// 例: CNNなら 8x8 画像を 64要素にフラット化して返す。
    /// </summary>
    (IReadOnlyList<double> Input, IReadOnlyList<double> Target) GetSample();
}
