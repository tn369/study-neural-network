using MultiLayerNet.Domain.Activations;

namespace MultiLayerNet.Domain.Mlp;

/// <summary>
/// 単一ニューロン。
/// 数式上の対応:
///   入力ベクトル x = (x₁, …, xₙ), 重みベクトル w = (w₁, …, wₙ), バイアス b
///   前活性 z = Σᵢ (wᵢ · xᵢ) + b
///   出力 y = σ(z)
/// 逆伝播では、
///   出力層:  δ = (y − t) · σ′(z)
///   隠れ層:  δ = (Σ_j w_next_{i,j} · δ_next_j) · σ′(z)
/// 勾配降下:
///   wᵢ ← wᵢ − η · (δ · xᵢ),  b ← b − η · δ
/// </summary>
public sealed class Neuron
{
    private readonly List<double> _weightVector; // w₁..wₙ
    private double _bias;                        // b
    private readonly IActivationFunction _activationFunction; // σ, σ′

    // 逆伝播用に保持（直近の順伝播値）
    public List<double> LastInputVector { get; private set; } = new();   // x
    public double LastPreActivationZ { get; private set; }               // z
    public double LastOutputY { get; private set; }                      // y=σ(z)

    public Neuron(IReadOnlyList<double> initialWeightVector, double initialBias, IActivationFunction activationFunction)
    {
        if (initialWeightVector == null || initialWeightVector.Count == 0) throw new ArgumentException("weights is empty");
        _weightVector = new List<double>(initialWeightVector);
        _bias = initialBias;
        _activationFunction = activationFunction;
    }

    /// <summary>
    /// 順伝播（y = σ(z) を返す）。
    /// 数式: z = Σᵢ (wᵢ · xᵢ) + b,  y = σ(z)
    /// </summary>
    public double FeedForward(IReadOnlyList<double> inputVector)
    {
        if (inputVector.Count != _weightVector.Count) throw new ArgumentException("input size mismatch");
        LastInputVector = new List<double>(inputVector); // x を保持

        double preActivationZ = 0; // z
        for (int i = 0; i < inputVector.Count; i++)
            preActivationZ += inputVector[i] * _weightVector[i]; // Σ wᵢ·xᵢ
        preActivationZ += _bias; // + b

        LastPreActivationZ = preActivationZ;            // z
        LastOutputY = _activationFunction.Invoke(preActivationZ); // y = σ(z)
        return LastOutputY;
    }

    /// <summary>
    /// 逆伝播: このニューロンの δ を受けて、重みとバイアスを更新し、
    /// 前段（入力側）へ伝える δ（= δ · w）を返します。
    /// 数式: wᵢ ← wᵢ − η · (δ · xᵢ),  b ← b − η · δ
    /// 返却: δ_prev_i = δ · wᵢ（更新前の wᵢ を用いる）
    /// </summary>
    public IReadOnlyList<double> BackPropagate(double deltaForThisNeuron, double learningRate)
    {
        // 伝播用に更新前の重みを退避（δ_prev 計算で使用）
        var previousWeightVector = new double[_weightVector.Count];
        for (int i = 0; i < _weightVector.Count; i++) previousWeightVector[i] = _weightVector[i];

        // 勾配適用: wᵢ ← wᵢ − η · (δ · xᵢ)
        for (int i = 0; i < _weightVector.Count; i++)
            _weightVector[i] -= learningRate * deltaForThisNeuron * LastInputVector[i];

        // バイアス更新: b ← b − η · δ
        _bias -= learningRate * deltaForThisNeuron;

        // 前段へ返す δ_prev: δ_prev_i = δ · wᵢ (update前の wᵢ)
        var propagatedDeltaVector = new double[previousWeightVector.Length];
        for (int i = 0; i < previousWeightVector.Length; i++)
            propagatedDeltaVector[i] = deltaForThisNeuron * previousWeightVector[i];

        return propagatedDeltaVector;
    }

    /// <summary>
    /// 活性化の導関数 σ′(z) を返す（δ 計算に使用）。
    /// </summary>
    public double ActivationDerivative() => _activationFunction.DerivativeAt(LastPreActivationZ);
}