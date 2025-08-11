namespace MultiLayerNet.Domain.Mlp;

/// <summary>
/// レイヤ（同次元のニューロン集合）。
/// 順伝播: 各ニューロンで y = σ(Σ w·x + b)
/// 逆伝播: incomingDeltaVector を受け、各ニューロンの δ を計算して更新し、
///         前段へ合算 δ を返却。
/// </summary>
public sealed class Layer
{
    private readonly List<Neuron> _neurons;

    public Layer(IReadOnlyList<Neuron> neurons)
    {
        if (neurons == null || neurons.Count == 0) throw new ArgumentException("neurons is empty");
        _neurons = new List<Neuron>(neurons);
    }

    /// <summary>
    /// 順伝播。
    /// 出力ベクトル y を返す。各成分は y_j = σ(z_j), z_j = Σᵢ (w_{j,i} · xᵢ) + b_j
    /// </summary>
    public IReadOnlyList<double> FeedForward(IReadOnlyList<double> inputVector)
    {
        var outputVector = new double[_neurons.Count];
        for (int j = 0; j < _neurons.Count; j++)
            outputVector[j] = _neurons[j].FeedForward(inputVector);
        return outputVector;
    }

    /// <summary>
    /// 逆伝播。
    /// incomingDeltaVector の意味:
    ///  - 出力層の場合: (y − t) を要素にもつベクトル（後で σ′(z) を掛けて δ を作る）
    ///  - 隠れ層の場合: Σ_j w_next_{i,j} · δ_next_j（すでに合算された前段向けの値）
    /// ここで最終的な δ は δ = incoming * σ′(z)。
    /// 返り値は前段へ流す δ ベクトル（各入力次元 i に対し δ_prev_i を合算）。
    /// </summary>
    public IReadOnlyList<double> BackPropagate(IReadOnlyList<double> incomingDeltaVector, double learningRate)
    {
        int inputDimension = _neurons[0].LastInputVector.Count; // x の次元
        var outgoingDeltaVector = new double[inputDimension];   // 前段へ返す δ_prev を各 i で合算

        for (int j = 0; j < _neurons.Count; j++)
        {
            // このニューロンの δ = incoming * σ′(z)
            double deltaForNeuron = incomingDeltaVector[j] * _neurons[j].ActivationDerivative();

            var propagatedDeltaVector = _neurons[j].BackPropagate(deltaForNeuron, learningRate);

            // 各入力次元 i へ δ_prev_i を加算（複数出力先から集約）
            for (int i = 0; i < inputDimension; i++)
                outgoingDeltaVector[i] += propagatedDeltaVector[i];
        }
        return outgoingDeltaVector;
    }
}