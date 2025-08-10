using System;
using System.Collections.Generic;

namespace MultiLayerNet
{
    /// <summary>
    /// 単一ニューロン：複数入力＋学習用情報保持
    /// </summary>
    class Neuron
    {
        private readonly List<double> _weights;           // 重み w1…wn
        private double _bias;                             // バイアス b

        // --- 直近の順伝播時に保持しておく値（逆伝播に使う） ---
        public List<double> LastInputs { get; private set; }
        public double LastZ { get; private set; }  // Σ(w*x)+b
        public double LastOutput { get; private set; }  // σ(z)
        // ------------------------------------------------------

        public IReadOnlyList<double> Weights => _weights; // 外部から読み取りだけ可

        public Neuron(List<double> weights, double bias)
        {
            if (weights == null || weights.Count == 0)
                throw new ArgumentException("weights は空にできません");

            _weights = new List<double>(weights);
            _bias = bias;
        }

        // 活性化関数 σ とその導関数
        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private static double SigmoidDerivative(double x)
        {
            double s = Sigmoid(x);
            return s * (1 - s);
        }

        /// <summary>順方向計算（出力を返し、内部状態も保存）</summary>
        public double FeedForward(List<double> inputs)
        {
            if (inputs.Count != _weights.Count)
                throw new ArgumentException("入力数と重み数が一致しません");

            LastInputs = inputs;           // そのまま参照保持で OK（外で再利用しない想定）
            double z = 0.0;
            for (int i = 0; i < inputs.Count; i++)
                z += inputs[i] * _weights[i];
            z += _bias;

            LastZ = z;
            LastOutput = Sigmoid(z);
            return LastOutput;
        }

        /// <summary>
        /// 逆伝播：このニューロンの δ を受け取り重み・バイアスを更新。  
        /// 返り値は “入力側へ流す δ (δ * w)” で、前層計算用に使う。
        /// </summary>
        public List<double> BackPropagate(double delta, double learningRate)
        {
            // 伝播用に更新"前"の重みを退避
            var prevW = new double[_weights.Count];
            for (int i = 0; i < _weights.Count; i++) prevW[i] = _weights[i];

            // 勾配で更新
            for (int i = 0; i < _weights.Count; i++)
                _weights[i] -= learningRate * delta * LastInputs[i];
            _bias -= learningRate * delta;

            // 前段へ返す δ = delta * w (更新前の w を使用)
            List<double> propagated = new();
            for (int i = 0; i < prevW.Length; i++)
                propagated.Add(delta * prevW[i]);

            return propagated;
        }


        /// <summary>σ'(z) を返す（δ 計算に使う）</summary>
        public double ActivationDerivative() => SigmoidDerivative(LastZ);
    }

    /// <summary>複数ニューロンを束ねた Layer</summary>
    class Layer
    {
        private readonly List<Neuron> _neurons;
        public int NeuronCount => _neurons.Count;
        public Neuron this[int i] => _neurons[i]; // インデクサ

        public Layer(List<Neuron> neurons)
        {
            if (neurons == null || neurons.Count == 0)
                throw new ArgumentException("neurons は空にできません");
            _neurons = neurons;
        }

        /// <summary>順伝播：層の出力ベクトルを返す</summary>
        public List<double> FeedForward(List<double> inputs)
        {
            List<double> outputs = new();
            foreach (var n in _neurons)
                outputs.Add(n.FeedForward(inputs));
            return outputs;
        }

        /// <summary>
        /// 逆伝播（出力層なら targets、隠れ層なら nextDeltas を渡す）  
        /// 戻り値は “この層より前段へ流す δ ベクトル”
        /// </summary>
        public List<double> BackPropagate(
            List<double> incoming,   // 出力層: (output-target)、隠れ層: Σ w*δ
            double learningRate)
        {
            int inputCount = _neurons[0].LastInputs.Count;
            List<double> outgoing = new List<double>(inputCount);
            for (int i = 0; i < inputCount; i++) outgoing.Add(0.0);

            for (int i = 0; i < _neurons.Count; i++)
            {
                double delta = incoming[i] *
                    _neurons[i].ActivationDerivative();

                List<double> propagated = _neurons[i].BackPropagate(delta, learningRate);

                // 各入力次元分 δ を加算（前層ニューロンは複数出力先から δ が流れてくる）
                for (int j = 0; j < outgoing.Count; j++)
                    outgoing[j] += propagated[j];
            }
            return outgoing; // さらに前段へ
        }
    }

    /// <summary>Layer を複数持つ全体ネットワーク</summary>
    class Network
    {
        private readonly List<Layer> _layers;
        public Network(List<Layer> layers)
        {
            if (layers == null || layers.Count == 0)
                throw new ArgumentException("layers は空にできません");
            _layers = layers;
        }

        /// <summary>順伝播（最終出力を返す）</summary>
        public List<double> FeedForward(List<double> inputs)
        {
            List<double> current = inputs;
            foreach (var layer in _layers)
                current = layer.FeedForward(current);
            return current;
        }

        /// <summary>1 サンプルで学習（誤差逆伝播）</summary>
        public void Train(List<double> inputs, List<double> targets, double lr)
        {
            // --- 1) 順伝播で出力取得 ---
            List<double> outputs = FeedForward(inputs);

            // --- 2) 出力層 δ = (output - target) ---
            List<double> deltas = new();
            for (int i = 0; i < outputs.Count; i++)
                deltas.Add(outputs[i] - targets[i]);

            // --- 3) 層を逆順にたどって BackPropagate ---
            for (int layerIdx = _layers.Count - 1; layerIdx >= 0; layerIdx--)
            {
                deltas = _layers[layerIdx].BackPropagate(deltas, lr);
            }
        }

        /// <summary>平均二乗誤差 (MSE/2) を計算</summary>
        public double CalcLoss(List<double> outputs, List<double> targets)
        {
            double sum = 0.0;
            for (int i = 0; i < outputs.Count; i++)
                sum += Math.Pow(outputs[i] - targets[i], 2);
            return 0.5 * sum;
        }
    }

    class Program
    {
        static void Main()
        {
            // ---------- ネットワーク構造 ----------
            // 入力3 → 隠れ層3 → 隠れ層2 → 出力層1
            Random rnd = new(0); // 再現性のためシード固定

            Layer input = new(new List<Neuron>
            {
                new Neuron(new List<double>{ rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5 }, bias: 0),
                new Neuron(new List<double>{ rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5 }, bias: 0),
                new Neuron(new List<double>{ rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5 }, bias: 0)
            });

            // 隠れ層（2ニューロン, 3入力）
            Layer hidden = new(new List<Neuron>
            {
                new Neuron(new List<double>{ rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5 }, bias: 0),
                new Neuron(new List<double>{ rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5 }, bias: 0)
            });

            // 出力層（1ニューロン, 隠れ層2出力を入力）
            Layer output = new(new List<Neuron>
            {
                new Neuron(new List<double>{ rnd.NextDouble()-0.5,
                                             rnd.NextDouble()-0.5 }, bias: 0)
            });

            Network net = new(new List<Layer> { input, hidden, output });

            // ---------- 学習用データ ----------
            List<double> inputs = new() { 1.0, 0.5, -1.2 };
            List<double> targets = new() { 0.8 };          // 1 出力
            double lr = 0.5;   // 学習率
            int epochs = 20;    // 学習回数
            // ----------------------------------

            Console.WriteLine("=== 学習開始 ===");
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                // 学習前の出力・損失
                List<double> outputs = net.FeedForward(inputs);
                double loss = net.CalcLoss(outputs, targets);

                Console.WriteLine($"\n-- Epoch {epoch} --");
                Console.WriteLine($"出力   : {outputs[0]:F4}  目標: {targets[0]}  損失: {loss:F4}");

                // 学習（誤差逆伝播で重み更新）
                net.Train(inputs, targets, lr);
            }

            Console.WriteLine("\n=== 学習終了 ===");
        }
    }
}
