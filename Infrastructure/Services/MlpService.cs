using MultiLayerNet.Application;
using MultiLayerNet.Domain.Activations;
using MultiLayerNet.Domain.Mlp;
using NeuralNetwork.Infrastructure.Common;

namespace MultiLayerNet.Infrastructure.Services;

/// <summary>
/// MLP（全結合ネットワーク）のサービス実装。
/// 方式比較のため、INeuralNetService を通じて Predict/Train/Loss を提供。
/// </summary>
public sealed class MlpService : INeuralNetService
{
    public string Name => "MLP (Fully-Connected FeedForward)";

    private readonly FeedForwardNetwork _feedForwardNetwork;
    private readonly ILossFunction _lossFunction;

    public MlpService(IActivationFunction activationFunction, ILossFunction lossFunction, IRandomInitializer randomInitializer)
    {
        _lossFunction = lossFunction;

        // ネットワーク構成例: 入力3 → 隠れ3 → 隠れ2 → 出力1
        var layer1 = new Layer(new[]
        {
            new Neuron(new [] { randomInitializer.NextWeight(), randomInitializer.NextWeight(), randomInitializer.NextWeight() }, randomInitializer.NextBias(), activationFunction),
            new Neuron(new [] { randomInitializer.NextWeight(), randomInitializer.NextWeight(), randomInitializer.NextWeight() }, randomInitializer.NextBias(), activationFunction),
            new Neuron(new [] { randomInitializer.NextWeight(), randomInitializer.NextWeight(), randomInitializer.NextWeight() }, randomInitializer.NextBias(), activationFunction),
        });

        var layer2 = new Layer(new[]
        {
            new Neuron(new [] { randomInitializer.NextWeight(), randomInitializer.NextWeight(), randomInitializer.NextWeight() }, randomInitializer.NextBias(), activationFunction),
            new Neuron(new [] { randomInitializer.NextWeight(), randomInitializer.NextWeight(), randomInitializer.NextWeight() }, randomInitializer.NextBias(), activationFunction),
        });

        var layer3 = new Layer(new[]
        {
            new Neuron(new [] { randomInitializer.NextWeight(), randomInitializer.NextWeight() }, randomInitializer.NextBias(), activationFunction),
        });

        _feedForwardNetwork = new FeedForwardNetwork(new[] { layer1, layer2, layer3 });
    }

    public IReadOnlyList<double> Predict(IReadOnlyList<double> inputVector)
        => _feedForwardNetwork.Predict(inputVector);

    public void Train(IReadOnlyList<double> inputVector, IReadOnlyList<double> targetVector, double learningRate)
        => _feedForwardNetwork.TrainOne(inputVector, targetVector, learningRate);

    public double CalcLoss(IReadOnlyList<double> outputVector, IReadOnlyList<double> targetVector)
        => _lossFunction.Loss(outputVector, targetVector);
}