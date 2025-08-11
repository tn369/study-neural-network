namespace NeuralNetwork.Infrastructure.Common;

public sealed class DefaultRandomInitializer : IRandomInitializer
{
    private readonly Random _random = new(0); // 再現性のため固定シード

    public double NextWeight() => _random.NextDouble() - 0.5;
    public double NextBias() => 0.0;
}