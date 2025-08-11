using Microsoft.Extensions.DependencyInjection;
using MultiLayerNet.Application;
using MultiLayerNet.Infrastructure.Services;

namespace MultiLayerNet.Infrastructure.Factories;

public sealed class NeuralNetFactory : INeuralNetFactory
{
    private readonly IServiceProvider _sp;
    public NeuralNetFactory(IServiceProvider sp) => _sp = sp;

    public INeuralNetService Create(ModelKind kind) => kind switch
    {
        ModelKind.Mlp => _sp.GetRequiredService<MlpService>(),
        ModelKind.Cnn => _sp.GetRequiredService<CnnService>(),
        ModelKind.Rnn => _sp.GetRequiredService<RnnServiceStub>(),
        ModelKind.Transformer => _sp.GetRequiredService<TransformerServiceStub>(),
        _ => throw new NotSupportedException(kind.ToString())
    };
}
