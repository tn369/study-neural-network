using MultiLayerNet.Application;
using Microsoft.Extensions.DependencyInjection;

namespace MultiLayerNet.Infrastructure.Datasets;

public sealed class DemoDataFactory : IDemoDataFactory
{
    private readonly IServiceProvider _sp;
    public DemoDataFactory(IServiceProvider sp) => _sp = sp;

    public IDemoDataProvider Resolve(ModelKind kind) => kind switch
    {
        ModelKind.Mlp => _sp.GetRequiredService<MlpDemoDataProvider>(),
        ModelKind.Cnn => _sp.GetRequiredService<CnnDemoDataProvider>(),
        //ModelKind.Rnn => _sp.GetRequiredService<RnnDemoDataProvider>(),         // 将来
        //odelKind.Transformer => _sp.GetRequiredService<TransformerDemoDataProvider>(), // 将来
        _ => throw new NotSupportedException(kind.ToString())
    };
}
