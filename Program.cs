using Microsoft.Extensions.DependencyInjection;
using MultiLayerNet.Application;
using MultiLayerNet.Domain.Activations;
using MultiLayerNet.Infrastructure.Datasets;
using MultiLayerNet.Infrastructure.Factories;
using MultiLayerNet.Infrastructure.Services;
using MultiLayerNet.Presentation;
using NeuralNetwork.Domain.Losses;
using NeuralNetwork.Infrastructure.Common;

namespace MultiLayerNet;

public class Program
{
    public static void Main(string[] args)
    {
        // Program.cs は「構成と起動」だけに責務を限定（DDDの分離）
        var serviceCollection = new ServiceCollection();

        // --- Domain（純粋関数オブジェクトなど） ---
        serviceCollection.AddSingleton<IActivationFunction, Sigmoid>();
        serviceCollection.AddSingleton<ILossFunction, MeanSquaredError>();
        serviceCollection.AddSingleton<IRandomInitializer, DefaultRandomInitializer>();
        //TODO: 整理
        serviceCollection.AddSingleton<ReLU, ReLU>();
        serviceCollection.AddSingleton<Sigmoid, Sigmoid>();

        // --- Services（将来 RNN/Transformer もここに登録） ---
        serviceCollection.AddSingleton<MlpService>();
        serviceCollection.AddSingleton<CnnService>();
        serviceCollection.AddSingleton<RnnServiceStub>();
        serviceCollection.AddSingleton<TransformerServiceStub>();

        // --- Factory ---
        serviceCollection.AddSingleton<INeuralNetFactory, NeuralNetFactory>();

        // --- UseCase / Presentation ---
        serviceCollection.AddSingleton<IAppRunner, ConsoleAppRunner>();

        // --- Datasets ---
        serviceCollection.AddSingleton<IDemoDataFactory, DemoDataFactory>();
        serviceCollection.AddSingleton<MlpDemoDataProvider>();
        serviceCollection.AddSingleton<CnnDemoDataProvider>();

        var serviceProvider = serviceCollection.BuildServiceProvider();

        // 方式切替：ここでは MLP を実行（引数や設定で切替可能）
        var appRunner = serviceProvider.GetRequiredService<IAppRunner>();

        appRunner.Run(ModelKind.Mlp);
        appRunner.Run(ModelKind.Cnn);
    }
}