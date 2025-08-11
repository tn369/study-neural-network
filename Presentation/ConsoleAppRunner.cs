using MultiLayerNet.Application;

namespace MultiLayerNet.Presentation;

/// <summary>
/// コンソール入出力と学習ループのみを担当（ユースケース実装）。
/// 数式の流れ：
///  反復 e = 1..EPOCHS:
///    y = f(x)
///    L = 1/2 Σ_k (y_k − t_k)²
///    勾配降下で w, b を更新
/// </summary>
public sealed class ConsoleAppRunner : IAppRunner
{
    private readonly INeuralNetFactory _netFactory;
    private readonly IDemoDataFactory _dataFactory;

    public ConsoleAppRunner(INeuralNetFactory netFactory, IDemoDataFactory dataFactory)
    { _netFactory = netFactory; _dataFactory = dataFactory; }


    public void Run(ModelKind modelKind)
    {
        var neuralNetService = _netFactory.Create(modelKind);
        var data = _dataFactory.Resolve(modelKind);

        Console.WriteLine($"=== {neuralNetService.Name} : 学習デモ開始 ===");

        // デモ用の固定データ（単一サンプル）

        double learningRate = modelKind == ModelKind.Cnn ? 0.1 : 0.5;
        int epochs = 20;

        for (int e = 1; e <= epochs; e++)
        {
            var (input, target) = data.GetSample();
            var y = neuralNetService.Predict(input);
            var loss = neuralNetService.CalcLoss(y, target);
            Console.WriteLine($"[Epoch {e}] 出力:{y[0]:F4}  目標:{target[0]}  Loss:{loss:F4}");
            neuralNetService.Train(input, target, learningRate);
        }
        Console.WriteLine($"=== {neuralNetService.Name} : 学習デモ終了 ===");
    }
}