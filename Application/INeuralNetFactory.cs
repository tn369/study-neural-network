namespace MultiLayerNet.Application;

public interface INeuralNetFactory
{
    INeuralNetService Create(ModelKind kind);
}
