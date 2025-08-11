namespace MultiLayerNet.Application;
public interface IDemoDataFactory
{
    IDemoDataProvider Resolve(ModelKind kind);
}
