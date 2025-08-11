using System;
using MultiLayerNet.Domain.Activations;

namespace MultiLayerNet.Domain.Cnn
{
    /// <summary>
    /// 2D 畳み込み層（valid, stride=1, 単一チャネル・単一フィルタの最小実装）
    /// - 入力画像: X ∈ R^{H×W}
    /// - カーネル(フィルタ): K ∈ R^{kH×kW}
    /// - バイアス: b ∈ R
    /// - 活性化: φ（例: ReLU）
    ///
    /// 【順伝播】
    ///   前活性 Z[oR, oC] = Σ_{kr=0..kH-1} Σ_{kc=0..kW-1} X[oR+kr, oC+kc] * K[kr, kc] + b
    ///   出力   Y = φ(Z)
    ///
    /// 【逆伝播】（loss L に対して）
    ///   与えられるもの: dL/dY（= upstreamGradY）
    ///   1) dL/dZ = dL/dY ⊙ φ'(Z)
    ///   2) dL/dK[kr,kc] = Σ_{oR,oC} X[oR+kr, oC+kc] * dL/dZ[oR,oC]
    ///   3) dL/db = Σ_{oR,oC} dL/dZ[oR,oC]
    ///   4) dL/dX[i,j] = Σ_{kr,kc} dL/dZ[i-kr, j-kc] * K[kr,kc]   （インデックスが出力範囲内のときのみ加算）
    ///
    /// 学習時には 2) と 3) で得た勾配を使い K と b を更新します。
    /// </summary>
    public sealed class Conv2D
    {
        // --- ハイパーパラメータ（この最小実装では固定） ---
        private readonly int _kernelHeight; // kH
        private readonly int _kernelWidth;  // kW
        private readonly IActivationFunction _activation; // φ と φ'

        // --- 学習対象パラメータ ---
        private readonly double[,] _kernelWeights; // K[kr,kc]
        private double _bias;                      // b

        // --- 逆伝播のために保持するキャッシュ（直近の順伝播の値） ---
        private double[,] _cacheInputX = default!;        // X
        private double[,] _cachePreActivationZ = default!; // Z
        private double[,] _cacheOutputY = default!;        // Y

        /// <summary>
        /// コンストラクタ
        /// </summary>
        /// <param name="kernelHeight">カーネルの高さ kH</param>
        /// <param name="kernelWidth">カーネルの幅 kW</param>
        /// <param name="activation">活性化関数 φ（例: ReLU）</param>
        /// <param name="weightInit">重み初期化関数（呼ぶたび新しい乱数等を返す）</param>
        /// <param name="biasInit">バイアス初期化関数</param>
        public Conv2D(
            int kernelHeight,
            int kernelWidth,
            IActivationFunction activation,
            Func<double> weightInit,
            Func<double> biasInit)
        {
            if (kernelHeight <= 0) throw new ArgumentOutOfRangeException(nameof(kernelHeight));
            if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth));
            _kernelHeight = kernelHeight;
            _kernelWidth = kernelWidth;
            _activation = activation ?? throw new ArgumentNullException(nameof(activation));

            _kernelWeights = new double[_kernelHeight, _kernelWidth];
            for (int kr = 0; kr < _kernelHeight; kr++)
            {
                for (int kc = 0; kc < _kernelWidth; kc++)
                {
                    _kernelWeights[kr, kc] = weightInit();
                }
            }

            _bias = biasInit();
        }

        /// <summary>
        /// 順伝播: 入力 X（H×W）→ 前活性 Z → 活性化 Y（(H-kH+1)×(W-kW+1)）
        /// </summary>
        public double[,] Forward(double[,] inputImageX)
        {
            if (inputImageX is null) throw new ArgumentNullException(nameof(inputImageX));

            // 入力サイズ
            int inH = inputImageX.GetLength(0);
            int inW = inputImageX.GetLength(1);

            // 出力サイズ（valid, stride=1 のためこの式）
            int outH = inH - _kernelHeight + 1;
            int outW = inW - _kernelWidth + 1;
            if (outH <= 0 || outW <= 0)
                throw new ArgumentException("入力がカーネルより小さいため、valid 畳み込みの出力サイズが負または 0 になります。");

            // 逆伝播用にキャッシュ（X は破壊しないよう clone）
            _cacheInputX = (double[,])inputImageX.Clone();

            // 1) 畳み込み + バイアス = 前活性 Z
            var preActivationZ = new double[outH, outW];
            for (int outRow = 0; outRow < outH; outRow++)
            {
                for (int outCol = 0; outCol < outW; outCol++)
                {
                    double sum = 0.0;

                    // カーネルの各位置（kr,kc）に対して、入力の重み付き和を取る
                    for (int kr = 0; kr < _kernelHeight; kr++)
                    {
                        for (int kc = 0; kc < _kernelWidth; kc++)
                        {
                            // 入力上の対応座標（スライド窓の位置）
                            int inRow = outRow + kr;
                            int inCol = outCol + kc;
                            sum += inputImageX[inRow, inCol] * _kernelWeights[kr, kc];
                        }
                    }

                    preActivationZ[outRow, outCol] = sum + _bias;
                }
            }
            _cachePreActivationZ = preActivationZ;

            // 2) 活性化 Y = φ(Z)
            var outputY = new double[outH, outW];
            for (int outRow = 0; outRow < outH; outRow++)
            {
                for (int outCol = 0; outCol < outW; outCol++)
                {
                    outputY[outRow, outCol] = _activation.Invoke(preActivationZ[outRow, outCol]);
                }
            }
            _cacheOutputY = outputY;

            return outputY;
        }

        /// <summary>
        /// 逆伝播:
        /// 上流からの勾配 dL/dY（= upstreamGradY）を受け取り、
        ///   - dL/dZ を計算（活性化の微分を掛ける）
        ///   - dL/dK, dL/db を求めてパラメータ更新
        ///   - dL/dX（= 戻り値）を計算して前段に渡す
        /// </summary>
        /// <param name="upstreamGradY">dL/dY（出力 Y と同じ形: outH×outW）</param>
        /// <param name="learningRate">学習率 η</param>
        /// <returns>dL/dX（入力と同じ形: inH×inW）</returns>
        public double[,] Backward(double[,] upstreamGradY, double learningRate)
        {
            if (upstreamGradY is null) throw new ArgumentNullException(nameof(upstreamGradY));
            if (_cacheOutputY is null || _cachePreActivationZ is null || _cacheInputX is null)
                throw new InvalidOperationException("Backward を呼ぶ前に Forward を実行してください。");

            int outH = _cacheOutputY.GetLength(0);
            int outW = _cacheOutputY.GetLength(1);

            // 形状チェック（学習時の取り違え防止）
            if (upstreamGradY.GetLength(0) != outH || upstreamGradY.GetLength(1) != outW)
                throw new ArgumentException("upstreamGradY の形状が Forward 出力と一致しません。");

            // --- 1) dL/dZ = dL/dY ⊙ φ'(Z) ---
            var gradZ = new double[outH, outW];
            for (int outRow = 0; outRow < outH; outRow++)
            {
                for (int outCol = 0; outCol < outW; outCol++)
                {
                    double dL_dY = upstreamGradY[outRow, outCol];
                    double dPhi_dZ = _activation.DerivativeAt(_cachePreActivationZ[outRow, outCol]);
                    gradZ[outRow, outCol] = dL_dY * dPhi_dZ;
                }
            }

            // --- 2) dL/dK を計算（カーネル毎の畳み込み勾配） ---
            var gradKernel = new double[_kernelHeight, _kernelWidth];
            for (int kr = 0; kr < _kernelHeight; kr++)
            {
                for (int kc = 0; kc < _kernelWidth; kc++)
                {
                    double sum = 0.0;
                    // 出力位置 (outRow, outCol) を総当たりして寄与を集計
                    for (int outRow = 0; outRow < outH; outRow++)
                    {
                        for (int outCol = 0; outCol < outW; outCol++)
                        {
                            int inRow = outRow + kr;
                            int inCol = outCol + kc;
                            sum += _cacheInputX[inRow, inCol] * gradZ[outRow, outCol];
                        }
                    }
                    gradKernel[kr, kc] = sum;
                }
            }

            // --- 3) dL/db を計算（前活性 Z への勾配の総和） ---
            double gradBias = 0.0;
            for (int outRow = 0; outRow < outH; outRow++)
            {
                for (int outCol = 0; outCol < outW; outCol++)
                {
                    gradBias += gradZ[outRow, outCol];
                }
            }

            // --- 4) dL/dX を計算（前段へ返す勾配） ---
            int inH = _cacheInputX.GetLength(0);
            int inW = _cacheInputX.GetLength(1);
            var gradX = new double[inH, inW];

            // 各入力ピクセル (inRow,inCol) に対する寄与を「カーネルで重み付けして」集める
            for (int inRow = 0; inRow < inH; inRow++)
            {
                for (int inCol = 0; inCol < inW; inCol++)
                {
                    double sum = 0.0;
                    for (int kr = 0; kr < _kernelHeight; kr++)
                    {
                        for (int kc = 0; kc < _kernelWidth; kc++)
                        {
                            // 逆畳み込みの対応： out 座標 = in 座標 - カーネルオフセット
                            int outRow = inRow - kr;
                            int outCol = inCol - kc;
                            bool inside = (0 <= outRow && outRow < outH) && (0 <= outCol && outCol < outW);
                            if (inside)
                            {
                                sum += gradZ[outRow, outCol] * _kernelWeights[kr, kc];
                            }
                        }
                    }
                    gradX[inRow, inCol] = sum;
                }
            }

            // --- 5) パラメータ更新（SGD） ---
            for (int kr = 0; kr < _kernelHeight; kr++)
            {
                for (int kc = 0; kc < _kernelWidth; kc++)
                {
                    _kernelWeights[kr, kc] -= learningRate * gradKernel[kr, kc];
                }
            }
            _bias -= learningRate * gradBias;

            // 前段に勾配を返す
            return gradX;
        }

        // --- 参考：デバッグや学習監視用の簡易アクセサ（必要なら） ---
        public double[,] GetKernelSnapshot() => (double[,])_kernelWeights.Clone();
        public double GetBias() => _bias;
    }
}
