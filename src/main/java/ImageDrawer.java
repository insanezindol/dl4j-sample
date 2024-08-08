import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.collect.Streams;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.Random;

public class ImageDrawer {

    private JFrame mainFrame;
    private MultiLayerNetwork nn;

    private BufferedImage originalImage;
    private JLabel generatedLabel;

    private INDArray xyOut;

    private Java2DNativeImageLoader j2dNil;
    private FastRGB rgb;
    private Random random;

    private void init() throws Exception {
        // JFrame 인스턴스 만들기
        mainFrame = new JFrame("Image drawer example");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        String localDataPath = DownloaderUtility.DATAEXAMPLES.Download();
        originalImage = ImageIO.read(new File(localDataPath, "Mona_Lisa.png"));

        // 원본과 동일한 크기의 빈 이미지로 시작합니다.
        BufferedImage generatedImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        final JLabel originalLabel = new JLabel(new ImageIcon(originalImage));
        generatedLabel = new JLabel(new ImageIcon(generatedImage));

        originalLabel.setBounds(0, 0, width, height);
        //x축, y축, 폭, 높이
        generatedLabel.setBounds(width, 0, width, height);

        mainFrame.add(originalLabel);
        mainFrame.add(generatedLabel);

        mainFrame.setSize(2 * width, height + 25);
        mainFrame.setLayout(null);
        // UI 표시
        mainFrame.setVisible(true);

        //이미지 작성에 사용되는 Datavec 클래스입니다.
        j2dNil = new Java2DNativeImageLoader();
        random = new Random();
        // 신경망을 만듭니다.
        nn = createNN();
        // 이미지를 생성하는 데 사용되는 메쉬를 만듭니다.
        xyOut = calcGrid();

        // 원본 이미지에서 컬러 채널을 읽습니다.
        rgb = new FastRGB(originalImage);

        SwingUtilities.invokeLater(this::onCalc);
    }

    public static void main(String[] args) throws Exception {
        ImageDrawer imageDrawer = new ImageDrawer();
        imageDrawer.init();
    }

    /**
     * 신경망을 구축합니다.
     */
    private static MultiLayerNetwork createNN() {
        int seed = 2345;
        double learningRate = 0.001;
        // x와 y.
        int numInputs = 2;
        int numHiddenNodes = 1000;
        // R, G, B 값.
        int numOutputs = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    /**
     * NN을 교육하고 현재 그래픽 출력을 업데이트합니다.
     */
    private void onCalc() {
        // 생성된 재도면당 배치 크기와 배치 수 사이의 합리적인 균형을 찾습니다.
        // 배치 크기가 클수록 계산이 느려지지만 배치당 학습 속도가 빨라집니다
        int batchSize = 1000;
        // 생성된 이미지를 그리는 속도가 느립니다. 다시 그리기 전에 여러 번 배치하면 속도가 빨라집니다.
        int numBatches = 10;
        for (int i = 0; i < numBatches; i++) {
            DataSet ds = generateDataSet(batchSize);
            nn.fit(ds);
        }
        drawImage();
        mainFrame.invalidate();
        mainFrame.repaint();

        SwingUtilities.invokeLater(this::onCalc); //TODO: move training to a worker thread,
    }

    /**
     * 원본 이미지에서 임의 샘플의 배치 크기를 가져옵니다.
     * 이것은 사용자 정의 데이터 세트를 생성하는 방법을 보여줍니다. 일반적인 방법은 전체 소스 이미지의 데이터 세트를 생성하는 것입니다. 여기서 혼합 배치를 훈련하는 것입니다.
     *
     * @param 영상에서 꺼낼 샘플 점의 배치 크기입니다.
     * @return DeepLearning4J DataSet.
     */
    private DataSet generateDataSet(int batchSize) {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        float[][] in = new float[batchSize][2];
        float[][] out = new float[batchSize][3];
        final int[] i = {0};
        Streams.forEachPair(
                random.ints(batchSize, 0, w).boxed(),
                random.ints(batchSize, 0, h).boxed(),
                (a, b) -> {
                    final short[] parts = rgb.getRGB(a, b);
                    in[i[0]] = new float[]{((a / (float) w) - 0.5f) * 2f, ((b / (float) h) - 0.5f) * 2f};
                    out[i[0]] = new float[]{parts[0], parts[1], parts[2]};
                    i[0]++;
                }
        );
        final INDArray input = Nd4j.create(in);
        final INDArray labels = Nd4j.create(out).divi(255);
        return new DataSet(input, labels);
    }

    /**
     * Neural Network(신경망)가 이미지를 그리도록 합니다.
     */
    private void drawImage() {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        // 원시 NN 출력입니다.
        INDArray out = nn.output(xyOut);
        // 0에서 1 사이를 잘라냅니다.
        BooleanIndexing.replaceWhere(out, 0.0, Conditions.lessThan(0.0));
        BooleanIndexing.replaceWhere(out, 1.0, Conditions.greaterThan(1.0));
        // 바이트로 convert합니다.
        out = out.mul(255).castTo(DataType.INT8);

        // 개별 컬러 레이어를 추출합니다.
        INDArray r = out.getColumn(0);
        INDArray g = out.getColumn(1);
        INDArray b = out.getColumn(2);

        // 색상을 재조합하고 이미지 크기로 모양을 바꿉니다.
        INDArray imgArr = Nd4j.vstack(b, g, r).reshape(3, h, w);

        // UI를 업데이트합니다.
        BufferedImage img = j2dNil.asBufferedImage(imgArr);
        generatedLabel.setIcon(new ImageIcon(img));
    }

    /**
     * NN 출력을 계산하기 위한 x,y 그리드. 한 번만 계산하면 됩니다.
     */
    private INDArray calcGrid() {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();
        INDArray xPixels = Nd4j.linspace(-1.0, 1.0, w, DataType.DOUBLE);
        INDArray yPixels = Nd4j.linspace(-1.0, 1.0, h, DataType.DOUBLE);
        INDArray[] mesh = Nd4j.meshgrid(xPixels, yPixels);

        return Nd4j.vstack(mesh[0].ravel(), mesh[1].ravel()).transpose();
    }


    public class FastRGB {
        int width;
        int height;
        private boolean hasAlphaChannel;
        private int pixelLength;
        private byte[] pixels;

        FastRGB(BufferedImage image) {
            pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            width = image.getWidth();
            height = image.getHeight();
            hasAlphaChannel = image.getAlphaRaster() != null;
            pixelLength = 3;
            if (hasAlphaChannel)
                pixelLength = 4;
        }

        short[] getRGB(int x, int y) {
            int pos = (y * pixelLength * width) + (x * pixelLength);
            short rgb[] = new short[4];
            if (hasAlphaChannel)
                rgb[3] = (short) (pixels[pos++] & 0xFF); // Alpha
            rgb[2] = (short) (pixels[pos++] & 0xFF); // Blue
            rgb[1] = (short) (pixels[pos++] & 0xFF); // Green
            rgb[0] = (short) (pixels[pos] & 0xFF); // Red
            return rgb;
        }
    }
}
