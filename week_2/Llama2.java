import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class Llama2 {

    // container for the model's configuration parameters.
    public static class Config {
        int dim;
        int hiddenDim;
        int nLayers;
        int nHeads;
        int nKvHeads;
        int vocabSize;
        int seqLen;

        @Override
        public String toString() {
            return "Config{" +
                   "dim=" + dim +
                   ", hiddenDim=" + hiddenDim +
                   ", nLayers=" + nLayers +
                   ", nHeads=" + nHeads +
                   ", nKvHeads=" + nKvHeads +
                   ", vocabSize=" + vocabSize +
                   ", seqLen=" + seqLen +
                   '}';
        }
    }

    // container for the model's weights, using multidimensional arrays for clarity.
    public static class TransformerWeights {
        float[][] tokenEmbeddingTable;
        float[][] rmsAttWeight;
        float[][] rmsFfnWeight;
        float[][][] wq;
        float[][][] wk;
        float[][][] wv;
        float[][][] wo;
        float[][][] w1;
        float[][][] w2;
        float[][][] w3;
        float[] rmsFinalWeight;
        float[][] wcls;
    }
    
    public static class Transformer {
        Config config;
        TransformerWeights weights;
    }
    
    private static float[][] readMatrix(ByteBuffer buffer, int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = buffer.getFloat();
            }
        }
        return matrix;
    }

    private static float[][][] read3DMatrix(ByteBuffer buffer, int dim1, int dim2, int dim3) {
        float[][][] matrix = new float[dim1][dim2][dim3];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    matrix[i][j][k] = buffer.getFloat();
                }
            }
        }
        return matrix;
    }

    public static Transformer readCheckpoint(String checkpointPath) throws IOException {
        Transformer transformer = new Transformer();
        transformer.config = new Config();
        transformer.weights = new TransformerWeights();
        
        try (FileInputStream fis = new FileInputStream(checkpointPath);
             FileChannel channel = fis.getChannel()) {
            
            // Use a standard heap buffer.
            ByteBuffer buffer = ByteBuffer.allocate((int) channel.size());
            
            // Loop the read call to ensure the entire file is read.
            while(buffer.hasRemaining()) {
                channel.read(buffer);
            }
            
            // Prepare buffer for reading.
            buffer.flip();

            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // Read Config struct.
            Config config = transformer.config;
            config.dim = buffer.getInt();
            config.hiddenDim = buffer.getInt();
            config.nLayers = buffer.getInt();
            config.nHeads = buffer.getInt();
            config.nKvHeads = buffer.getInt();
            int vocabSizeRaw = buffer.getInt();
            config.seqLen = buffer.getInt();

            boolean sharedWeights = vocabSizeRaw > 0;
            config.vocabSize = Math.abs(vocabSizeRaw);

            // Read Weights. The buffer's position advances automatically.
            TransformerWeights weights = transformer.weights;
            int headSize = config.dim / config.nHeads;
            
            weights.tokenEmbeddingTable = readMatrix(buffer, config.vocabSize, config.dim);
            weights.rmsAttWeight = readMatrix(buffer, config.nLayers, config.dim);
            weights.wq = read3DMatrix(buffer, config.nLayers, config.dim, config.dim);
            weights.wk = read3DMatrix(buffer, config.nLayers, config.dim, config.dim);
            weights.wv = read3DMatrix(buffer, config.nLayers, config.dim, config.dim);
            weights.wo = read3DMatrix(buffer, config.nLayers, config.dim, config.dim);
            weights.rmsFfnWeight = readMatrix(buffer, config.nLayers, config.dim);
            weights.w1 = read3DMatrix(buffer, config.nLayers, config.hiddenDim, config.dim);
            weights.w2 = read3DMatrix(buffer, config.nLayers, config.dim, config.hiddenDim);
            weights.w3 = read3DMatrix(buffer, config.nLayers, config.hiddenDim, config.dim);
            
            weights.rmsFinalWeight = new float[config.dim];
            for (int i = 0; i < config.dim; i++) {
                weights.rmsFinalWeight[i] = buffer.getFloat();
            }

            // Skip RoPE weights by advancing buffer position.
            int ropeSize = config.seqLen * headSize / 2;
            buffer.position(buffer.position() + ropeSize * Float.BYTES);
            buffer.position(buffer.position() + ropeSize * Float.BYTES);

            if (sharedWeights) {
                weights.wcls = weights.tokenEmbeddingTable;
            } else {
                weights.wcls = readMatrix(buffer, config.vocabSize, config.dim);
            }
        }
        return transformer;
    }

    public static void main(String[] args) {
        String checkpointPath = "stories15M.bin";
        System.out.println("load checkpoint: " + checkpointPath + "\n");
        
        try {
            Transformer transformer = readCheckpoint(checkpointPath);
            
            System.out.println("Test 1: Verifying Configuration");
            Config config = transformer.config;
            System.out.println("Loaded Config: " + config);
            boolean configOk = config.dim == 288 && config.hiddenDim == 768 && config.nLayers == 6 &&
                               config.nHeads == 6 && config.nKvHeads == 6 && config.vocabSize == 32000 &&
                               config.seqLen == 256;
            System.out.println("Config verification: " + (configOk ? "SUCCESS" : "FAILURE"));

            System.out.println("Test 2: Verifying Weight Dimensions");
            TransformerWeights weights = transformer.weights;
            System.out.printf("tokenEmbeddingTable: [%d][%d]\n", weights.tokenEmbeddingTable.length, weights.tokenEmbeddingTable[0].length);
            System.out.printf("wq (layer 0):        [%d][%d]\n", weights.wq[0].length, weights.wq[0][0].length);
            System.out.printf("w1 (layer 0):        [%d][%d]\n", weights.w1[0].length, weights.w1[0][0].length);
            System.out.printf("rmsFinalWeight:      [%d]\n", weights.rmsFinalWeight.length);
            boolean dimsOk = weights.tokenEmbeddingTable.length == config.vocabSize &&
                             weights.wq[0].length == config.dim &&
                             weights.w1[0].length == config.hiddenDim &&
                             weights.rmsFinalWeight.length == config.dim;
            System.out.println("Dimensions verification: " + (dimsOk ? "SUCCESS" : "FAILURE"));

        } catch (IOException e) {
            System.err.println("Failed to read checkpoint file: " + e.getMessage());
        }
    }
}