package raf.nvelickovic10.masterProjekat.factory;

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

public abstract class LayerFactory {

//	private static final Logger LOG = new Logger(LayerFactory.class.getSimpleName());

//	Convolutional layers

	/**
	 * Convolutional layer <br />
	 * For initial convolutional layer the in parameter is the input depth,
	 * otherwise is the previous layer size.
	 * 
	 * @param name   Layer name
	 * @param in     Number of inputs - usually the previous layer size
	 * @param out    Number of outputs
	 * @param kernel Kernel dimensions new int[]{x, y}
	 * @param stride Stride dimensions new int[]{x, y}
	 * @param pad    Padding dimensions new int[]{x, y}
	 * @param bias   Bias value
	 */
	public static ConvolutionLayer convolutionalLayer(String name, int in, int out, int[] kernel, int[] stride,
			int[] pad, double bias) {
		Builder builder = new ConvolutionLayer.Builder(kernel, stride, pad).name(name);

		if (in > 0) {
			builder = builder.nIn(in);
		}

		return builder.nOut(out).biasInit(bias).build();
	}

	/**
	 * Convolutional layer with 3x3 kernel
	 * 
	 * @param name   Layer name
	 * @param in     Number of inputs - usually the previous layer size
	 * @param out    Number of outputs
	 * @param stride Stride dimensions new int[]{x, y}
	 * @param pad    Padding dimensions new int[]{x, y}
	 * @param bias   Bias value
	 */
	public static ConvolutionLayer convolutionalLayer3x3(String name, int in, int out, int[] stride, int[] pad,
			double bias) {
		return convolutionalLayer(name, in, out, new int[] { 3, 3 }, stride, pad, bias);
	}

	/**
	 * Convolutional layer with 5x5 kernel
	 * 
	 * @param name   Layer name
	 * @param in     Number of inputs - usually the previous layer size
	 * @param out    Number of outputs
	 * @param stride Stride dimensions new int[]{x, y}
	 * @param pad    Padding dimensions new int[]{x, y}
	 * @param bias   Bias value
	 */
	public static ConvolutionLayer convolutionalLayer5x5(String name, int in, int out, int[] stride, int[] pad,
			double bias) {
		return convolutionalLayer(name, in, out, new int[] { 5, 5 }, stride, pad, bias);
	}

//	Pooling layers

	/**
	 * Max pooling layer
	 * 
	 * @param name   Layer name
	 * @param kernel Kernel dimensions new int[]{x, y}
	 * @param stride Stride dimensions new int[]{x, y}
	 */
	public static SubsamplingLayer maxPoolLayer(String name, int[] kernel, int[] stride) {
		return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
	}

//	Fully connected layers

	/**
	 * Fully connected layer <br />
	 * 
	 * @param name    Layer name
	 * @param out     Number of outputs
	 * @param dropOut Dropout value
	 * @param dist    Distribution
	 * @param bias    Bias value
	 */
	public static DenseLayer fullyConnectedLayer(String name, int out, double dropOut, Distribution dist, double bias) {
		return new DenseLayer.Builder().name(name).nOut(out).dropOut(dropOut).dist(dist).biasInit(bias).build();
	}

}
