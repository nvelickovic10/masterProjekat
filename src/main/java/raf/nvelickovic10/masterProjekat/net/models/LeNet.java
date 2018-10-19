package raf.nvelickovic10.masterProjekat.net.models;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import raf.nvelickovic10.masterProjekat.factory.LayerFactory;
import raf.nvelickovic10.masterProjekat.net.Net;
import raf.nvelickovic10.masterProjekat.util.AppConfig;

public class LeNet extends Net {

	public LeNet(int numberOfLabels) {
		super(LeNet.class.getSimpleName(), numberOfLabels);
	}

	@Override
	public void build() {
		/**
		 * Revisde Lenet Model approach developed by ramgo2 achieves slightly above
		 * random <br/>
		 * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
		 **/
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(AppConfig.seed).l2(0.005)
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER).updater(new Nesterovs(0.0001, 0.9)).list()
				.layer(0,
						LayerFactory.convolutionalLayer("cnn1", AppConfig.channels, 50, new int[] { 5, 5 },
								new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(1, LayerFactory.maxPoolLayer("maxpool1", new int[] { 2, 2 }, new int[] { 2, 2 }))
				.layer(2, LayerFactory.convolutionalLayer5x5("cnn2", 0, 100, new int[] { 5, 5 }, new int[] { 1, 1 }, 0))
				.layer(3, LayerFactory.maxPoolLayer("maxpool2", new int[] { 2, 2 }, new int[] { 2, 2 }))
				.layer(4, new DenseLayer.Builder().nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.nOut(this.numberOfLabels).activation(Activation.SOFTMAX).build())
				.backprop(true).pretrain(false)
				.setInputType(InputType.convolutional(AppConfig.height, AppConfig.width, AppConfig.channels)).build();

		this.model = new MultiLayerNetwork(conf);
	}
}
