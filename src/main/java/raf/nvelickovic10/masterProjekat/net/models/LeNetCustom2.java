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

public class LeNetCustom2 extends Net {

	public LeNetCustom2(int numberOfLabels) {
		super(LeNetCustom2.class.getSimpleName(), numberOfLabels);
	}
	
	public LeNetCustom2(MultiLayerNetwork model) {
		super(LeNetCustom2.class.getSimpleName(), model, 0);
	}

	@Override
	public void build() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(AppConfig.seed).l2(0.001)
				.activation(Activation.RELU).weightInit(WeightInit.XAVIER_UNIFORM).updater(new Nesterovs(0.002, 0.01)).list()
				.layer(0,
						LayerFactory.convolutionalLayer("cnn1", AppConfig.channels, 30, new int[] { 3, 3 },
								new int[] { 1, 1 }, new int[] { 1, 1 }, 0))
				.layer(1,
						LayerFactory.convolutionalLayer("cnn2", 0, 60, new int[] { 3, 3 },
								new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(2, LayerFactory.maxPoolLayer("maxpool1", new int[] { 2, 2 }, new int[] { 1, 1 }))
				.layer(3,
						LayerFactory.convolutionalLayer("cnn3", 0, 100, new int[] { 3, 3 },
								new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(4,
						LayerFactory.convolutionalLayer("cnn4", 0, 100, new int[] { 3, 3 },
								new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(5, LayerFactory.maxPoolLayer("maxpool2", new int[] { 2, 2 }, new int[] { 2, 2 }))
				.layer(6,
						LayerFactory.convolutionalLayer("cnn5", 0, 40, new int[] { 3, 3 },
								new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(7, LayerFactory.maxPoolLayer("maxpool3", new int[] { 2, 2 }, new int[] { 1, 1 }))
				
				.layer(8, new DenseLayer.Builder().nOut(800).build())
				.layer(9, new DenseLayer.Builder().nOut(400).build())
				.layer(10, new DenseLayer.Builder().nOut(200).build())
				.layer(11,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.nOut(this.numberOfLabels).activation(Activation.SOFTMAX).build())
				.backprop(true).pretrain(false)
				.setInputType(InputType.convolutional(AppConfig.height, AppConfig.width, AppConfig.channels)).build();

		this.model = new MultiLayerNetwork(conf);
	}
}
